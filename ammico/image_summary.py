from ammico.utils import AnalysisMethod, AnalysisType
from ammico.model import MultimodalSummaryModel

import os
import torch
from PIL import Image
import warnings

from typing import List, Optional, Union, Dict, Any, Tuple
from collections.abc import Sequence as _Sequence
from transformers import GenerationConfig
from qwen_vl_utils import process_vision_info


class ImageSummaryDetector(AnalysisMethod):
    def __init__(
        self,
        summary_model: MultimodalSummaryModel,
        subdict: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Class for analysing images using QWEN-2.5-VL model.
        It provides methods for generating captions and answering questions about images.

        Args:
            summary_model ([type], optional): An instance of MultimodalSummaryModel to be used for analysis.
            subdict (dict, optional): Dictionary containing the image to be analysed. Defaults to {}.

        Returns:
            None.
        """
        if subdict is None:
            subdict = {}

        super().__init__(subdict)
        self.summary_model = summary_model

    def _load_pil_if_needed(
        self, filename: Union[str, os.PathLike, Image.Image]
    ) -> Image.Image:
        if isinstance(filename, (str, os.PathLike)):
            return Image.open(filename).convert("RGB")
        elif isinstance(filename, Image.Image):
            return filename.convert("RGB")
        else:
            raise ValueError("filename must be a path or PIL.Image")

    @staticmethod
    def _is_sequence_but_not_str(obj: Any) -> bool:
        """True for sequence-like but not a string/bytes/PIL.Image."""
        return isinstance(obj, _Sequence) and not isinstance(
            obj, (str, bytes, Image.Image)
        )

    def _prepare_inputs(
        self, list_of_questions: list[str], entry: Optional[Dict[str, Any]] = None
    ) -> Dict[str, torch.Tensor]:
        filename = entry.get("filename")
        if filename is None:
            raise ValueError("entry must contain key 'filename'")

        if isinstance(filename, (str, os.PathLike, Image.Image)):
            images_context = self._load_pil_if_needed(filename)
        elif self._is_sequence_but_not_str(filename):
            images_context = [self._load_pil_if_needed(i) for i in filename]
        else:
            raise ValueError(
                "Unsupported 'filename' entry: expected path, PIL.Image, or sequence."
            )

        images_only_messages = [
            {
                "role": "user",
                "content": [
                    *(
                        [{"type": "image", "image": img} for img in images_context]
                        if isinstance(images_context, list)
                        else [{"type": "image", "image": images_context}]
                    )
                ],
            }
        ]

        try:
            image_inputs, _ = process_vision_info(images_only_messages)
        except Exception as e:
            raise RuntimeError(f"Image processing failed: {e}")

        texts: List[str] = []
        for q in list_of_questions:
            messages = [
                {
                    "role": "user",
                    "content": [
                        *(
                            [
                                {"type": "image", "image": image}
                                for image in images_context
                            ]
                            if isinstance(images_context, list)
                            else [{"type": "image", "image": images_context}]
                        ),
                        {"type": "text", "text": q},
                    ],
                }
            ]
            text = self.summary_model.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)

        images_batch = [image_inputs] * len(texts)
        inputs = self.summary_model.processor(
            text=texts,
            images=images_batch,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.summary_model.device) for k, v in inputs.items()}

        return inputs

    def _validate_analysis_type(
        self,
        analysis_type: Union["AnalysisType", str],
        list_of_questions: Optional[List[str]],
        max_questions_per_image: int,
    ) -> Tuple[str, List[str], bool, bool]:
        if isinstance(analysis_type, AnalysisType):
            analysis_type = analysis_type.value

        allowed = {"summary", "questions", "summary_and_questions"}
        if analysis_type not in allowed:
            raise ValueError(f"analysis_type must be one of {allowed}")

        if list_of_questions is None:
            list_of_questions = [
                "Are there people in the image?",
                "What is this picture about?",
            ]

        if analysis_type in ("questions", "summary_and_questions"):
            if len(list_of_questions) > max_questions_per_image:
                raise ValueError(
                    f"Number of questions per image ({len(list_of_questions)}) exceeds safety cap ({max_questions_per_image}). Reduce questions or increase max_questions_per_image."
                )

        is_summary = analysis_type in ("summary", "summary_and_questions")
        is_questions = analysis_type in ("questions", "summary_and_questions")

        return analysis_type, list_of_questions, is_summary, is_questions

    def analyse_image(
        self,
        entry: dict,
        analysis_type: Union[str, AnalysisType] = AnalysisType.SUMMARY_AND_QUESTIONS,
        list_of_questions: Optional[List[str]] = None,
        max_questions_per_image: int = 32,
        is_concise_summary: bool = True,
        is_concise_answer: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyse a single image entry. Returns dict with keys depending on analysis_type:
            - 'caption' (str) if summary requested
            - 'vqa' (dict) if questions requested
        """
        self.subdict = entry
        analysis_type, list_of_questions, is_summary, is_questions = (
            self._validate_analysis_type(
                analysis_type, list_of_questions, max_questions_per_image
            )
        )

        if is_summary:
            try:
                caps = self.generate_caption(
                    entry,
                    num_return_sequences=1,
                    is_concise_summary=is_concise_summary,
                )
                self.subdict["caption"] = caps[0] if caps else ""
            except Exception as e:
                warnings.warn(f"Caption generation failed: {e}")

        if is_questions:
            try:
                vqa_map = self.answer_questions(
                    list_of_questions, entry, is_concise_answer
                )
                self.subdict["vqa"] = vqa_map
            except Exception as e:
                warnings.warn(f"VQA failed: {e}")

        return self.subdict

    def analyse_images_from_dict(
        self,
        analysis_type: Union[AnalysisType, str] = AnalysisType.SUMMARY_AND_QUESTIONS,
        list_of_questions: Optional[List[str]] = None,
        max_questions_per_image: int = 32,
        keys_batch_size: int = 16,
        is_concise_summary: bool = True,
        is_concise_answer: bool = True,
    ) -> Dict[str, dict]:
        """
        Analyse image with  model.

        Args:
            analysis_type (str): type of the analysis.
            list_of_questions (list[str]): list of questions.
            max_questions_per_image (int): maximum number of questions per image. We recommend to keep it low to avoid long processing times and high memory usage.
            keys_batch_size (int): number of images to process in a batch.
            is_concise_summary (bool): whether to generate concise summary.
            is_concise_answer (bool): whether to generate concise answers.
        Returns:
            self.subdict (dict): dictionary with analysis results.
        """
        # TODO: add option to ask multiple questions per image as one batch.
        analysis_type, list_of_questions, is_summary, is_questions = (
            self._validate_analysis_type(
                analysis_type, list_of_questions, max_questions_per_image
            )
        )

        keys = list(self.subdict.keys())
        for batch_start in range(0, len(keys), keys_batch_size):
            batch_keys = keys[batch_start : batch_start + keys_batch_size]
            for key in batch_keys:
                entry = self.subdict[key]
                if is_summary:
                    try:
                        caps = self.generate_caption(
                            entry,
                            num_return_sequences=1,
                            is_concise_summary=is_concise_summary,
                        )
                        entry["caption"] = caps[0] if caps else ""
                    except Exception as e:
                        warnings.warn(f"Caption generation failed: {e}")

                if is_questions:
                    try:
                        vqa_map = self.answer_questions(
                            list_of_questions, entry, is_concise_answer
                        )
                        entry["vqa"] = vqa_map
                    except Exception as e:
                        warnings.warn(f"VQA failed: {e}")

                self.subdict[key] = entry
        return self.subdict

    def generate_caption(
        self,
        entry: Optional[Dict[str, Any]] = None,
        num_return_sequences: int = 1,
        is_concise_summary: bool = True,
    ) -> List[str]:
        """
        Create caption for image. Depending on is_concise_summary it will be either concise or detailed.

        Args:
            entry (dict): dictionary containing the image to be captioned.
            num_return_sequences (int): number of captions to generate.
            is_concise_summary (bool): whether to generate concise summary.

        Returns:
            results (list[str]): list of generated captions.
        """
        if is_concise_summary:
            prompt = ["Describe this image in one concise caption."]
            max_new_tokens = 64
        else:
            prompt = ["Describe this image."]
            max_new_tokens = 256
        inputs = self._prepare_inputs(prompt, entry)

        gen_conf = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_return_sequences=num_return_sequences,
        )

        with torch.inference_mode():
            try:
                if self.summary_model.device == "cuda":
                    with torch.cuda.amp.autocast(enabled=True):
                        generated_ids = self.summary_model.model.generate(
                            **inputs, generation_config=gen_conf
                        )
                else:
                    generated_ids = self.summary_model.model.generate(
                        **inputs, generation_config=gen_conf
                    )
            except RuntimeError as e:
                warnings.warn(
                    f"Retry without autocast failed: {e}. Attempting cudnn-disabled retry."
                )
                cudnn_was_enabled = (
                    torch.backends.cudnn.is_available() and torch.backends.cudnn.enabled
                )
                if cudnn_was_enabled:
                    torch.backends.cudnn.enabled = False
                try:
                    generated_ids = self.summary_model.model.generate(
                        **inputs, generation_config=gen_conf
                    )
                except Exception as retry_error:
                    raise RuntimeError(
                        f"Failed to generate ids after retry: {retry_error}"
                    ) from retry_error
                finally:
                    if cudnn_was_enabled:
                        torch.backends.cudnn.enabled = True

        decoded = None
        if "input_ids" in inputs:
            in_ids = inputs["input_ids"]
            trimmed = [
                out_ids[len(inp_ids) :]
                for inp_ids, out_ids in zip(in_ids, generated_ids)
            ]
            decoded = self.summary_model.tokenizer.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        else:
            decoded = self.summary_model.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        results = [d.strip() for d in decoded]
        return results

    def answer_questions(
        self,
        list_of_questions: list[str],
        entry: Optional[Dict[str, Any]] = None,
        is_concise_answer: bool = True,
    ) -> List[str]:
        """
        Create answers for list of questions about image.
        Args:
            list_of_questions (list[str]): list of questions.
            entry (dict): dictionary containing the image to be captioned.
            is_concise_answer (bool): whether to generate concise answers.
        Returns:
            answers (list[str]): list of answers.
        """
        if is_concise_answer:
            gen_conf = GenerationConfig(max_new_tokens=64, do_sample=False)
            for i in range(len(list_of_questions)):
                if not list_of_questions[i].strip().endswith("?"):
                    list_of_questions[i] = list_of_questions[i].strip() + "?"
                if not list_of_questions[i].lower().startswith("answer concisely"):
                    list_of_questions[i] = "Answer concisely: " + list_of_questions[i]
        else:
            gen_conf = GenerationConfig(max_new_tokens=128, do_sample=False)

        question_chunk_size = 8
        answers: List[str] = []
        n = len(list_of_questions)
        for i in range(0, n, question_chunk_size):
            chunk = list_of_questions[i : i + question_chunk_size]
            inputs = self._prepare_inputs(chunk, entry)
            with torch.inference_mode():
                if self.summary_model.device == "cuda":
                    with torch.cuda.amp.autocast(enabled=True):
                        out_ids = self.summary_model.model.generate(
                            **inputs, generation_config=gen_conf
                        )
                else:
                    out_ids = self.summary_model.model.generate(
                        **inputs, generation_config=gen_conf
                    )

            if "input_ids" in inputs:
                in_ids = inputs["input_ids"]
                trimmed_batch = [
                    out_row[len(inp_row) :] for inp_row, out_row in zip(in_ids, out_ids)
                ]
                decoded = self.summary_model.tokenizer.batch_decode(
                    trimmed_batch,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            else:
                decoded = self.summary_model.tokenizer.batch_decode(
                    out_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

            answers.extend([d.strip() for d in decoded])

        if len(answers) != len(list_of_questions):
            raise ValueError(
                f"Expected {len(list_of_questions)} answers, but got {len(answers)}, try vary amount of questions"
            )

        return answers
