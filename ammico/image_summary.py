from ammico.utils import AnalysisMethod, AnalysisType
from ammico.inference import InferenceModel

import os
from PIL import Image
import warnings

from typing import List, Optional, Union, Dict, Any, Tuple
from collections.abc import Sequence as _Sequence


class ImageSummaryDetector(AnalysisMethod):
    token_prompt_config = {
        "default": {
            "summary": {"prompt": "Describe this image.", "max_new_tokens": 256},
            "questions": {"prompt": "", "max_new_tokens": 128},
        },
        "concise": {
            "summary": {
                "prompt": "Describe this image in one concise caption.",
                "max_new_tokens": 64,
            },
            "questions": {"prompt": "Answer concisely: ", "max_new_tokens": 128},
        },
    }
    MAX_QUESTIONS_PER_IMAGE = 32
    KEYS_BATCH_SIZE = 16

    def __init__(
        self,
        summary_model: InferenceModel,
        subdict: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Class for analysing images using an externally hosted vision-language model.
        It provides methods for generating captions and answering questions about images.

        Args:
            summary_model (InferenceModel): An InferenceModel instance used for analysis.
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

    def _load_images(
        self, entry: Optional[Dict[str, Any]] = None
    ) -> Union[Image.Image, List[Image.Image]]:
        """Load the image(s) for an entry as a PIL image or list of PIL images."""
        filename = entry.get("filename") if entry else None
        if filename is None:
            raise ValueError("entry must contain key 'filename'")

        if isinstance(filename, (str, os.PathLike, Image.Image)):
            return self._load_pil_if_needed(filename)
        elif self._is_sequence_but_not_str(filename):
            return [self._load_pil_if_needed(i) for i in filename]
        else:
            raise ValueError(
                "Unsupported 'filename' entry: expected path, PIL.Image, or sequence."
            )

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

    @staticmethod
    def _entry_label(entry: Optional[Dict[str, Any]]) -> str:
        """Human-readable identifier for an image entry, for log/warning messages."""
        if entry:
            filename = entry.get("filename")
            if filename:
                return str(filename)
        return "<unknown image>"

    def analyse_image(
        self,
        entry: dict,
        analysis_type: Union[str, AnalysisType] = AnalysisType.SUMMARY_AND_QUESTIONS,
        list_of_questions: Optional[List[str]] = None,
        max_questions_per_image: int = MAX_QUESTIONS_PER_IMAGE,
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
            self.subdict["caption"] = self._safe_generate_caption(
                entry, is_concise_summary
            )

        if is_questions:
            self.subdict["vqa"] = self._safe_answer_questions(
                list_of_questions, entry, is_concise_answer
            )

        return self.subdict

    def _safe_generate_caption(
        self, entry: Dict[str, Any], is_concise_summary: bool
    ) -> str:
        """Generate a caption for one entry, never raising.

        Returns an empty string on failure and emits an actionable warning so a
        single problematic image does not abort a whole batch.
        """
        label = self._entry_label(entry)
        try:
            caps = self.generate_caption(
                entry,
                num_return_sequences=1,
                is_concise_summary=is_concise_summary,
            )
        except Exception as e:
            warnings.warn(
                f"Caption generation failed for {label}: {e}. "
                "Skipping this image and continuing. Check that the inference "
                "endpoint is reachable and the configured model id "
                f"('{getattr(self.summary_model, 'model_id', 'unknown')}') is "
                "available there."
            )
            return ""
        caption = (caps[0] if caps else "").strip()
        if not caption:
            warnings.warn(
                f"No caption produced for {label}: the model returned an empty "
                "response. Continuing with an empty caption."
            )
        return caption

    def _safe_answer_questions(
        self,
        list_of_questions: List[str],
        entry: Dict[str, Any],
        is_concise_answer: bool,
    ) -> List[str]:
        """Answer VQA questions for one entry, never raising.

        Returns an empty list on failure and emits an actionable warning so a
        single problematic image does not abort a whole batch.
        """
        label = self._entry_label(entry)
        try:
            return self.answer_questions(list_of_questions, entry, is_concise_answer)
        except Exception as e:
            warnings.warn(
                f"VQA failed for {label}: {e}. Skipping this image and continuing. "
                "Check that the inference endpoint is reachable and the configured "
                f"model id ('{getattr(self.summary_model, 'model_id', 'unknown')}') "
                "is available there."
            )
            return []

    def analyse_images_from_dict(
        self,
        analysis_type: Union[AnalysisType, str] = AnalysisType.SUMMARY_AND_QUESTIONS,
        list_of_questions: Optional[List[str]] = None,
        max_questions_per_image: int = MAX_QUESTIONS_PER_IMAGE,
        keys_batch_size: int = KEYS_BATCH_SIZE,
        is_concise_summary: bool = True,
        is_concise_answer: bool = True,
    ) -> Dict[str, dict]:
        """
        Analyse image with  model.

        Args:
            analysis_type (str): type of the analysis.
            list_of_questions (list[str]): list of questions.
            max_questions_per_image (int): maximum number of questions per image.
                We recommend to keep it low to avoid long processing times and high memory usage.
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
                    entry["caption"] = self._safe_generate_caption(
                        entry, is_concise_summary
                    )

                if is_questions:
                    entry["vqa"] = self._safe_answer_questions(
                        list_of_questions, entry, is_concise_answer
                    )

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
        prompt = self.token_prompt_config[
            "concise" if is_concise_summary else "default"
        ]["summary"]["prompt"]
        max_new_tokens = self.token_prompt_config[
            "concise" if is_concise_summary else "default"
        ]["summary"]["max_new_tokens"]

        images = self._load_images(entry)
        messages = self.summary_model.build_messages(images, prompt)
        return self.summary_model.chat(
            messages, max_new_tokens=max_new_tokens, n=num_return_sequences
        )

    def _clean_list_of_questions(
        self, list_of_questions: list[str], prompt: str
    ) -> list[str]:
        """Clean the list of questions to contain correctly formatted strings."""
        # remove all None or empty questions
        list_of_questions = [i for i in list_of_questions if i and i.strip()]
        # ensure each question ends with a question mark
        list_of_questions = [
            i.strip() + "?" if not i.strip().endswith("?") else i.strip()
            for i in list_of_questions
        ]
        # ensure each question starts with the prompt
        list_of_questions = [
            i if i.lower().startswith(prompt.lower()) else prompt + i
            for i in list_of_questions
        ]
        return list_of_questions

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
        prompt = self.token_prompt_config[
            "concise" if is_concise_answer else "default"
        ]["questions"]["prompt"]
        max_new_tokens = self.token_prompt_config[
            "concise" if is_concise_answer else "default"
        ]["questions"]["max_new_tokens"]

        list_of_questions = self._clean_list_of_questions(list_of_questions, prompt)

        images = self._load_images(entry)
        messages_batch = [
            self.summary_model.build_messages(images, q) for q in list_of_questions
        ]
        answers = self.summary_model.chat_batch(
            messages_batch, max_new_tokens=max_new_tokens
        )

        if len(answers) != len(list_of_questions):
            raise ValueError(
                f"Expected {len(list_of_questions)} answers, but got {len(answers)}, try varying amount of questions"
            )

        return answers
