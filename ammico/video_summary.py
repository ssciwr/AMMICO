import re
import math
import torch
import warnings
from PIL import Image
from torchcodec.decoders import VideoDecoder

from ammico.model import MultimodalSummaryModel
from ammico.utils import (
    AnalysisMethod,
    AnalysisType,
    _categorize_outputs,
    _strip_prompt_prefix_literal,
)

from typing import List, Dict, Any, Generator, Tuple, Union, Optional
from transformers import GenerationConfig


class VideoSummaryDetector(AnalysisMethod):
    MAX_SAMPLES_CAP = 1000  # safety cap for total extracted frames

    def __init__(
        self,
        summary_model: MultimodalSummaryModel,
        subdict: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Class for analysing videos using QWEN-2.5-VL model.
        It provides methods for generating captions and answering questions about videos.

        Args:
            summary_model ([type], optional): An instance of MultimodalSummaryModel to be used for analysis.
            subdict (dict, optional): Dictionary containing the video to be analysed. Defaults to {}.

        Returns:
            None.
        """
        if subdict is None:
            subdict = {}

        super().__init__(subdict)
        self.summary_model = summary_model

    def _frame_batch_generator(
        self,
        timestamps: torch.Tensor,
        batch_size: int,
        video_decoder: VideoDecoder,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """
        Yield batches of (frames, timestamps) for given frame indices.
        - frames are returned as a torch.Tensor with shape (B, C, H, W).
        - timestamps is a 1D torch.Tensor with B elements.
        """
        total = int(timestamps.numel())

        for start in range(0, total, batch_size):
            batch_secs = timestamps[start : start + batch_size].tolist()
            fb = video_decoder.get_frames_played_at(batch_secs)
            frames = fb.data

            if not frames.is_contiguous():
                frames = frames.contiguous()
            pts = fb.pts_seconds
            pts_out = (
                pts.cpu().to(dtype=torch.float32)
                if isinstance(pts, torch.Tensor)
                else torch.tensor(pts, dtype=torch.float32)
            )
            yield frames, pts_out

    def _extract_video_frames(
        self,
        entry: Dict[str, Any],
        frame_rate_per_second: float = 2.0,
        batch_size: int = 32,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """
        Extract frames from a video at a specified frame rate and return them as a generator of batches.
        Args:
            filename (Union[str, os.PathLike]): Path to the video file.
            frame_rate_per_second (float, optional): Frame extraction rate in frames per second. Default is 2.
            batch_size (int, optional): Number of frames to include in each batch. Default is 32.
        Returns:
            Generator[Tuple[torch.Tensor, torch.Tensor], None, None]: A generator yielding tuples of
            (frames, timestamps), where frames is a tensor of shape (B, C, H, W) and timestamps is a 1D tensor of length B.
        """

        filename = entry.get("filename")
        if not filename:
            raise ValueError("entry must contain key 'filename'")

        video_decoder = VideoDecoder(filename)
        meta = video_decoder.metadata

        video_fps = getattr(meta, "average_fps", None)
        if video_fps is None or not (
            isinstance(video_fps, (int, float)) and video_fps > 0
        ):
            video_fps = 30.0

        begin_stream_seconds = getattr(meta, "begin_stream_seconds", None)
        end_stream_seconds = getattr(meta, "end_stream_seconds", None)
        nframes = len(video_decoder)
        if getattr(meta, "duration_seconds", None) is not None:
            duration = float(meta.duration_seconds)
        elif begin_stream_seconds is not None and end_stream_seconds is not None:
            duration = float(end_stream_seconds) - float(begin_stream_seconds)
        elif nframes:
            duration = float(nframes) / float(video_fps)
        else:
            duration = 0.0

        if frame_rate_per_second <= 0:
            raise ValueError("frame_rate_per_second must be > 0")

        n_samples = max(1, int(math.floor(duration * frame_rate_per_second)))
        n_samples = min(n_samples, self.MAX_SAMPLES_CAP)

        if begin_stream_seconds is not None and end_stream_seconds is not None:
            sample_times = torch.linspace(
                float(begin_stream_seconds), float(end_stream_seconds), steps=n_samples
            )
            if sample_times.numel() > 1:
                sample_times = torch.clamp(
                    sample_times,
                    min=float(begin_stream_seconds),
                    max=float(end_stream_seconds) - 1e-6,
                )
        else:
            sample_times = torch.linspace(0.0, max(0.0, duration), steps=n_samples)

        sample_times = sample_times.to(dtype=torch.float32, device="cpu")
        generator = self._frame_batch_generator(sample_times, batch_size, video_decoder)

        return generator

    def _decode_trimmed_outputs(
        self,
        generated_ids: torch.Tensor,
        inputs: Dict[str, torch.Tensor],
        tokenizer,
        prompt_texts: List[str],
    ) -> List[str]:
        """
        Trim prompt tokens using attention_mask/input_ids when available and decode to strings.
        Then remove any literal prompt prefix using prompt_texts (one per batch element).
        """

        batch_size = generated_ids.shape[0]

        if "input_ids" in inputs:
            token_for_padding = (
                tokenizer.pad_token_id
                if getattr(tokenizer, "pad_token_id", None) is not None
                else getattr(tokenizer, "eos_token_id", None)
            )
            if token_for_padding is None:
                lengths = [int(inputs["input_ids"].shape[1])] * batch_size
            else:
                lengths = inputs["input_ids"].ne(token_for_padding).sum(dim=1).tolist()
        else:
            lengths = [0] * batch_size

        trimmed_ids = []
        for i in range(batch_size):
            out_ids = generated_ids[i]
            in_len = int(lengths[i]) if i < len(lengths) else 0
            if out_ids.size(0) > in_len:
                t = out_ids[in_len:]
            else:
                t = out_ids.new_empty((0,), dtype=out_ids.dtype)
            t_cpu = t.to("cpu")
            trimmed_ids.append(t_cpu.tolist())

        decoded = tokenizer.batch_decode(
            trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        decoded_results = []
        for ptext, raw in zip(prompt_texts, decoded):
            cleaned = _strip_prompt_prefix_literal(raw, ptext)
            decoded_results.append(cleaned)
        return decoded_results

    def _generate_from_processor_inputs(
        self,
        processor_inputs: Dict[str, torch.Tensor],
        prompt_texts: List[str],
        tokenizer,
    ):
        """
        Run model.generate on already-processed processor_inputs (tensors moved to device),
        then decode and trim prompt tokens & remove literal prompt prefixes using prompt_texts.
        """
        gen_conf = GenerationConfig(
            max_new_tokens=64,
            do_sample=False,
            num_return_sequences=1,
        )

        for k, v in list(processor_inputs.items()):
            if isinstance(v, torch.Tensor):
                processor_inputs[k] = v.to(self.summary_model.device)

        with torch.inference_mode():
            try:
                if self.summary_model.device == "cuda":
                    with torch.amp.autocast("cuda", enabled=True):
                        generated_ids = self.summary_model.model.generate(
                            **processor_inputs, generation_config=gen_conf
                        )
                else:
                    generated_ids = self.summary_model.model.generate(
                        **processor_inputs, generation_config=gen_conf
                    )
            except RuntimeError as e:
                warnings.warn(
                    f"Generation failed with error: {e}. Retrying with cuDNN disabled.",
                    RuntimeWarning,
                )
                cudnn_was_enabled = (
                    torch.backends.cudnn.is_available() and torch.backends.cudnn.enabled
                )
                if cudnn_was_enabled:
                    torch.backends.cudnn.enabled = False
                try:
                    generated_ids = self.summary_model.model.generate(
                        **processor_inputs, generation_config=gen_conf
                    )
                except Exception as retry_error:
                    raise RuntimeError(
                        f"Failed to generate ids after retry: {retry_error}"
                    ) from retry_error
                finally:
                    if cudnn_was_enabled:
                        torch.backends.cudnn.enabled = True

        decoded = self._decode_trimmed_outputs(
            generated_ids, processor_inputs, tokenizer, prompt_texts
        )
        return decoded

    def _tensor_batch_to_pil_list(self, batch: torch.Tensor) -> List[Image.Image]:
        """
        Convert a uint8 torch tensor batch (B, C, H, W) on CPU to list of PIL images (RGB).
        The conversion is done on CPU and returns PIL.Image objects.
        """
        if batch.device.type != "cpu":
            batch = batch.to("cpu")

        batch = batch.contiguous()
        if batch.dtype.is_floating_point:
            batch = (batch.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        elif batch.dtype != torch.uint8:
            batch = batch.to(torch.uint8)
        pil_list: List[Image.Image] = []
        for frame in batch:
            arr = frame.permute(1, 2, 0).numpy()
            pil_list.append(Image.fromarray(arr))
        return pil_list

    def make_captions_from_extracted_frames(
        self,
        extracted_video_gen: Generator[Tuple[torch.Tensor, torch.Tensor], None, None],
        list_of_questions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate captions for all extracted frames and then produce a concise summary of the video.
        Args:
            extracted_video_dict (Dict[str, Any]): Dictionary containing the frame generator and number of frames.
            summary_instruction (str, optional): Instruction for summarizing the captions. Defaults to a concise paragraph.
        Returns:
            Dict[str, Any]: A dictionary containing the list of captions with timestamps and the final summary.
        """

        caption_instruction = "Describe this image in one concise caption."
        include_questions = bool(list_of_questions)
        if include_questions:
            q_block = "\n".join(
                [f"{i + 1}. {q.strip()}" for i, q in enumerate(list_of_questions)]
            )
            caption_instruction += (
                " In addition to the concise caption, also answer the following questions based ONLY on the image. Answers must be very brief and concise."
                " Produce exactly two labeled sections: \n\n"
                "Summary: <concise summary>\n\n"
                "VQA Answers: \n1. <answer to question 1>\n2. <answer to question 2>\n etc."
                "\nReturn only those two sections for each image (do not add extra commentary)."
                "\nIf the answer cannot be determined based on the provided answer blocks,"
                ' reply with the line "The answer cannot be determined based on the information provided."'
                f"\n\nQuestions:\n{q_block}"
            )
        collected: List[Tuple[float, str]] = []
        proc = self.summary_model.processor
        try:
            for batch_frames, batch_times in extracted_video_gen:
                pil_list = self._tensor_batch_to_pil_list(batch_frames.cpu())

                prompt_texts = []
                for p in pil_list:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": p},
                                {"type": "text", "text": caption_instruction},
                            ],
                        }
                    ]

                    prompt_text = proc.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    prompt_texts.append(prompt_text)

                processor_inputs = proc(
                    text=prompt_texts,
                    images=pil_list,
                    return_tensors="pt",
                    padding=True,
                )
                captions = self._generate_from_processor_inputs(
                    processor_inputs,
                    prompt_texts,
                    self.summary_model.tokenizer,
                )

                if isinstance(batch_times, torch.Tensor):
                    batch_times_list = batch_times.cpu().tolist()
                else:
                    batch_times_list = list(batch_times)
                for t, c in zip(batch_times_list, captions):
                    collected.append((float(t), c))
        finally:
            try:
                extracted_video_gen.close()
            except Exception:
                warnings.warn("Failed to close video frame generator.", RuntimeWarning)

        collected.sort(key=lambda x: x[0])
        bullets_summary, bullets_vqa = _categorize_outputs(collected, include_questions)

        return {
            "summary_bullets": bullets_summary,
            "vqa_bullets": bullets_vqa,
        }  # TODO consider taking out time stamps from the returned structure

    def final_summary(self, summary_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Produce a concise summary of the video, based on generated captions for all extracted frames.
        Args:
            summary_dict (Dict[str, Any]): Dictionary containing captions for the frames.
        Returns:
            Dict[str, Any]: A dictionary containing the list of captions with timestamps and the final summary.
        """
        summary_instruction = "Analyze the following captions from multiple frames of the same video and summarize the overall content of the video in one concise paragraph (1-3 sentences). Focus on the key themes, actions, or events across the video, not just the individual frames."
        proc = self.summary_model.processor

        bullets = summary_dict.get("summary_bullets", [])
        if not bullets:
            raise ValueError("No captions available for summary generation.")

        combined_captions_text = "\n".join(bullets)
        summary_user_text = summary_instruction + "\n\n" + combined_captions_text + "\n"

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": summary_user_text}],
            }
        ]

        summary_prompt_text = proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        summary_inputs = proc(
            text=[summary_prompt_text], return_tensors="pt", padding=True
        )

        summary_inputs = {
            k: v.to(self.summary_model.device) if isinstance(v, torch.Tensor) else v
            for k, v in summary_inputs.items()
        }
        final_summary_list = self._generate_from_processor_inputs(
            summary_inputs,
            [summary_prompt_text],
            self.summary_model.tokenizer,
        )
        final_summary = final_summary_list[0].strip() if final_summary_list else ""

        return {
            "summary": final_summary,
        }

    def final_answers(
        self,
        answers_dict: Dict[str, Any],
        list_of_questions: List[str],
    ) -> Dict[str, Any]:
        """
        Answer the list of questions for the video based on the VQA bullets from the frames.
        Args:
            answers_dict (Dict[str, Any]): Dictionary containing the VQA bullets.
        Returns:
            Dict[str, Any]: A dictionary containing the list of answers to the questions.
        """
        vqa_bullets = answers_dict.get("vqa_bullets", [])
        if not vqa_bullets:
            raise ValueError(
                "No VQA bullets generated for single frames available for answering questions."
            )

        include_questions = bool(list_of_questions)
        if include_questions:
            q_block = "\n".join(
                [f"{i + 1}. {q.strip()}" for i, q in enumerate(list_of_questions)]
            )
            prompt = (
                "You are provided with a set of short VQA-captions, each of which is a block of short answers"
                " extracted from individual frames of the same video.\n\n"
                "VQA-captions (use ONLY these to answer):\n"
                f"{vqa_bullets}\n\n"
                "Answer the following questions briefly, based ONLY on the lists of answers provided above. The VQA-captions above contain answers"
                " to the same questions you are about to answer. If the answer cannot be determined based on the provided answer blocks,"
                ' reply with the line "The answer cannot be determined based on the information provided."'
                "Questions:\n"
                f"{q_block}\n\n"
                "Produce an ordered list with answers in the same order as the questions. You must have this structure of your output: "
                "Answers: \n1. <answer to question 1>\n2. <answer to question 2>\n etc."
                "Return ONLY the ordered list with answers and NOTHING else — no commentary, no explanation, no surrounding markdown."
            )
        else:
            raise ValueError(
                "list_of_questions must be provided for making final answers."
            )

        proc = self.summary_model.processor
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        final_vqa_prompt_text = proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        final_vqa_inputs = proc(
            text=[final_vqa_prompt_text], return_tensors="pt", padding=True
        )
        final_vqa_inputs = {
            k: v.to(self.summary_model.device) if isinstance(v, torch.Tensor) else v
            for k, v in final_vqa_inputs.items()
        }

        final_vqa_list = self._generate_from_processor_inputs(
            final_vqa_inputs,
            [final_vqa_prompt_text],
            self.summary_model.tokenizer,
        )

        final_vqa_output = final_vqa_list[0].strip() if final_vqa_list else ""
        vqa_answers = []
        answer_matches = re.findall(
            r"\d+\.\s*(.*?)(?=\n\d+\.|$)", final_vqa_output, flags=re.DOTALL
        )
        for answer in answer_matches:
            vqa_answers.append(answer.strip())
        return {
            "vqa_answers": vqa_answers,
        }

    def analyse_videos_from_dict(
        self,
        analysis_type: Union[AnalysisType, str] = AnalysisType.SUMMARY,
        frame_rate_per_second: float = 2.0,
        list_of_questions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Analyse the video specified in self.subdict using frame extraction and captioning.
        Args:
            analysis_type (Union[AnalysisType, str], optional): Type of analysis to perform. Defaults to AnalysisType.SUMMARY.
            frame_rate_per_second (float): Frame extraction rate in frames per second. Default is 2.0.
            list_of_questions (List[str], optional): List of questions to answer about the video. Required if analysis_type includes questions.
        Returns:
            Dict[str, Any]: A dictionary containing the analysis results, including summary and answers for provided questions(if any).
        """

        all_answers = {}
        analysis_type, is_summary, is_questions = AnalysisType._validate_analysis_type(
            analysis_type, list_of_questions
        )

        for video_key, entry in self.subdict.items():
            summary_and_vqa = {}
            frames_generator = self._extract_video_frames(
                entry, frame_rate_per_second=frame_rate_per_second
            )

            answers_dict = self.make_captions_from_extracted_frames(
                frames_generator, list_of_questions=list_of_questions
            )
            # TODO: captions has to be post-processed with foreseeing audio analysis
            # TODO: captions and answers may lead to prompt, that superior model limits. Consider hierarchical approach.
            if is_summary:
                answer = self.final_summary(answers_dict)
                summary_and_vqa["summary"] = answer["summary"]
            if is_questions:
                answer = self.final_answers(answers_dict, list_of_questions)
                summary_and_vqa["vqa_answers"] = answer["vqa_answers"]

            all_answers[video_key] = summary_and_vqa

        return all_answers
