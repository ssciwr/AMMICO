import decord
import os
import re
import math
import torch
import warnings
import numpy as np
from PIL import Image

from ammico.model import MultimodalSummaryModel
from ammico.utils import AnalysisMethod, AnalysisType

from typing import List, Optional, Union, Dict, Any, Generator, Tuple
from transformers import GenerationConfig


class VideoSummaryDetector(AnalysisMethod):
    def __init__(
        self,
        summary_model: MultimodalSummaryModel,
        subdict: dict = {},
        gpu_id: int = 0,
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

        super().__init__(subdict)
        self.summary_model = summary_model
        self.gpu_id = gpu_id

    def _normalize_whitespace(self, s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    def _strip_prompt_prefix_literal(self, decoded: str, prompt: str) -> str:
        """
        Remove any literal prompt prefix from decoded text using a normalized-substring match.
        Guarantees no prompt text remains at the start of returned string (best-effort).
        """
        if not decoded:
            return ""
        if not prompt:
            return decoded.strip()

        d_norm = self._normalize_whitespace(decoded)
        p_norm = self._normalize_whitespace(prompt)

        idx = d_norm.find(p_norm)
        if idx != -1:
            running = []
            for i, ch in enumerate(decoded):
                running.append(ch if not ch.isspace() else " ")
                cur_norm = self._normalize_whitespace("".join(running))
                if cur_norm.endswith(p_norm):
                    return decoded[i + 1 :].lstrip() if i + 1 < len(decoded) else ""
        m = re.match(
            r"^(?:\s*(system|user|assistant)[:\s-]*\n?)+", decoded, flags=re.IGNORECASE
        )
        if m:
            return decoded[m.end() :].lstrip()

        return decoded.lstrip("\n\r ").lstrip(":;- ").strip()

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

        decoded_results = []
        batch_size = generated_ids.shape[0]

        if "input_ids" in inputs:
            lengths = (
                inputs["input_ids"]
                .ne(
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                )
                .sum(dim=1)
                .tolist()
            )
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
            trimmed_ids.append(t)

        decoded = tokenizer.batch_decode(
            trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        for ptext, raw in zip(prompt_texts, decoded):
            cleaned = self._strip_prompt_prefix_literal(raw, ptext)
            decoded_results.append(cleaned)
        return decoded_results

    def _generate_from_processor_inputs(
        self,
        processor_inputs: Dict[str, torch.Tensor],
        prompt_texts: List[str],
        model,
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
                processor_inputs[k] = v.to(model.device)

        with torch.inference_mode():
            try:
                if self.summary_model.device == "cuda":
                    with torch.cuda.amp.autocast(enabled=True):
                        generated_ids = self.summary_model.model.generate(
                            **processor_inputs, generation_config=gen_conf
                        )
                else:
                    generated_ids = self.summary_model.model.generate(
                        **processor_inputs, generation_config=gen_conf
                    )
            except RuntimeError as e:
                warnings.warn(
                    "Retry without autocast failed: %s. Attempting cudnn-disabled retry.",
                    e,
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
        if batch.dtype != torch.uint8:
            batch = batch.to(torch.uint8)
        pil_list: List[Image.Image] = []
        for frame in batch:
            arr = frame.permute(1, 2, 0).numpy()
            pil_list.append(Image.fromarray(arr))
        return pil_list

    def _extract_video_frames(
        self,
        video_path: Union[str, os.PathLike],
        frame_rate_per_second: float = 2,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """
        Extract frames from a video at a specified frame rate and return them as a generator of batches.
        Args:
            video_path (Union[str, os.PathLike]): Path to the video file.
            frame_rate_per_second (float, optional): Frame extraction rate in frames per second. Default is 2.
            batch_size (int, optional): Number of frames to include in each batch. Default is 32.
        Returns:
            Dict[str, Any]: A dictionary containing a generator that yields batches of frames and their timestamps
                            and the total number of extracted frames.
        """

        device = (
            torch.device("cuda") if (torch.cuda.is_available()) else torch.device("cpu")
        )
        if device == "cuda":
            ctx = decord.gpu(self.gpu_id)
        else:
            ctx = decord.cpu()
        # TODO: to support GPU version of decord: build from source to enable GPU acclerator
        # https://github.com/dmlc/decord

        vr = decord.VideoReader(video_path, ctx=ctx)
        nframes = len(vr)
        video_fps = vr.get_avg_fps()
        if video_fps is None or video_fps <= 0:
            video_fps = 30.0

        duration = nframes / float(video_fps)

        if frame_rate_per_second <= 0:
            raise ValueError("frame_rate_per_second must be > 0")

        n_samples = max(1, int(math.floor(duration * frame_rate_per_second)))
        sample_times = (
            torch.linspace(0, duration, steps=n_samples)
            if n_samples > 1
            else torch.tensor([0.0])
        )
        indices = (sample_times * video_fps).round().long()
        indices = torch.clamp(indices, 0, nframes - 1).unique(sorted=True)
        timestamps = indices.to(torch.float32) / float(video_fps)

        total_samples = indices.numel()

        def gen() -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
            for batch_start in range(0, total_samples, batch_size):
                batch_idx_tensor = indices[batch_start : batch_start + batch_size]
                batch_idx_list = [int(x.item()) for x in batch_idx_tensor]
                batch_frames_np = vr.get_batch(batch_idx_list).asnumpy()
                batch_frames = (
                    torch.from_numpy(batch_frames_np).permute(0, 3, 1, 2).contiguous()
                )
                batch_times = timestamps[batch_start : batch_start + batch_size]
                if device is not None:
                    batch_frames = batch_frames.to(device, non_blocking=True)
                    batch_times = batch_times.to(device, non_blocking=True)
                yield batch_frames, batch_times

        return {"generator": gen(), "n_frames": total_samples}

    def brute_force_summary(
        self,
        extracted_video_dict: Dict[str, Any],
        summary_instruction: str = "Summarize the following frame captions into a concise paragraph (1-3 sentences):",
    ) -> Dict[str, Any]:
        """
        Generate captions for all extracted frames and then produce a concise summary of the video.
        Args:
            extracted_video_dict (Dict[str, Any]): Dictionary containing the frame generator and number of frames.
            summary_instruction (str, optional): Instruction for summarizing the captions. Defaults to a concise paragraph.
        Returns:
            Dict[str, Any]: A dictionary containing the list of captions with timestamps and the final summary.
        """

        gen = extracted_video_dict["generator"]
        caption_instruction = "Describe this image in one concise caption."
        collected: List[Tuple[float, str]] = []

        for batch_frames, batch_times in gen:
            pil_list = self._tensor_batch_to_pil_list(batch_frames.cpu())
            proc = self.summary_model.processor

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
                try:
                    prompt_text = proc.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except TypeError:
                    prompt_text = proc.apply_chat_template(messages)
                prompt_texts.append(prompt_text)

            processor_inputs = proc(
                text=prompt_texts, images=pil_list, return_tensors="pt", padding=True
            )
            captions = self._generate_from_processor_inputs(
                processor_inputs,
                prompt_texts,
                self.summary_model,
                self.summary_model.tokenizer,
            )

            batch_times_cpu = (
                batch_times.cpu().tolist()
                if isinstance(batch_times, torch.Tensor)
                else list(batch_times)
            )
            for t, c in zip(batch_times_cpu, captions):
                collected.append((float(t), c))

        collected.sort(key=lambda x: x[0])

        MAX_CAPTIONS_FOR_SUMMARY = 200
        caps_for_summary = (
            collected[-MAX_CAPTIONS_FOR_SUMMARY:]
            if len(collected) > MAX_CAPTIONS_FOR_SUMMARY
            else collected
        )

        bullets = []
        for t, c in caps_for_summary:
            snippet = c.replace("\n", " ").strip()
            bullets.append(f"- [{t:.3f}s] {snippet}")

        combined_captions_text = "\n".join(bullets)
        summary_user_text = (
            summary_instruction
            + "\n\n"
            + combined_captions_text
            + "\n\nPlease produce a single concise paragraph."
        )

        proc = self.summary_model.processor
        if hasattr(proc, "apply_chat_template"):
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": summary_user_text}],
                }
            ]
            try:
                summary_prompt_text = proc.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except TypeError:
                summary_prompt_text = proc.apply_chat_template(messages)
            summary_inputs = proc(
                text=[summary_prompt_text], return_tensors="pt", padding=True
            )
        else:
            summary_prompt_text = summary_user_text
            summary_inputs = self.summary_model.tokenizer(
                summary_prompt_text, return_tensors="pt"
            )

        summary_inputs = {
            k: v.to(self.summary_model.device) if isinstance(v, torch.Tensor) else v
            for k, v in summary_inputs.items()
        }
        final_summary_list = self._generate_from_processor_inputs(
            summary_inputs,
            [summary_prompt_text],
            self.summary_model.model,
            self.summary_model.tokenizer,
        )
        final_summary = final_summary_list[0].strip() if final_summary_list else ""

        return {
            "captions": collected,
            "summary": final_summary,
        }

    def analyse_videos(self, frame_rate_per_second: float = 2.0) -> Dict[str, Any]:
        """
        Analyse the video specified in self.subdict using frame extraction and captioning.
        For short videos (<=50 frames at the specified frame rate), it uses brute-force captioning.
        For longer videos, it currently defaults to brute-force captioning but can be extended for more complex methods.

        Args:
            frame_rate_per_second (float): Frame extraction rate in frames per second. Default is 2.0.
        Returns:
            Dict[str, Any]: A dictionary containing the analysis results, including captions and summary.
        """

        minimal_edge_of_frames = 50
        extracted_video_dict = self._extract_video_frames(
            self.subdict["video_path"], frame_rate_per_second=frame_rate_per_second
        )
        if extracted_video_dict["n_frames"] <= minimal_edge_of_frames:
            answer = self.brute_force_summary(extracted_video_dict)

        else:
            # TODO: implement processing for long videos
            summary_instruction = "Describe this image in a single caption, including all important details."
            answer = self.brute_force_summary(
                extracted_video_dict, summary_instruction=summary_instruction
            )

        return answer
