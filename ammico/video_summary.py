from ammico.model import MultimodalSummaryModel, AudioToTextModel
from ammico.utils import (
    AnalysisMethod,
    AnalysisType,
    _categorize_outputs,
    _strip_prompt_prefix_literal,
    _validate_subdict,
)
from ammico.prompt_builder import PromptBuilder

import os
import re
import cv2
import math
import numpy as np
import subprocess
import tempfile
import torch
import warnings
import whisperx

from scipy import signal
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from typing import (
    List,
    Dict,
    Any,
    Tuple,
    Union,
    Optional,
)
from transformers import GenerationConfig


class VideoSummaryDetector(AnalysisMethod):
    def __init__(
        self,
        summary_model: MultimodalSummaryModel = None,
        audio_model: Optional[AudioToTextModel] = None,
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
        _validate_subdict(subdict)
        self.summary_model = summary_model or None
        self.audio_model = audio_model
        self.prompt_builder = PromptBuilder()

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
        Args:
            generated_ids (torch.Tensor): Generated token IDs from the model.
            inputs (Dict[str, torch.Tensor]): Original input tensors used for generation.
            tokenizer: Tokenizer used for decoding the generated outputs.
            prompt_texts (List[str]): List of prompt texts corresponding to each input in the batch.
        Returns:
            List[str]: Decoded generated texts after trimming and cleaning.
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
        len_objects: Optional[int] = None,
    ) -> List[str]:
        """
        Run model.generate on already-processed processor_inputs (tensors moved to device),
        then decode and trim prompt tokens & remove literal prompt prefixes using prompt_texts.
        Args:
            processor_inputs (Dict[str, torch.Tensor]): Inputs prepared by the processor.
            prompt_texts (List[str]): List of prompt texts corresponding to each input in the batch.
            tokenizer: Tokenizer used for decoding the generated outputs.
            len_objects (Optional[int], optional): Number of objects/frames to adjust max_new_tokens. Defaults to None.
        Returns:
            List[str]: Decoded generated texts after trimming and cleaning.
        """
        # In case of many frames, allow more max_new_tokens # TODO recheck the logic
        if len_objects is not None:
            max_new_tokens = len_objects * 128
        else:
            max_new_tokens = 128
        gen_conf = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_return_sequences=1,
        )

        for k, v in processor_inputs.items():
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

    def _audio_to_text(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Convert audio file to text using an whisper model.
        Args:
            audio_path (str): Path to the audio file.
        Returns:
            List[Dict[str, Any]]: List of transcribed audio segments with start_time, end_time, text, and duration.
        """

        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file {audio_path} does not exist.")

        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            audio = whisperx.load_audio(audio_path)
            transcribe_result = self.audio_model.model.transcribe(audio)
            model_a, metadata = whisperx.load_align_model(
                language_code=transcribe_result["language"],
                device=self.audio_model.device,
            )
            aligned_result = whisperx.align(
                transcribe_result["segments"],
                model_a,
                metadata,
                audio,
                self.audio_model.device,
            )
            audio_descriptions = []
            for segment in aligned_result["segments"]:
                audio_descriptions.append(
                    {
                        "start_time": segment["start"],
                        "end_time": segment["end"],
                        "text": segment["text"].strip(),
                        "duration": segment["end"] - segment["start"],
                    }
                )
            return audio_descriptions
        except Exception as e:
            raise RuntimeError(f"Failed to transcribe audio: {e}")

    def _check_audio_stream(self, filename: str) -> bool:
        """
        Check if the video file has an audio stream.
        Args:
            filename (str): Path to the video file.
        Returns:
            bool: True if audio stream exists, False otherwise.
        """
        try:
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                filename,
            ]
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            output = result.stdout.strip()
            return bool(output)
        except Exception as e:
            warnings.warn(
                f"Failed to check audio stream in video {filename}: {e}",
                RuntimeWarning,
            )
            return False

    def _extract_transcribe_audio_part(
        self,
        filename: str,
    ) -> List[Dict[str, Any]]:
        """
        Extract audio part from the video file and generate captions using an audio whisperx model.
        Args:
            filename (str): Path to the video file.
        Returns:
            List[Dict[str, Any]]: List of transcribed audio segments with start_time, end_time, text, and duration.
        """

        if not self._check_audio_stream(filename):
            self.audio_model.close()
            self.audio_model = None
            return []

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_output_path = os.path.join(tmpdir, "audio_extracted.wav")
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-i",
                        filename,
                        "-vn",
                        "-acodec",
                        "pcm_s16le",
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        "-y",
                        audio_output_path,
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to extract audio from video: {e}")

            audio_descriptions = self._audio_to_text(audio_output_path)

        # and close the audio model to free up resources
        self.audio_model.close()
        self.audio_model = None

        return audio_descriptions

    def _detect_scene_cuts(
        self,
        filename: str,
    ) -> Dict[str, Any]:
        """
        Detect scene cuts in the video using frame differencing method.
        Args:
            filename: Path to the video file
        Returns:
            List of segments with 'start_time' and 'end_time'
        """

        cap = cv2.VideoCapture(filename)
        fps = cap.get(cv2.CAP_PROP_FPS)

        frames = []
        img_height, img_width = None, None
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img_height, img_width = frame.shape[:2]

            try:
                if img_width / img_height > 1.2:
                    frame_small = cv2.resize(frame, (320, 240))
                elif img_width / img_height < 0.8:
                    frame_small = cv2.resize(frame, (240, 320))
                else:
                    frame_small = cv2.resize(frame, (320, 320))
            except Exception as e:
                raise RuntimeError(
                    f"Failed to resize frame for scene cut detection: {e}"
                )

            gray = cv2.cvtColor(
                frame_small, cv2.COLOR_BGR2GRAY
            )  # TODO check if it is ok, maybe we can use color info as well
            frames.append(gray)

        cap.release()
        if img_height is None or img_width is None:
            raise ValueError(
                "Failed to read frames from video for scene cut detection."
            )

        # Compute frame differences to keep memory usage low
        frame_diffs = []
        for i in range(1, len(frames)):
            diff = cv2.absdiff(frames[i], frames[i - 1])
            mean_diff = np.mean(diff)
            frame_diffs.append(mean_diff)

        # Find peaks in differences (scene cuts) via adaptive threshold based on median
        threshold = 25.0
        median_diff = np.median(frame_diffs)
        cut_threshold = median_diff + threshold

        cut_frames = signal.find_peaks(
            frame_diffs,
            height=cut_threshold,
            distance=int(fps * 0.5),  # At least 0.5s between cuts
        )[0]

        video_segments = []
        cut_frames_with_starts = [0] + list(cut_frames) + [len(frames)]

        for i in range(len(cut_frames_with_starts) - 1):
            start_frame = cut_frames_with_starts[i]
            end_frame = cut_frames_with_starts[i + 1]

            video_segments.append(
                {
                    "type": "video_scene",
                    "start_time": start_frame / fps,
                    "end_time": end_frame / fps,
                    "duration": (end_frame - start_frame) / fps,
                }
            )

        # Since there may be issues with last frame detection, slightly adjust the end_time of the last segment
        last_segment = video_segments[-1]
        last_segment["end_time"] -= 0.5
        # Ensure the end_time does not go below the start_time in case of very short last segment/video
        if last_segment["end_time"] < last_segment["start_time"]:
            last_segment["end_time"] = last_segment["start_time"]

        return {
            "segments": video_segments,
            "video_meta": {
                "width": img_width,
                "height": img_height,
            },
        }

    def _extract_frame_timestamps_from_clip(
        self,
        filename: str,
    ) -> Dict[str, Any]:
        """
        Extract frame timestamps for each detected video segment.
        Args:
            filename: Path to the video file
            frame_rate_per_clip: Number of frames to sample per second within each segment
        Returns:
            List of segments with 'start_time', 'end_time', and 'frame_timestamps'
        """
        base_frames_per_clip = 4.0
        result = self._detect_scene_cuts(filename)
        segments = result["segments"]
        video_meta = result["video_meta"]
        for seg in segments:
            if seg["duration"] < 2.0:
                frame_rate_per_clip = 2.0
            elif seg["duration"] > 20.0:
                frame_rate_per_clip = 6.0
            else:
                frame_rate_per_clip = base_frames_per_clip

            start_time = seg["start_time"]
            end_time = seg["end_time"]
            n_samples = max(1, int(frame_rate_per_clip))
            sample_times = torch.linspace(
                start_time, end_time, steps=n_samples, dtype=torch.float32
            )
            seg["frame_timestamps"] = sample_times.tolist()

        return {
            "segments": segments,
            "video_meta": video_meta,
        }

    def _reassign_video_timestamps_to_segments(
        self,
        segments: List[Dict[str, Any]],
        video_segs: List[Dict[str, Any]],
    ) -> None:
        """
        Reassign video frame timestamps to each new segment based on overlapping video scenes.
        Args:
            segments: List of segments to assign timestamps to.
            video_segs: List of video scenes with original frame timestamps.
        Returns:
            None
        """

        boundary_margin = 0.5
        eps = 1e-6

        video_list = list(video_segs)
        for seg in segments:
            seg_start = seg["start_time"]
            seg_end = seg["end_time"]

            merged_timestamps: List[float] = []
            for vscene in video_list:
                if "frame_timestamps" not in vscene:
                    raise ValueError("Video scene missing 'frame_timestamps' key.")

                contrib = [
                    float(t)
                    for t in vscene["frame_timestamps"]
                    if (t + eps) >= (seg_start - boundary_margin)
                    and (t - eps) <= (seg_end + boundary_margin)
                    and (t + eps) >= seg_start
                    and (t - eps) <= seg_end
                ]
                if contrib:
                    merged_timestamps.extend(contrib)

            # dedupe & sort
            seg["video_frame_timestamps"] = sorted(set(merged_timestamps))

    def _combine_visual_frames_by_time(
        self,
        video_segs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Split too-long video segments (>25s).
        Args:
            video_segs: List of video segments with 'start_time' and 'end_time'
        Returns:
            List of combined segments
        """

        if not video_segs:
            raise ValueError("No video segments to combine.")
        out = []
        for vs in video_segs:
            st, ed, dur = (
                float(vs["start_time"]),
                float(vs["end_time"]),
                float(vs["duration"]),
            )
            if dur > 25.0:
                parts = int(math.ceil(dur / 25.0))
                part_dur = dur / parts
                for p in range(parts):
                    ps = st + p * part_dur
                    pe = st + (p + 1) * part_dur if p < parts - 1 else ed
                    out.append(
                        {
                            "start_time": ps,
                            "end_time": pe,
                            "duration": pe - ps,
                            "audio_phrases": [],
                            "video_scenes": [vs],
                        }
                    )
            else:
                out.append(
                    {
                        "start_time": st,
                        "end_time": ed,
                        "duration": dur,
                        "audio_phrases": [],
                        "video_scenes": [vs],
                    }
                )

        self._reassign_video_timestamps_to_segments(out, video_segs)
        return out

    def merge_audio_visual_boundaries(
        self,
        audio_segs: List[Dict[str, Any]],
        video_segs: List[Dict[str, Any]],
        segment_threshold_duration: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        Merge audio phrase boundaries and video scene cuts into coherent temporal segments for the model
        Args:
            audio_segs: List of audio segments with 'start_time' and 'end_time'
            video_segs: List of video segments with 'start_time' and 'end_time'
            segment_threshold_duration: Duration to create a new segment boundary
        Returns:
            List of merged segments
        """
        if not audio_segs:
            new_vid = self._combine_visual_frames_by_time(video_segs)
            return new_vid

        events = [
            ("audio", seg["start_time"], seg["end_time"], seg) for seg in audio_segs
        ] + [("video", seg["start_time"], seg["end_time"], seg) for seg in video_segs]

        if not events:
            raise ValueError("No audio and video segments to merge.")

        events.sort(key=lambda x: x[1])
        global_last_end = max(e[2] for e in events)
        # Create merged segments respecting both boundaries
        merged = []
        current_segment_start = 0
        current_audio_phrases = []
        current_video_scenes = []

        for event_type, start, _, data in events:
            current_duration = start - current_segment_start
            if current_duration > segment_threshold_duration:
                segment_end = start

                if segment_end < current_segment_start:
                    segment_end = current_segment_start

                merged.append(
                    {
                        "start_time": current_segment_start,
                        "end_time": segment_end,
                        "audio_phrases": current_audio_phrases,
                        "video_scenes": current_video_scenes,
                        "duration": segment_end - current_segment_start,
                    }
                )
                # start a new segment at the current event's start
                current_segment_start = segment_end
                current_audio_phrases = []
                current_video_scenes = []

            if event_type == "audio":
                current_audio_phrases.append(data)
            else:
                current_video_scenes.append(data)

        if current_audio_phrases or current_video_scenes:
            final_end = max(global_last_end, events[-1][2], current_segment_start)
            if final_end < current_segment_start:
                final_end = current_segment_start

            merged.append(
                {
                    "start_time": current_segment_start,
                    "end_time": final_end,
                    "audio_phrases": current_audio_phrases,
                    "video_scenes": current_video_scenes,
                    "duration": final_end - current_segment_start,
                }
            )

        self._reassign_video_timestamps_to_segments(merged, video_segs)
        return merged

    def _run_ffmpeg(
        self, cmd_args: List[str], timeout: Optional[float]
    ) -> subprocess.CompletedProcess:
        """
        Execute ffmpeg command and return the completed process.
        Args:
            cmd_args: List of ffmpeg command arguments.
            timeout: Timeout for the subprocess.
        Returns:
            CompletedProcess: Result of the subprocess execution.
        """

        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"] + cmd_args
        return subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout
        )

    def _build_extract_command(
        self, filename: str, timestamp: float, accurate: bool, codec: str = "png"
    ) -> List[str]:
        """
        Build ffmpeg command for frame extraction.

        Args:
            filename: Path to video file
            timestamp: Time in seconds
            accurate: If True, seek after input (slow but accurate)
            codec: Output codec ('png' or 'mjpeg')
        Returns:
            List of ffmpeg command arguments
        """
        ss_arg = f"{timestamp:.6f}"
        cmd = []
        # Position -ss based on accuracy requirement
        if accurate:
            cmd = ["-i", filename, "-ss", ss_arg]  # accurate mode
        else:
            cmd = ["-ss", ss_arg, "-i", filename]  # fast mode

        # Common extraction parameters
        cmd += ["-frames:v", "1", "-f", "image2pipe"]

        # Codec-specific settings
        if codec == "png":
            cmd += ["-vcodec", "png", "-pix_fmt", "rgb24"]
        elif codec == "mjpeg":
            cmd += ["-vcodec", "mjpeg", "-pix_fmt", "yuvj420p"]

        cmd.append("pipe:1")
        return cmd

    def _run_ffmpeg_extraction(
        self,
        filename: str,
        timestamp: float,
        out_w: int,
        out_h: int,
        timeout: Optional[float] = 30.0,
    ) -> Image.Image:
        """
        Extract a single frame at the specified timestamp.

        Args:
            filename: Path to video file
            timestamp: Time in seconds
            out_w: Optional output width
            out_h: Optional output height
            timeout: Subprocess timeout in seconds

        Returns:
            PIL Image in RGB format

        Raises:
            RuntimeError: If frame extraction fails
        """

        strategies = [
            ("mjpeg", False),
            ("png", True),
        ]

        last_error = None

        for codec, use_accurate in strategies:
            try:
                cmd = self._build_extract_command(
                    filename, timestamp, use_accurate, codec
                )
                proc = self._run_ffmpeg(cmd, timeout)

                if proc.returncode == 0 and proc.stdout:
                    img = Image.open(BytesIO(proc.stdout)).convert("RGB")
                    img = img.resize((out_w, out_h), resample=Image.BILINEAR)
                    return img
                else:
                    last_error = proc.stderr.decode("utf-8", errors="replace")

            except Exception as e:
                last_error = str(e)
                warnings.warn(
                    f"Frame extraction failed at {timestamp:.3f}s with codec {codec} "
                    f"({'accurate' if use_accurate else 'fast'}): {last_error}",
                    RuntimeWarning,
                )

        raise RuntimeError(
            f"Failed to extract frame at {timestamp:.3f}s from {filename}. "
            f"Last error: {last_error[:500]}"
        )

    def _calculate_output_dimensions(
        self, original_w: int, original_h: int
    ) -> Tuple[int, int]:
        """
        Calculate output dimensions in a fully adaptive way, preserving aspect ratio, but decreasing size.
        It works both for landscape and portrait videos.
        Args:
            original_w: Original width
            original_h: Original height
        Returns:
            Tuple of (out_w, out_h)
        """
        aspect_ratio = original_w / original_h
        max_dimension = 720

        if aspect_ratio > 1.2:
            out_w = max_dimension
            out_h = int(max_dimension / aspect_ratio)
        elif aspect_ratio < 0.8:
            out_h = max_dimension
            out_w = int(max_dimension * aspect_ratio)
        else:
            out_w = max_dimension
            out_h = max_dimension
        return out_w, out_h

    def _extract_frames_ffmpeg(
        self,
        filename: str,
        timestamps: List[float],
        original_w: int,
        original_h: int,
        workers: int = 4,
    ) -> List[Tuple[float, Image.Image]]:
        """
        Extract multiple frames using a thread pool (parallel ffmpeg processes).
        Args:
            filename: Path to video file
            timestamps: List of times in seconds
            out_w: Frame width
            out_h: Frame height
            workers: Number of parallel threads
        Returns:
          List of (timestamp, PIL.Image) preserving order of timestamps.
        """
        results = {}
        out_w, out_h = self._calculate_output_dimensions(original_w, original_h)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(self._run_ffmpeg_extraction, filename, t, out_w, out_h): i
                for i, t in enumerate(timestamps)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    img = fut.result()
                    results[idx] = img
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to extract frame for {timestamps[idx]}s: {e}"
                    ) from e

        return [(timestamps[i], results[i]) for i in range(len(timestamps))]

    def _make_captions_from_extracted_frames(
        self,
        filename: str,
        merged_segments: List[Tuple[float, Image.Image]],
        video_meta: Dict[str, Any],
        list_of_questions: Optional[List[str]] = None,
    ) -> None:
        """
        Generate captions for all extracted frames and then produce a concise summary of the video.
        Args:
            filename (str): Path to the video file.
            merged_segments (List[Dict[str, Any]]): List of merged segments with frame timestamps.
            list_of_questions (Optional[List[str]]): List of questions for VQA.
        Returns:
            None. Modifies merged_segments in place to add 'summary_bullets' and 'vqa_bullets'.
        """
        proc = self.summary_model.processor

        img_width = video_meta.get("width")
        img_height = video_meta.get("height")
        if img_width is None or img_height is None:
            raise ValueError(
                "Frame dimensions not found in the last segment for extraction."
            )

        for seg in merged_segments:  # TODO might be generator faster, so changes to ffmmpeg extraction may be needed
            collected: List[Tuple[float, str]] = []
            frame_timestamps = seg.get("video_frame_timestamps", [])
            if not frame_timestamps:
                raise ValueError(
                    f"No frame timestamps found for segment {seg['start_time']:.2f}s to {seg['end_time']:.2f}s"
                )
            include_questions = bool(list_of_questions)
            caption_instruction = self.prompt_builder.build_frame_prompt(
                include_vqa=include_questions,
                questions=list_of_questions,
            )
            pairs = self._extract_frames_ffmpeg(
                filename,
                frame_timestamps,
                original_w=img_width,
                original_h=img_height,
                workers=min(8, (os.cpu_count() or 1) // 2),
            )
            prompt_texts = []

            for ts, img in pairs:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
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
                images=[img for _, img in pairs],
                return_tensors="pt",
                padding=True,
            )
            len_objects = len(pairs)
            if include_questions:
                len_objects *= 2  # because we expect two outputs per input when questions are included
            captions = self._generate_from_processor_inputs(
                processor_inputs,
                prompt_texts,
                self.summary_model.tokenizer,
                len_objects=len_objects,
            )
            for t, c in zip(frame_timestamps, captions):
                collected.append((float(t), c))

            collected.sort(key=lambda x: x[0])
            bullets_summary, bullets_vqa = _categorize_outputs(
                collected, include_questions
            )

            seg["summary_bullets"] = bullets_summary
            seg["vqa_bullets"] = bullets_vqa

    def make_captions_for_subclips(
        self,
        entry: Dict[str, Any],
        list_of_questions: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate captions for video subclips using both audio and visual information, for a further full video summary/VQA.
        Args:
            entry (Dict[str, Any]): Dictionary containing the video file information.
            list_of_questions (Optional[List[str]]): List of questions for VQA.
        Returns:
            List[Dict[str, Any]]: List of dictionaries containing timestamps and generated captions.
        """

        filename = entry.get("filename")
        if not filename:
            raise ValueError("entry must contain key 'filename'")

        if not os.path.exists(filename):
            raise ValueError(f"Video file {filename} does not exist.")

        audio_generated_captions = []
        if self.audio_model is not None:
            audio_generated_captions = self._extract_transcribe_audio_part(filename)

        video_result_segments = self._extract_frame_timestamps_from_clip(filename)
        video_segments_w_timestamps = video_result_segments["segments"]
        video_meta = video_result_segments["video_meta"]
        merged_segments = self.merge_audio_visual_boundaries(
            audio_generated_captions,
            video_segments_w_timestamps,
        )

        self._make_captions_from_extracted_frames(
            filename,
            merged_segments,
            video_meta,
            list_of_questions=list_of_questions,
        )
        results = []
        proc = self.summary_model.processor
        for seg in merged_segments:
            frame_timestamps = seg.get("video_frame_timestamps", [])

            collected: List[Tuple[float, str]] = []
            include_audio = False
            audio_lines = seg["audio_phrases"]
            if audio_lines:
                include_audio = True

            include_questions = bool(list_of_questions)
            caption_instruction = self.prompt_builder.build_clip_prompt(
                frame_bullets=seg.get("summary_bullets", []),
                include_audio=include_audio,
                audio_transcription=seg.get("audio_phrases", []),
                include_vqa=include_questions,
                questions=list_of_questions,
                vqa_bullets=seg.get("vqa_bullets", []),
            )
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": caption_instruction}],
                }
            ]
            prompt_text = proc.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            processor_inputs = proc(
                text=[prompt_text],
                return_tensors="pt",
                padding=True,
            )
            final_outputs = self._generate_from_processor_inputs(
                processor_inputs,
                [prompt_text],
                self.summary_model.tokenizer,
            )
            for t, c in zip(frame_timestamps, final_outputs):
                collected.append((float(t), c))

            collected.sort(key=lambda x: x[0])
            bullets_summary, bullets_vqa = _categorize_outputs(
                collected, include_questions
            )

            results.append(
                {
                    "start_time": seg["start_time"],
                    "end_time": seg["end_time"],
                    "summary_bullets": bullets_summary,
                    "vqa_bullets": bullets_vqa,
                }
            )

        return results

    def final_summary(self, summary_dict: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Produce a concise summary of the video, based on generated captions for all extracted frames.
        Args:
            summary_dict (Dict[str, Any]): Dictionary containing captions for the frames.
        Returns:
            Dict[str, Any]: A dictionary containing the list of captions with timestamps and the final summary.
        """
        proc = self.summary_model.processor

        bullets = []
        for seg in summary_dict:
            seg_bullets = seg.get("summary_bullets", [])
            bullets.extend(seg_bullets)
        if not bullets:
            raise ValueError("No captions available for summary generation.")

        summary_user_prompt = self.prompt_builder.build_video_prompt(
            summary_only=True,
            include_vqa=False,
            clip_summaries=bullets,
        )
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": summary_user_prompt}],
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
        answers_dict: List[Dict[str, Any]],
        list_of_questions: List[str],
    ) -> Dict[str, Any]:
        """
        Answer the list of questions for the video based on the VQA bullets from the frames.
        Args:
            answers_dict (Dict[str, Any]): Dictionary containing the VQA bullets.
        Returns:
            Dict[str, Any]: A dictionary containing the list of answers to the questions.
        """
        vqa_bullets = []
        for seg in answers_dict:
            seg_bullets = seg.get("vqa_bullets", [])
            vqa_bullets.extend(seg_bullets)

        if not vqa_bullets:
            raise ValueError(
                "No VQA bullets generated for single frames available for answering questions."
            )

        include_questions = bool(list_of_questions)
        if include_questions:
            prompt = self.prompt_builder.build_video_prompt(
                summary_only=False,
                include_vqa=include_questions,
                questions=list_of_questions,
                vqa_bullets=vqa_bullets,
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
            r"\d+\.\s+(.+?)(?=\n\d+\.|$)", final_vqa_output, flags=re.DOTALL
        )
        for answer in answer_matches:
            vqa_answers.append(answer.strip())
        return {
            "vqa_answers": vqa_answers,
        }

    def analyse_videos_from_dict(
        self,
        analysis_type: Union[AnalysisType, str] = AnalysisType.SUMMARY,
        list_of_questions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Analyse the video specified in self.subdict using frame extraction and captioning.
        Args:
            analysis_type (Union[AnalysisType, str], optional): Type of analysis to perform. Defaults to AnalysisType.SUMMARY.
            list_of_questions (List[str], optional): List of questions to answer about the video. Required if analysis_type includes questions.
        Returns:
            Dict[str, Any]: A dictionary containing the analysis results, including summary and answers for provided questions(if any).
        """
        if list_of_questions is not None and not isinstance(list_of_questions, list):
            raise TypeError("Expected list_of_questions to be a list of strings.")
        if list_of_questions and any(not isinstance(q, str) for q in list_of_questions):
            raise ValueError("All items in list_of_questions must be strings.")

        all_answers = {}
        analysis_type, is_summary, is_questions = AnalysisType._validate_analysis_type(
            analysis_type, list_of_questions
        )

        for video_key, entry in self.subdict.items():
            summary_and_vqa = {}
            answers_dict = self.make_captions_for_subclips(
                entry,
                list_of_questions=list_of_questions,
            )
            if is_summary:
                answer = self.final_summary(answers_dict)
                summary_and_vqa["summary"] = answer["summary"]
            if is_questions:
                answer = self.final_answers(answers_dict, list_of_questions)
                summary_and_vqa["vqa_answers"] = answer["vqa_answers"]

            all_answers[video_key] = summary_and_vqa

        return all_answers
