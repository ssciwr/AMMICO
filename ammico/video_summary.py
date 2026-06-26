from ammico.inference import InferenceModel, AudioTranscriptionModel
from ammico.utils import (
    AnalysisMethod,
    AnalysisType,
    _categorize_outputs,
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


class VideoSummaryDetector(AnalysisMethod):
    def __init__(
        self,
        summary_model: InferenceModel = None,
        audio_model: Optional[AudioTranscriptionModel] = None,
        subdict: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Class for analysing videos using an externally hosted vision-language model.
        It provides methods for generating captions and answering questions about videos.

        Args:
            summary_model (InferenceModel): An InferenceModel instance used for analysis.
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

    def _audio_to_text(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Convert audio file to text using the externally hosted transcription model.
        Args:
            audio_path (str): Path to the audio file.
        Returns:
            List[Dict[str, Any]]: List of transcribed audio segments with start_time, end_time, text, and duration.
        """
        return self.audio_model.transcribe(audio_path)

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
        Extract audio part from the video file and transcribe it using the external audio model.
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
        Returns:
            List of segments with 'start_time', 'end_time', and 'frame_timestamps'
        """
        base_frames_per_clip = 4.0
        max_seconds_between_frames = 30.0
        result = self._detect_scene_cuts(filename)
        segments = result["segments"]
        video_meta = result["video_meta"]
        for seg in segments:
            if seg["duration"] < 2.0:
                frame_rate_per_clip = 2.0
            elif seg["duration"] > max_seconds_between_frames:
                frame_rate_per_clip = max(
                    1,
                    int(math.ceil(seg["duration"] / max_seconds_between_frames)) + 1,
                )
            else:
                frame_rate_per_clip = base_frames_per_clip

            start_time = seg["start_time"]
            end_time = seg["end_time"]
            sample_times = torch.linspace(
                start_time,
                end_time,
                steps=int(frame_rate_per_clip),
                dtype=torch.float32,
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

        eps = 1e-6
        video_list = list(video_segs)

        for seg in segments:
            seg_start = seg["start_time"]
            seg_end = seg["end_time"]

            merged_timestamps: List[float] = []
            fallback_candidates: List[Tuple[float, float]] = []
            for vscene in video_list:
                if "frame_timestamps" not in vscene:
                    raise ValueError("Video scene missing 'frame_timestamps' key.")

                v_start = float(vscene["start_time"])
                v_end = float(vscene["end_time"])

                overlap_start = max(seg_start, v_start)
                overlap_end = min(seg_end, v_end)

                has_overlap = overlap_end + eps >= overlap_start

                if has_overlap:
                    fallback_candidates.append((overlap_start, overlap_end))

                contrib = [
                    float(t)
                    for t in vscene["frame_timestamps"]
                    if (t + eps) >= seg_start and (t - eps) <= seg_end
                ]

                if contrib:
                    merged_timestamps.extend(contrib)

            if not merged_timestamps and fallback_candidates:
                best_start, best_end = max(
                    fallback_candidates,
                    key=lambda bounds: bounds[1] - bounds[0],
                )
                fallback_ts = (best_start + best_end) / 2.0
                merged_timestamps.append(fallback_ts)

            seg["video_frame_timestamps"] = sorted(
                {
                    round(float(t), 6)
                    for t in merged_timestamps
                    if math.isfinite(float(t))
                }
            )

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
            original_w: Original width of the video
            original_h: Original height of the video
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
        merged_segments: List[Dict[str, Any]],
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
        img_width = video_meta.get("width")
        img_height = video_meta.get("height")
        if img_width is None or img_height is None:
            raise ValueError(
                "Frame dimensions not found in the last segment for extraction."
            )

        for seg in merged_segments:  # TODO might be generator faster, so changes to ffmpeg extraction may be needed
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
            # one request per frame (single image + caption instruction), fanned out
            messages_batch = [
                self.summary_model.build_messages(img, caption_instruction)
                for _, img in pairs
            ]
            # questions yield both a summary and VQA answers, so allow more tokens
            max_new_tokens = 256 if include_questions else 128
            captions = self.summary_model.chat_batch(
                messages_batch, max_new_tokens=max_new_tokens
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
            entry["audio_descriptions"] = audio_generated_captions

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
            messages = self.summary_model.build_messages(None, caption_instruction)
            final_outputs = self.summary_model.chat(messages, max_new_tokens=512)
            clip_text = final_outputs[0].strip() if final_outputs else ""

            if frame_timestamps:
                clip_timestamp = float(frame_timestamps[0])
            else:
                clip_timestamp = (
                    float(seg["start_time"]) + float(seg["end_time"])
                ) / 2.0

            if clip_text:
                collected.append((clip_timestamp, clip_text))

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
        bullets = []
        for seg in summary_dict:
            seg_bullets = seg.get("summary_bullets", [])
            bullets.extend(seg_bullets)
        if not bullets:
            raise ValueError("No captions available for summary generation.")

        summary_user_prompt = self.prompt_builder.build_video_prompt(
            include_vqa=False,
            clip_summaries=bullets,
        )
        messages = self.summary_model.build_messages(None, summary_user_prompt)
        final_summary_list = self.summary_model.chat(messages, max_new_tokens=512)
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
        summary_bullets = []
        for seg in answers_dict:
            summary_bullets.extend(seg.get("summary_bullets", []))
            seg_bullets = seg.get("vqa_bullets", [])
            vqa_bullets.extend(seg_bullets)

        if not vqa_bullets:
            raise ValueError(
                "No VQA bullets generated for single frames available for answering questions."
            )

        include_questions = bool(list_of_questions)
        if include_questions:
            prompt = self.prompt_builder.build_video_prompt(
                include_vqa=include_questions,
                questions=list_of_questions,
                vqa_bullets=vqa_bullets,
                clip_summaries=summary_bullets,
            )
        else:
            raise ValueError(
                "list_of_questions must be provided for making final answers."
            )

        messages = self.summary_model.build_messages(None, prompt)
        final_vqa_list = self.summary_model.chat(messages, max_new_tokens=512)
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

        analysis_type, is_summary, is_questions = AnalysisType._validate_analysis_type(
            analysis_type, list_of_questions
        )

        for video_key, entry in self.subdict.items():
            answers_dict = self.make_captions_for_subclips(
                entry,
                list_of_questions=list_of_questions,
            )
            if is_summary:
                answer = self.final_summary(answers_dict)
                entry["summary"] = answer["summary"]
            if is_questions:
                answer = self.final_answers(answers_dict, list_of_questions)
                entry["vqa_answers"] = answer["vqa_answers"]

            self.subdict[video_key] = entry

        return self.subdict
