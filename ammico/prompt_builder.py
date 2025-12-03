from typing import List, Dict, Optional, Any
from enum import Enum


class ProcessingLevel(Enum):
    """Define the three processing levels in a pipeline.
    FRAME: individual frame analysis
    CLIP: video segment (multiple frames)
    VIDEO: full video (multiple clips)"""

    FRAME = "frame"
    CLIP = "clip"
    VIDEO = "video"


class PromptBuilder:
    """
    Modular prompt builder for multi-level video analysis.
    Handles frame-level, clip-level, and video-level prompts.
    """

    ROLE_MODULE = """You are a precise video analysis AI. Your purpose is to:
    - Extract only information explicitly present in provided sources
    - Never hallucinate or infer beyond what is shown
    - Generate clear, concise, well-structured outputs
    - Maintain logical coherence across visual and audio sources"""

    CONSTRAINTS_MODULE = """## Quality Requirements

    - **Accuracy:** Use only explicitly provided information
    - **Conciseness:** Eliminate redundancy; be direct
    - **Clarity:** Use accessible language
    - **Consistency:** Align audio and visual information when both exist
    - **Format Compliance:** Follow output format exactly"""

    @staticmethod
    def visual_frames_module() -> str:
        """For frame-level processing with actual images."""
        str_to_return = """## Visual Information
        
        Visual information is represented by keyframe extracted from the video segment at specified timestamp."""
        return str_to_return

    @staticmethod
    def visual_captions_module(frame_bullets: List[str]) -> str:
        """For clip-level processing with frame summaries."""
        bullets_text = "\n".join(frame_bullets)
        str_to_return = f"""## Visual Information

        The following are summary bullets extracted from video frames with timestamps.
        These bullets represent the visual content detected in each frame of the video segment:

        {bullets_text}"""

        return str_to_return

    @staticmethod
    def visual_captions_final_module(clip_summaries: List[str]) -> str:
        """For video-level processing with clip summaries."""
        str_to_return = f"""## Visual Information

        The following are brief summaries obtained for each segment of the video.
        These summaries are associated with the timestamp of each segment's beginning:

        {clip_summaries}"""

        return str_to_return

    @staticmethod
    def audio_module(audio_transcription: List[Dict[str, Any]]) -> str:
        """Audio transcription with timestamps."""
        audio_text = "\n".join(
            [
                f"[{a['start_time']:.2f}s - {a['end_time']:.2f}s]: {a['text'].strip()}"
                for a in audio_transcription
            ]
        )
        str_to_return = f"""## Audio Information

        The following is the audio transcription for the same video segment,
        with precise timestamps for each spoken element:

        {audio_text}"""
        return str_to_return

    @staticmethod
    def summary_task(has_audio: bool = False) -> str:
        """Generate summary task (with or without audio)."""
        sources = "visual and audio information" if has_audio else "visual information"
        str_to_return = f"""## Task: Generate Concise Summary

        Based on the {sources} provided, generate a brief summary that:
        - Captures and summarizes the main events and themes
        - Uses clear, accessible language
        - Is between 1-3 sentences
        - Contains no unsupported claims

        Return ONLY this format:

        Summary: <your summary here>"""
        return str_to_return

    @staticmethod
    def summary_vqa_task(
        level: ProcessingLevel,
        has_audio: bool = False,
    ) -> str:
        """Generate summary+VQA task (adapts based on level and audio). For Frame and Clip levels."""

        if level == ProcessingLevel.FRAME:
            vqa_task = """Answer the provided questions based ONLY on information from the visual information. Answers must be brief and direct.

            **Critical Rule:** If you cannot answer a question from the provided sources,
            respond with: "Cannot be determined from provided information."
            """
        elif level == ProcessingLevel.CLIP:
            if has_audio:
                priority_list = """
                    1. **Frame-Level Answers** - Pre-computed per-frame answers (auxiliary reference only)
                    2. **Audio Information** - Spoken content from audio transcription
                    3. **Visual Information** - Direct visual content from video frames"""
            else:
                priority_list = """1. **Frame-Level Answers** - Pre-computed per-frame answers (auxiliary reference only)
                    2. **Visual Information** - Direct visual content from video frames"""
            vqa_task = f"""For each question, use the BEST available source in this priority:
                {priority_list}

                **Critical Logic:**
                - If frame-level answer is a REAL answer (not "Cannot be determined from provided information.") → use it
                - If frame-level answer is "Cannot be determined" → SKIP IT and check audio/visual instead
                - If answer is found in visual OR audio information → use that
                - ONLY respond "Cannot be determined" if truly no information exists anywhere

                """

        str_to_return = f"""## You have two tasks:

        ### task 1: Concise Summary
        Generate a brief summary that captures and summarizes main events and themes from the visual information (1-3 sentences).

        ### task 2: Question Answering
        {vqa_task}

        Return ONLY this format:

        Summary: <your summary here>

        VQA Answers:
        1. <answer to question 1>
        2. <answer to question 2>
        [etc.]"""

        return str_to_return

    @staticmethod
    def vqa_only_task() -> str:
        """VQA-only task for video-level processing."""
        str_to_return = """## Task: Answer Questions

        For each question, use the BEST available source in this priority:
            1. **Segment-Level Answers** - Pre-computed per-frame answers (auxiliary reference only)
            2. **Visual Information** - Direct visual content from video frames

        **Critical Logic:**
                - If segment-level answer is a REAL answer (not "Cannot be determined from provided information.") → use it
                - If frame-level answer is "Cannot be determined" → SKIP IT and check visual information instead
                - If answer is found in visual information → use that
                - ONLY respond "Cannot be determined" if truly no information exists anywhere

        Return ONLY this format:

        VQA Answers:
        1. <answer to question 1>
        2. <answer to question 2>
        [etc.]"""

        return str_to_return

    @staticmethod
    def questions_module(questions: List[str]) -> str:
        """Format questions list."""
        questions_text = "\n".join(
            [f"{i + 1}. {q.strip()}" for i, q in enumerate(questions)]
        )
        str_to_return = f"""## Questions to Answer

        {questions_text}
        """
        return str_to_return

    @staticmethod
    def vqa_context_module(vqa_bullets: List[str], is_final: bool = False) -> str:
        """VQA context (frame-level or clip-level answers)."""
        if is_final:
            header = """## SEGMENT-Level Answer Context (Reference Only)

            The following are answers to above questions obtained for each segment of the video.
            These answers are associated with the timestamp of each segment's beginning:"""
        else:
            header = """## FRAME-Level Answer Context (Reference Only)

            For each question, the following are frame-level answers provided as reference.
            If these answers are "Cannot be determined", do not accept that as final—instead, 
            use visual and audio information to answer the question:"""

        bullets_text = "\n".join(vqa_bullets)
        return f"{header}\n\n{bullets_text}"

    @classmethod
    def build_frame_prompt(
        cls, include_vqa: bool = False, questions: Optional[List[str]] = None
    ) -> str:
        """Build prompt for frame-level analysis."""
        modules = [cls.ROLE_MODULE, cls.visual_frames_module()]

        if include_vqa and not questions:
            raise ValueError("Questions must be provided when VQA should be included.")

        if include_vqa:
            modules.append(cls.summary_vqa_task(ProcessingLevel.FRAME))
            modules.append(cls.questions_module(questions))
        else:
            modules.append(cls.summary_task())

        modules.append(cls.CONSTRAINTS_MODULE)
        return "\n\n".join(modules)

    @classmethod
    def build_clip_prompt(
        cls,
        frame_bullets: List[str],
        include_audio: bool = False,
        audio_transcription: Optional[List[Dict]] = None,
        include_vqa: bool = False,
        questions: Optional[List[str]] = None,
        vqa_bullets: Optional[List[str]] = None,
    ) -> str:
        """Build prompt for clip-level analysis."""
        modules = [cls.ROLE_MODULE, cls.visual_captions_module(frame_bullets)]

        if include_audio and audio_transcription:
            modules.append(cls.audio_module(audio_transcription))

        if include_vqa and questions:
            modules.append(
                cls.summary_vqa_task(ProcessingLevel.CLIP, has_audio=include_audio)
            )
            modules.append(cls.questions_module(questions))
            if vqa_bullets:
                modules.append(cls.vqa_context_module(vqa_bullets, is_final=False))
        else:
            modules.append(cls.summary_task(has_audio=include_audio))

        modules.append(cls.CONSTRAINTS_MODULE)
        return "\n\n".join(modules)

    @classmethod
    def build_video_prompt(
        cls,
        summary_only: bool = False,
        include_vqa: bool = False,
        clip_summaries: Optional[List[str]] = None,
        questions: Optional[List[str]] = None,
        vqa_bullets: Optional[List[str]] = None,
    ) -> str:
        """Build prompt for video-level analysis."""
        modules = [cls.ROLE_MODULE]

        if summary_only:
            modules.append(cls.visual_captions_final_module(clip_summaries))
            modules.append(cls.summary_task())
        elif include_vqa and questions:
            if vqa_bullets:
                modules.append(cls.visual_captions_final_module(clip_summaries))
                modules.append(cls.vqa_context_module(vqa_bullets, is_final=True))
                modules.append(cls.vqa_only_task())
                modules.append(cls.questions_module(questions))
            else:
                raise ValueError(
                    "vqa_bullets must be provided for VQA-only video prompt."
                )

        modules.append(cls.CONSTRAINTS_MODULE)
        return "\n\n".join(modules)
