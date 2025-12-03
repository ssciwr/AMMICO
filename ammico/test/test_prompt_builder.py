from ammico import prompt_builder as pb
import pytest


@pytest.fixture
def setup_prompt_builder():
    return pb.PromptBuilder()


@pytest.fixture
def sample_audio_transcription():
    """Sample audio transcription data for testing."""
    return [
        {"start_time": 0.0, "end_time": 2.5, "text": "Knock knock"},
        {"start_time": 2.5, "end_time": 5.8, "text": "who's there?"},
        {"start_time": 5.8, "end_time": 9.2, "text": "Interrupting cow"},
        {"start_time": 9.2, "end_time": 12.0, "text": "Interrupting cow wh"},
    ]


@pytest.fixture
def sample_frame_bullets():
    return [
        "Frame shows a cat sitting on a windowsill.",
        "There is a sunny day outside.",
    ]


@pytest.fixture
def sample_vqa_bullets():
    return ["Answer 1", "Answer 2"]


@pytest.fixture
def sample_questions():
    return ["What is happening in the frame?", "Describe the main subject."]


def test_ProcessingLevel():
    assert len(pb.ProcessingLevel) == 3
    assert pb.ProcessingLevel.FRAME.value == "frame"
    assert pb.ProcessingLevel.CLIP.value == "clip"
    assert pb.ProcessingLevel.VIDEO.value == "video"
    assert pb.ProcessingLevel("frame") == pb.ProcessingLevel.FRAME


def test_PromptBuilder():
    assert "precise" in pb.PromptBuilder.ROLE_MODULE
    assert "Accuracy" in pb.PromptBuilder.CONSTRAINTS_MODULE


def tests_visual_frames_module(setup_prompt_builder):
    assert "frame" in setup_prompt_builder.visual_frames_module()


def test_visual_captions_module(setup_prompt_builder, sample_vqa_bullets):
    returned_string = setup_prompt_builder.visual_captions_module(
        frame_bullets=sample_vqa_bullets
    )
    assert sample_vqa_bullets[0] in returned_string
    assert sample_vqa_bullets[1] in returned_string
    assert "frames" in returned_string


def test_visual_captions_final_module(setup_prompt_builder):
    clip_summaries = "test summary"
    returned_string = setup_prompt_builder.visual_captions_final_module(
        clip_summaries=clip_summaries
    )
    assert clip_summaries in returned_string
    assert "video" in returned_string


def test_audio_module(setup_prompt_builder, sample_audio_transcription):
    returned_string = setup_prompt_builder.audio_module(
        audio_transcription=sample_audio_transcription
    )
    assert "audio" in returned_string
    assert "Knock knock" in returned_string
    assert "who's there?" in returned_string
    assert "Interrupting cow" in returned_string
    assert "Interrupting cow wh" in returned_string
    assert "[0.00s - 2.50s]: Knock knock" in returned_string


def test_summary_task(setup_prompt_builder):
    returned_string = setup_prompt_builder.summary_task()
    assert "Concise Summary" in returned_string
    assert "visual information" in returned_string
    returned_string = setup_prompt_builder.summary_task(has_audio=True)
    assert "Concise Summary" in returned_string
    assert "visual and audio information" in returned_string


def test_summary_vqa_task(setup_prompt_builder):
    returned_string = setup_prompt_builder.summary_vqa_task(
        level=pb.ProcessingLevel.FRAME
    )
    assert "task 1: Concise Summary" in returned_string
    assert "task 2: Question Answering" in returned_string
    assert "frame-level" not in returned_string
    returned_string = setup_prompt_builder.summary_vqa_task(
        level=pb.ProcessingLevel.CLIP
    )
    assert "task 1: Concise Summary" in returned_string
    assert "task 2: Question Answering" in returned_string
    assert "frame-level" in returned_string
    assert "1. **Frame-Level Answers**" in returned_string
    assert "2. **Visual Information**" in returned_string
    returned_string = setup_prompt_builder.summary_vqa_task(
        level=pb.ProcessingLevel.CLIP, has_audio=True
    )
    assert "task 1: Concise Summary" in returned_string
    assert "task 2: Question Answering" in returned_string
    assert "frame-level" in returned_string
    assert "1. **Frame-Level Answers**" in returned_string
    assert "2. **Audio Information**" in returned_string


def test_vqa_only_task(setup_prompt_builder):
    assert "Task: Answer Questions" in setup_prompt_builder.vqa_only_task()


def test_questions_module(setup_prompt_builder, sample_questions):
    returned_string = setup_prompt_builder.questions_module(questions=sample_questions)
    assert "1. " + sample_questions[0] in returned_string
    assert "2. " + sample_questions[1] in returned_string


def test_vqa_context_module(setup_prompt_builder, sample_vqa_bullets):
    returned_string = setup_prompt_builder.vqa_context_module(
        vqa_bullets=sample_vqa_bullets
    )
    assert sample_vqa_bullets[0] in returned_string
    assert sample_vqa_bullets[1] in returned_string
    assert "FRAME-Level" in returned_string
    returned_string = setup_prompt_builder.vqa_context_module(
        vqa_bullets=sample_vqa_bullets, is_final=True
    )
    assert sample_vqa_bullets[0] in returned_string
    assert sample_vqa_bullets[1] in returned_string
    assert "SEGMENT-Level" in returned_string


def test_build_frame_prompt(setup_prompt_builder, sample_questions):
    # summary only
    returned_string = setup_prompt_builder.build_frame_prompt()
    assert "precise" in returned_string
    assert "Accuracy" in returned_string
    assert "Task: Generate Concise Summary" in returned_string
    # VQA but no questions
    with pytest.raises(ValueError):
        setup_prompt_builder.build_frame_prompt(include_vqa=True)
    # VQA
    returned_string = setup_prompt_builder.build_frame_prompt(
        include_vqa=True, questions=sample_questions
    )
    assert "precise" in returned_string
    assert "Accuracy" in returned_string
    assert "Answer the provided questions" in returned_string
    assert "You have two tasks:" in returned_string


def test_build_clip_prompt(
    setup_prompt_builder,
    sample_frame_bullets,
    sample_vqa_bullets,
    sample_audio_transcription,
    sample_questions,
):
    # summary only
    returned_string = setup_prompt_builder.build_clip_prompt(sample_frame_bullets)
    assert "precise" in returned_string
    assert "Visual Information" in returned_string
    assert "frames" in returned_string
    assert sample_frame_bullets[0] in returned_string
    assert sample_frame_bullets[1] in returned_string
    assert "Task: Generate Concise Summary" in returned_string
    # summary with audio
    returned_string = setup_prompt_builder.build_clip_prompt(
        sample_frame_bullets,
        include_audio=True,
        audio_transcription=sample_audio_transcription,
    )
    assert "precise" in returned_string
    assert "Visual Information" in returned_string
    assert "frames" in returned_string
    assert sample_frame_bullets[0] in returned_string
    assert sample_frame_bullets[1] in returned_string
    assert "Audio Information" in returned_string
    assert "audio" in returned_string
    assert "[0.00s - 2.50s]: Knock knock" in returned_string
    # summary with VQA but no questions
    with pytest.raises(ValueError):
        setup_prompt_builder.build_clip_prompt(sample_frame_bullets, include_vqa=True)
    # summary with VQA
    returned_string = setup_prompt_builder.build_clip_prompt(
        sample_frame_bullets, include_vqa=True, questions=sample_questions
    )
    assert "precise" in returned_string
    assert "Visual Information" in returned_string
    assert "frames" in returned_string
    assert sample_frame_bullets[0] in returned_string
    assert sample_frame_bullets[1] in returned_string
    assert "Frame-Level Answers" in returned_string
    assert "You have two tasks:" in returned_string
    assert "Questions to Answer" in returned_string
    assert "1. " + sample_questions[0] in returned_string
    assert "2. " + sample_questions[1] in returned_string
    # summary with VQA and audio
    returned_string = setup_prompt_builder.build_clip_prompt(
        sample_frame_bullets,
        include_vqa=True,
        questions=sample_questions,
        include_audio=True,
        audio_transcription=sample_audio_transcription,
    )
    assert "precise" in returned_string
    assert "Visual Information" in returned_string
    assert "frames" in returned_string
    assert sample_frame_bullets[0] in returned_string
    assert sample_frame_bullets[1] in returned_string
    assert "Audio Information" in returned_string
    assert "audio" in returned_string
    assert "[0.00s - 2.50s]: Knock knock" in returned_string
    assert "Frame-Level Answers" in returned_string
    assert "Audio Information" in returned_string
    assert "You have two tasks:" in returned_string
    assert "Questions to Answer" in returned_string
    assert "1. " + sample_questions[0] in returned_string
    assert "2. " + sample_questions[1] in returned_string


def test_build_video_prompt(
    setup_prompt_builder, sample_frame_bullets, sample_questions, sample_vqa_bullets
):
    # summary only
    returned_string = setup_prompt_builder.build_video_prompt(summary_only=True)
    assert "precise" in returned_string
    assert "Accuracy" in returned_string
    assert "Task: Generate Concise Summary" in returned_string
