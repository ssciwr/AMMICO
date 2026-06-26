from ammico.image_summary import ImageSummaryDetector
from ammico.test.conftest import MockInferenceModel
import os
from PIL import Image
import pytest


@pytest.mark.long
def test_image_summary_detector_summary_and_questions(model, get_testdict):
    """One Qwen load for summary + VQA (avoids loading the VL model twice in this module)."""
    detector = ImageSummaryDetector(summary_model=model, subdict=get_testdict)
    results = detector.analyse_images_from_dict(analysis_type="summary")
    assert len(results) == 2
    for key in get_testdict.keys():
        assert key in results
        assert "caption" in results[key]
        assert isinstance(results[key]["caption"], str)
        assert len(results[key]["caption"]) > 0

    list_of_questions = [
        "What is happening in the image?",
        "How many cars are in the image in total?",
    ]
    results = detector.analyse_images_from_dict(
        analysis_type="questions", list_of_questions=list_of_questions
    )
    assert len(results) == 2
    for key in get_testdict.keys():
        assert "vqa" in results[key]
        if key == "IMG_2746":
            assert (
                "marathon" in results[key]["vqa"][0].lower()
                or "competition" in results[key]["vqa"][0].lower()
            )

        if key == "IMG_2809":
            assert (
                "two" in results[key]["vqa"][1].lower() or "2" in results[key]["vqa"][1]
            )


def test_clean_list_of_questions(mock_model):
    list_of_questions = [
        "What is happening in the image?",
        "",
        "   ",
        None,
        "How many cars are in the image in total",
    ]
    detector = ImageSummaryDetector(summary_model=mock_model, subdict={})
    prompt = detector.token_prompt_config["default"]["questions"]["prompt"]
    cleaned_questions = detector._clean_list_of_questions(list_of_questions, prompt)
    assert len(cleaned_questions) == 2
    assert cleaned_questions[0] == "What is happening in the image?"
    assert cleaned_questions[1] == "How many cars are in the image in total?"
    prompt = detector.token_prompt_config["concise"]["questions"]["prompt"]
    cleaned_questions = detector._clean_list_of_questions(list_of_questions, prompt)
    assert len(cleaned_questions) == 2
    assert cleaned_questions[0] == prompt + "What is happening in the image?"
    assert cleaned_questions[1] == prompt + "How many cars are in the image in total?"


# Fast tests using mock model (no actual model loading)
def test_image_summary_detector_init_mock(mock_model, get_testdict):
    """Test detector initialization with mocked model."""
    detector = ImageSummaryDetector(summary_model=mock_model, subdict=get_testdict)
    assert detector.summary_model is mock_model
    assert len(detector.subdict) == 2


def test_load_pil_if_needed_string(mock_model):
    """Test loading image from file path."""
    detector = ImageSummaryDetector(summary_model=mock_model)
    # This will try to actually load a file, so we'll use a test image
    test_image_path = os.path.join(os.path.dirname(__file__), "data", "IMG_2746.png")
    if os.path.exists(test_image_path):
        img = detector._load_pil_if_needed(test_image_path)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"


def test_is_sequence_but_not_str(mock_model):
    """Test sequence detection utility."""
    detector = ImageSummaryDetector(summary_model=mock_model)
    assert detector._is_sequence_but_not_str([1, 2, 3]) is True
    assert detector._is_sequence_but_not_str("string") is False
    assert detector._is_sequence_but_not_str(b"bytes") is False
    assert (
        detector._is_sequence_but_not_str({"a": 1}) is False
    )  # dict is sequence-like but not handled as such


def test_validate_analysis_type(mock_model):
    """Test analysis type validation."""
    detector = ImageSummaryDetector(summary_model=mock_model)
    # Test valid types
    _, _, is_summary, is_questions = detector._validate_analysis_type(
        "summary", None, 10
    )
    assert is_summary is True
    assert is_questions is False

    _, _, is_summary, is_questions = detector._validate_analysis_type(
        "questions", ["What is this?"], 10
    )
    assert is_summary is False
    assert is_questions is True

    _, _, is_summary, is_questions = detector._validate_analysis_type(
        "summary_and_questions", ["What is this?"], 10
    )
    assert is_summary is True
    assert is_questions is True

    # Test invalid type
    with pytest.raises(ValueError):
        detector._validate_analysis_type("invalid", None, 10)

    # Test too many questions
    with pytest.raises(ValueError):
        detector._validate_analysis_type(
            "questions", ["Q" + str(i) for i in range(33)], 32
        )


class _RaisingInferenceModel(MockInferenceModel):
    """Mock whose chat/chat_batch always raise, to simulate a broken endpoint."""

    def chat(self, messages, max_new_tokens=256, n=1):
        raise RuntimeError("model 'mock-model' not found")

    def chat_batch(self, messages_batch, max_new_tokens=256):
        raise RuntimeError("model 'mock-model' not found")


def test_caption_failure_continues_and_warns(get_testdict):
    """A failing endpoint yields empty captions, a clear warning, and no abort."""
    detector = ImageSummaryDetector(
        summary_model=_RaisingInferenceModel(), subdict=get_testdict
    )
    with pytest.warns(UserWarning, match="Caption generation failed for .*mock-model"):
        results = detector.analyse_images_from_dict(analysis_type="summary")

    # every image is still processed and carries a (empty) caption key
    assert len(results) == len(get_testdict)
    for key in get_testdict:
        assert results[key]["caption"] == ""


def test_empty_caption_warns(get_testdict):
    """An endpoint returning blank content warns about an empty caption."""
    detector = ImageSummaryDetector(
        summary_model=MockInferenceModel(chat_return="   "), subdict=get_testdict
    )
    with pytest.warns(UserWarning, match="No caption produced for"):
        results = detector.analyse_images_from_dict(analysis_type="summary")
    for key in get_testdict:
        assert results[key]["caption"] == ""


def test_vqa_failure_continues_and_warns(get_testdict):
    """A failing endpoint yields empty vqa lists, a clear warning, and no abort."""
    detector = ImageSummaryDetector(
        summary_model=_RaisingInferenceModel(), subdict=get_testdict
    )
    with pytest.warns(UserWarning, match="VQA failed for .*mock-model"):
        results = detector.analyse_images_from_dict(
            analysis_type="questions", list_of_questions=["What is this?"]
        )
    assert len(results) == len(get_testdict)
    for key in get_testdict:
        assert results[key]["vqa"] == []
