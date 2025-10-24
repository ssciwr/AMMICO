from ammico.image_summary import ImageSummaryDetector

import pytest


@pytest.mark.long
def test_image_summary_detector(model, get_testdict):
    detector = ImageSummaryDetector(summary_model=model, subdict=get_testdict)
    results = detector.analyse_images_from_dict(analysis_type="summary")
    assert len(results) == 2
    for key in get_testdict.keys():
        assert key in results
        assert "caption" in results[key]
        assert isinstance(results[key]["caption"], str)
        assert len(results[key]["caption"]) > 0


@pytest.mark.long
def test_image_summary_detector_questions(model, get_testdict):
    list_of_questions = [
        "What is happening in the image?",
        "How many cars are in the image in total?",
    ]
    detector = ImageSummaryDetector(summary_model=model, subdict=get_testdict)
    results = detector.analyse_images_from_dict(
        analysis_type="questions", list_of_questions=list_of_questions
    )
    assert len(results) == 2
    for key in get_testdict.keys():
        assert "vqa" in results[key]
        if key == "IMG_2746":
            assert "marathon" in results[key]["vqa"][0].lower()

        if key == "IMG_2809":
            assert (
                "two" in results[key]["vqa"][1].lower() or "2" in results[key]["vqa"][1]
            )


def test_clean_list_of_questions(model):
    list_of_questions = [
        "What is happening in the image?",
        "",
        "   ",
        None,
        "How many cars are in the image in total",
    ]
    detector = ImageSummaryDetector(summary_model=model, subdict={})
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
