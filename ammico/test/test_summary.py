import os
import pytest
from torch import device, cuda
from lavis.models import load_model_and_preprocess
import ammico.summary as sm


IMAGES = ["d755771b-225e-432f-802e-fb8dc850fff7.png", "IMG_2746.png"]

SUMMARY_DEVICE = device("cuda" if cuda.is_available() else "cpu")

TEST_KWARGS = {
    "run1": {
        "name": "blip_caption",
        "model_type": "base_coco",
        "is_eval": True,
        "device": SUMMARY_DEVICE,
    },
    "run2": {
        "name": "blip_caption",
        "model_type": "base_coco",
        "is_eval": True,
        "device": SUMMARY_DEVICE,
    },
    "run3": {
        "name": "blip_caption",
        "model_type": "large_coco",
        "is_eval": True,
        "device": SUMMARY_DEVICE,
    },
}


@pytest.fixture
def get_dict(get_path):
    mydict = {}
    for img in IMAGES:
        id_ = os.path.splitext(os.path.basename(img))[0]
        mydict[id_] = {"filename": get_path + img}
    return mydict


# @pytest.mark.long
def test_analyse_image(get_dict):
    reference_results = {
        "run1": {
            "d755771b-225e-432f-802e-fb8dc850fff7": "a river running through a city next to tall buildings",
            "IMG_2746": "a crowd of people standing on top of a tennis court",
        },
        "run2": {
            "d755771b-225e-432f-802e-fb8dc850fff7": "a river running through a city next to tall buildings",
            "IMG_2746": "a crowd of people standing on top of a tennis court",
        },
        "run3": {
            "d755771b-225e-432f-802e-fb8dc850fff7": "a river running through a town next to tall buildings",
            "IMG_2746": "a crowd of people standing on top of a track",
        },
    }
    # test three different models
    for test_run in TEST_KWARGS.keys():
        summary_model, summary_vis_processors, _ = load_model_and_preprocess(
            **TEST_KWARGS[test_run]
        )
        # run two different images
        for key in get_dict.keys():
            get_dict[key] = sm.SummaryDetector(
                get_dict[key],
                analysis_type="summary",
                summary_model=summary_model,
                summary_vis_processors=summary_vis_processors,
            ).analyse_image()
        assert len(get_dict) == 2
        for key in get_dict.keys():
            assert len(get_dict[key]["3_non-deterministic_summary"]) == 3
            assert (
                get_dict[key]["const_image_summary"] == reference_results[test_run][key]
            )
        cuda.empty_cache()
        summary_model = None
        summary_vis_processors = None


@pytest.mark.win_skip
def test_analyse_questions(get_dict):
    list_of_questions = [
        "How many persons on the picture?",
        "What happends on the picture?",
    ]
    for key in get_dict:
        get_dict[key] = sm.SummaryDetector(
            get_dict[key],
            analysis_type="questions",
            list_of_questions=list_of_questions,
        ).analyse_image()
    assert len(get_dict) == 2
    list_of_questions_ans = ["2", "100"]
    list_of_questions_ans2 = ["flood", "festival"]
    test_answers = []
    test_answers2 = []
    for key in get_dict.keys():
        test_answers.append(get_dict[key][list_of_questions[0]])
        test_answers2.append(get_dict[key][list_of_questions[1]])
    assert sorted(test_answers) == sorted(list_of_questions_ans)
    assert sorted(test_answers2) == sorted(list_of_questions_ans2)


def test_init_summary():
    sd = sm.SummaryDetector({}, analysis_type="summary")
    assert sd.analysis_type == "summary"
    with pytest.raises(ValueError):
        sm.SummaryDetector({}, analysis_type="something")
    list_of_questions = ["Question 1", "Question 2"]
    sd = sm.SummaryDetector({}, list_of_questions=list_of_questions)
    assert sd.list_of_questions == list_of_questions
    with pytest.raises(ValueError):
        sm.SummaryDetector({}, list_of_questions={})
    with pytest.raises(ValueError):
        sm.SummaryDetector({}, list_of_questions=[None])
    with pytest.raises(ValueError):
        sm.SummaryDetector({}, list_of_questions=[0.1])


@pytest.mark.long
def test_advanced_init_summary():
    sd = sm.SummaryDetector({})
    assert sd.summary_model
    assert sd.summary_vis_processors
    sd = sm.SummaryDetector({}, model_type="large")
    assert sd.summary_model
    assert sd.summary_vis_processors
    with pytest.raises(ValueError):
        sm.SummaryDetector({}, model_type="bla")
    (
        summary_vqa_model,
        summary_vqa_vis_processors,
        summary_vqa_txt_processors,
    ) = load_model_and_preprocess(
        name="blip_vqa",
        model_type="vqav2",
        is_eval=True,
        device="cpu",
    )
    sd = sm.SummaryDetector(
        {},
        summary_vqa_model=summary_vqa_model,
        summary_vqa_vis_processors=summary_vqa_vis_processors,
        summary_vqa_txt_processors=summary_vqa_txt_processors,
    )
    assert sd.summary_vqa_model
    assert sd.summary_vqa_vis_processors
    assert sd.summary_vqa_txt_processors
