import os
import pytest
from torch import device, cuda
from lavis.models import load_model_and_preprocess
import misinformation.summary as sm

IMAGES = ["d755771b-225e-432f-802e-fb8dc850fff7.png", "IMG_2746.png"]

SUMMARY_DEVICE = device("cuda" if cuda.is_available() else "cpu")

TEST_KWARGS = {
    "run1": {},
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


@pytest.mark.long
def test_analyse_image(get_dict):
    for key in get_dict:
        get_dict[key] = sm.SummaryDetector(get_dict[key]).analyse_image()
    keys = list(get_dict.keys())
    assert len(get_dict) == 2
    for key in keys:
        assert len(get_dict[key]["3_non-deterministic summary"]) == 3

    const_image_summary_list = [
        "a river running through a city next to tall buildings",
        "a crowd of people standing on top of a tennis court",
    ]

    for i in range(len(const_image_summary_list)):
        assert get_dict[keys[i]]["const_image_summary"] == const_image_summary_list[i]

    del sm.SummaryDetector.summary_model, sm.SummaryDetector.summary_vis_processors
    cuda.empty_cache()

    summary_model, summary_vis_processors, _ = load_model_and_preprocess(
        **TEST_KWARGS["run2"]
    )

    for key in get_dict:
        get_dict[key] = sm.SummaryDetector(get_dict[key]).analyse_image(
            summary_model, summary_vis_processors
        )
    keys = list(get_dict.keys())

    assert len(get_dict) == 2
    for key in keys:
        assert len(get_dict[key]["3_non-deterministic summary"]) == 3

    const_image_summary_list2 = [
        "a river running through a city next to tall buildings",
        "a crowd of people standing on top of a tennis court",
    ]

    for i in range(len(const_image_summary_list2)):
        assert get_dict[keys[i]]["const_image_summary"] == const_image_summary_list2[i]

    del summary_model, summary_vis_processors
    cuda.empty_cache()

    summary_model, summary_vis_processors, _ = load_model_and_preprocess(
        **TEST_KWARGS["run3"]
    )

    for key in get_dict:
        get_dict[key] = sm.SummaryDetector(get_dict[key]).analyse_image(
            summary_model, summary_vis_processors
        )
    keys = list(get_dict.keys())
    assert len(get_dict) == 2
    for key in keys:
        assert len(get_dict[key]["3_non-deterministic summary"]) == 3

    const_image_summary_list3 = [
        "a river running through a town next to tall buildings",
        "a crowd of people standing on top of a track",
    ]

    for i in range(len(const_image_summary_list2)):
        assert get_dict[keys[i]]["const_image_summary"] == const_image_summary_list3[i]


@pytest.mark.long
def test_analyse_questions(get_dict):
    list_of_questions = [
        "How many persons on the picture?",
        "What happends on the picture?",
    ]
    for key in get_dict:
        get_dict[key] = sm.SummaryDetector(get_dict[key]).analyse_questions(
            list_of_questions
        )
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
