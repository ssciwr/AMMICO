import json
import ammico.display as ammico_display
import pytest


@pytest.fixture
def get_options(get_path):
    path_img_1 = get_path + "IMG_2809.png"
    path_img_2 = get_path + "IMG_2746.png"

    mydict = {
        "IMG_2809": {"filename": path_img_1},
        "IMG_2746": {"filename": path_img_2},
    }

    all_options_dict = {
        path_img_1: "IMG_2809",
        path_img_2: "IMG_2746",
    }
    return path_img_1, path_img_2, mydict, all_options_dict


@pytest.fixture
def get_AE(get_options):
    analysis_explorer = ammico_display.AnalysisExplorer(get_options[2])
    return analysis_explorer


def test_explore_analysis_faces(get_path):
    mydict = {"IMG_2746": {"filename": get_path + "IMG_2746.png"}}
    with open(get_path + "example_faces.json", "r") as file:
        outs = json.load(file)
    mydict["IMG_2746"].pop("filename", None)
    for im_key in mydict.keys():
        sub_dict = mydict[im_key]
        for key in sub_dict.keys():
            assert sub_dict[key] == outs[key]


def test_AnalysisExplorer(get_AE, get_options):
    get_AE.update_picture(get_options[0])
    assert get_AE.update_picture(None) is None


def test_right_output_analysis_summary(get_AE, get_options, monkeypatch):
    monkeypatch.setenv("SOME_VAR", "True")
    monkeypatch.setenv("OTHER_VAR", "True")
    get_AE._right_output_analysis(
        2,
        get_options[3],
        get_options[0],
        "SummaryDetector",
        True,
        None,
        None,
        "SOME_VAR",
        50,
        50,
        50,
        "OTHER_VAR",
        "CIE 1976",
        "summary_and_questions",
        "base",
        "How many people are in the picture?",
    )


def test_right_output_analysis_emotions(get_AE, get_options, monkeypatch):
    monkeypatch.setenv("SOME_VAR", "True")
    monkeypatch.setenv("OTHER_VAR", "True")
    get_AE._right_output_analysis(
        2,
        get_options[3],
        get_options[0],
        "EmotionDetector",
        True,
        None,
        None,
        "SOME_VAR",
        50,
        50,
        50,
        "OTHER_VAR",
        "CIE 1976",
        "summary_and_questions",
        "base",
        "How many people are in the picture?",
    )
