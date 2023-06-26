import json
import ammico.display as ammico_display
import pytest


def test_explore_analysis_faces(get_path):
    mydict = {"IMG_2746": {"filename": get_path + "IMG_2746.png"}}
    with open(get_path + "example_faces.json", "r") as file:
        outs = json.load(file)
    mydict["IMG_2746"].pop("filename", None)
    for im_key in mydict.keys():
        sub_dict = mydict[im_key]
        for key in sub_dict.keys():
            assert sub_dict[key] == outs[key]


def test_explore_analysis_objects(get_path):
    mydict = {"IMG_2809": {"filename": get_path + "IMG_2809.png"}}
    with open(get_path + "example_analysis_objects.json", "r") as file:
        outs = json.load(file)
    mydict["IMG_2809"].pop("filename", None)
    for im_key in mydict.keys():
        sub_dict = mydict[im_key]
        for key in sub_dict.keys():
            assert sub_dict[key] == outs[key]


def test_AnalysisExplorer(get_path):
    path_img_1 = get_path + "IMG_2809.png"
    path_img_2 = get_path + "IMG_2746.png"

    mydict = {
        "IMG_2809": {"filename": path_img_1},
        "IMG_2746": {"filename": path_img_2},
    }

    # all_options_dict = {
    # path_img_1: "IMG_2809",
    # path_img_2: "IMG_2746",
    # }

    analysis_explorer = ammico_display.AnalysisExplorer(mydict)

    analysis_explorer.update_picture(path_img_1)

    assert analysis_explorer.update_picture(None) is None

    # analysis_explorer._right_output_analysis(
    #     2,
    #     all_options_dict,
    #     path_img_1,
    #     "ObjectDetector",
    #     True,
    #     50,
    #     50,
    #     "CIE 1976",
    #     "summary_and_questions",
    #     "base",
    #     "How many people are in the picture?",
    # )

    # analysis_explorer._right_output_analysis(
    #     2,
    #     all_options_dict,
    #     path_img_1,
    #     "EmotionDetector",
    #     True,
    #     50,
    #     50,
    #     "CIE 1976",
    #     "summary_and_questions",
    #     "base",
    #     "How many people are in the picture?",
    # )

    # analysis_explorer._right_output_analysis(
    #     2,
    #     all_options_dict,
    #     path_img_1,
    #     "SummaryDetector",
    #     True,
    #     50,
    #     50,
    #     "CIE 1976",
    #     "summary_and_questions",
    #     "base",
    #     "How many people are in the picture?",
    # )

    # analysis_explorer._right_output_analysis(
    #     2,
    #     all_options_dict,
    #     path_img_1,
    #     "ColorDetector",
    #     True,
    #     50,
    #     50,
    #     "CIE 1976",
    #     "summary_and_questions",
    #     "base",
    #     "How many people are in the picture?",
    # )

    with pytest.raises(EnvironmentError):
        analysis_explorer.run_server(port=8050)
