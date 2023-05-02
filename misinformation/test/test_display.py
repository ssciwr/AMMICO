import json
import misinformation.display as misinf_display
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

    all_options_dict = {
        path_img_1: "IMG_2809",
        path_img_2: "IMG_2746",
    }

    analysis_explorer_faces = misinf_display.AnalysisExplorer(mydict, identify="faces")
    analysis_explorer_objects = misinf_display.AnalysisExplorer(
        mydict, identify="objects"
    )

    analysis_explorer_faces.update_picture(path_img_1)
    analysis_explorer_objects.update_picture(path_img_2)

    assert analysis_explorer_objects.update_picture(None) is None

    analysis_explorer_faces._right_output_analysis(
        None, ["something", "faces"], all_options_dict, path_img_1
    )
    analysis_explorer_objects._right_output_analysis(
        None, ["something", "objects"], all_options_dict, path_img_2
    )

    with pytest.raises(EnvironmentError):
        analysis_explorer_faces.run_server(port=8050)
    with pytest.raises(EnvironmentError):
        analysis_explorer_objects.run_server(port=8050)
