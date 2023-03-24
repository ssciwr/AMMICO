import json
import misinformation.display as misinf_display
import pytest


def test_explore_analysis_faces(get_path):
    mydict = {"IMG_2746": {"filename": get_path + "IMG_2746.png"}}
    misinf_display.explore_analysis(mydict, identify="faces")
    with open(get_path + "example_faces.json", "r") as file:
        outs = json.load(file)
    mydict["IMG_2746"].pop("filename", None)
    for im_key in mydict.keys():
        sub_dict = mydict[im_key]
        for key in sub_dict.keys():
            assert sub_dict[key] == outs[key]


def test_explore_analysis_objects(get_path):
    mydict = {"IMG_2809": {"filename": get_path + "IMG_2809.png"}}
    misinf_display.explore_analysis(mydict, identify="objects")
    with open(get_path + "example_analysis_objects.json", "r") as file:
        outs = json.load(file)
    mydict["IMG_2809"].pop("filename", None)
    print(mydict)
    for im_key in mydict.keys():
        sub_dict = mydict[im_key]
        for key in sub_dict.keys():
            print(outs[key])
            assert sub_dict[key] == outs[key]
