import json
from misinformation.display import explore_analysis
from pytest import approx


def test_explore_analysis_faces():
    mydict = {"IMG_2746": {"filename": "./test/data/IMG_2746.png"}}
    explore_analysis(mydict, identify="faces")
    with open("./test/data/example_faces.json", "r") as file:
        outs = json.load(file)

    for im_key in mydict.keys():
        sub_dict = mydict[im_key]
        for key in sub_dict.keys():
            assert sub_dict[key] == outs[key]


def test_explore_analysis_objects():
    mydict = {"IMG_2746": {"filename": "./test/data/IMG_2809.png"}}
    explore_analysis(mydict, identify="objects")
    with open("./test/data/example_analysis_objects.json", "r") as file:
        outs = json.load(file)

    assert str(mydict) == str(outs)
