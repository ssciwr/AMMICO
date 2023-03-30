import misinformation.faces as fc
import json
import pytest


def test_analyse_faces(get_path):
    mydict = {
        "filename": get_path + "IMG_2746.png",
    }
    mydict.update(fc.EmotionDetector(mydict).analyse_image())

    with open(get_path + "example_faces.json", "r") as file:
        out_dict = json.load(file)
    # delete the filename key
    mydict.pop("filename", None)
    for key in mydict.keys():
        assert mydict[key] == out_dict[key]
