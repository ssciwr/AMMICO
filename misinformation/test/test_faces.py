import os
import pytest
import misinformation
import misinformation.faces as fc


def test_analyse_faces():
    mydict = {
        "image_faces": {"filename": "./misinformation/test/data/image_faces.jpg"},
        "image_objects": {"filename": "./misinformation/test/data/image_objects.jpg"},
    }
    for key in mydict.keys():
        mydict[key] = fc.EmotionDetector(mydict[key]).analyse_image()
    print(mydict)

    with open("./misinformation/test/data/example_faces.txt", "r") as file:
        out_dict = file.read()

    assert str(mydict) == out_dict
