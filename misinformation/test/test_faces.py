import misinformation.faces as fc
import json
from pytest import approx


def test_analyse_faces():
    mydict = {
        "filename": "./test/data/IMG_2746.png",
    }
    mydict = fc.EmotionDetector(mydict).analyse_image()

    with open("./test/data/example_faces.json", "r") as file:
        out_dict = json.load(file)

    for key in mydict.keys():
        assert mydict[key] == out_dict[key]
