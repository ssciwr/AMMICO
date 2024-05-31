import ammico.faces as fc
import json
import pytest


def test_set_keys():
    ed = fc.EmotionDetector({})
    assert ed.subdict["face"] == "No"
    assert ed.subdict["multiple_faces"] == "No"
    assert ed.subdict["wears_mask"] == ["No"]
    assert ed.subdict["emotion"] == [None]
    with pytest.raises(ValueError):
        fc.EmotionDetector({}, emotion_threshold=150)
    with pytest.raises(ValueError):
        fc.EmotionDetector({}, emotion_threshold=-50)
    with pytest.raises(ValueError):
        fc.EmotionDetector({}, race_threshold=150)
    with pytest.raises(ValueError):
        fc.EmotionDetector({}, race_threshold=-50)


def test_analyse_faces(get_path):
    mydict = {
        "filename": get_path + "pexels-pixabay-415829.jpg",
    }
    mydict.update(fc.EmotionDetector(mydict).analyse_image())

    with open(get_path + "example_faces.json", "r") as file:
        out_dict = json.load(file)
    # delete the filename key
    mydict.pop("filename", None)
    # delete the age key, as this is conflicting - gives different results sometimes
    mydict.pop("age", None)
    for key in mydict.keys():
        assert mydict[key] == out_dict[key]
