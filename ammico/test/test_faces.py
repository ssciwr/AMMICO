import ammico.faces as fc
import json
import pytest
import os


def test_init_EmotionDetector(monkeypatch):
    # standard input
    monkeypatch.setattr("builtins.input", lambda _: "yes")
    ed = fc.EmotionDetector({}, accept_disclaimer="OTHER_VAR")
    assert ed.subdict["face"] == "No"
    assert ed.subdict["multiple_faces"] == "No"
    assert ed.subdict["wears_mask"] == ["No"]
    assert ed.subdict["emotion"] == [None]
    assert ed.subdict["age"] == [None]
    assert ed.emotion_threshold == 50
    assert ed.age_threshold == 50
    assert ed.gender_threshold == 50
    assert ed.race_threshold == 50
    assert ed.emotion_categories["angry"] == "Negative"
    assert ed.emotion_categories["happy"] == "Positive"
    assert ed.emotion_categories["surprise"] == "Neutral"
    assert ed.accept_disclaimer == "OTHER_VAR"
    assert os.environ.get(ed.accept_disclaimer) == "True"
    assert ed.accepted
    monkeypatch.delenv(ed.accept_disclaimer, raising=False)
    # different thresholds
    ed = fc.EmotionDetector(
        {},
        emotion_threshold=80,
        race_threshold=30,
        gender_threshold=70,
        age_threshold=90,
        accept_disclaimer="OTHER_VAR",
    )
    assert ed.emotion_threshold == 80
    assert ed.race_threshold == 30
    assert ed.gender_threshold == 70
    assert ed.age_threshold == 90
    monkeypatch.delenv(ed.accept_disclaimer, raising=False)
    # do not accept disclaimer
    monkeypatch.setattr("builtins.input", lambda _: "no")
    ed = fc.EmotionDetector({}, accept_disclaimer="OTHER_VAR")
    assert os.environ.get(ed.accept_disclaimer) == "False"
    assert not ed.accepted
    monkeypatch.delenv(ed.accept_disclaimer, raising=False)
    # now test the exceptions: thresholds
    monkeypatch.setattr("builtins.input", lambda _: "yes")
    with pytest.raises(ValueError):
        fc.EmotionDetector({}, emotion_threshold=150)
    with pytest.raises(ValueError):
        fc.EmotionDetector({}, emotion_threshold=-50)
    with pytest.raises(ValueError):
        fc.EmotionDetector({}, race_threshold=150)
    with pytest.raises(ValueError):
        fc.EmotionDetector({}, race_threshold=-50)
    with pytest.raises(ValueError):
        fc.EmotionDetector({}, gender_threshold=150)
    with pytest.raises(ValueError):
        fc.EmotionDetector({}, gender_threshold=-50)
    with pytest.raises(ValueError):
        fc.EmotionDetector({}, age_threshold=150)
    with pytest.raises(ValueError):
        fc.EmotionDetector({}, age_threshold=-50)
    # test pre-set variables: disclaimer
    monkeypatch.delattr("builtins.input", raising=False)
    monkeypatch.setenv("OTHER_VAR", "something")
    ed = fc.EmotionDetector({}, accept_disclaimer="OTHER_VAR")
    assert not ed.accepted
    monkeypatch.setenv("OTHER_VAR", "False")
    ed = fc.EmotionDetector({}, accept_disclaimer="OTHER_VAR")
    assert not ed.accepted
    monkeypatch.setenv("OTHER_VAR", "True")
    ed = fc.EmotionDetector({}, accept_disclaimer="OTHER_VAR")
    assert ed.accepted


def test_define_actions(monkeypatch):
    monkeypatch.setenv("OTHER_VAR", "True")
    ed = fc.EmotionDetector({}, accept_disclaimer="OTHER_VAR")
    actions = ed._define_actions({"wears_mask": True})
    assert actions == ["age", "gender"]
    actions = ed._define_actions({"wears_mask": False})
    assert actions == ["age", "gender", "race", "emotion"]
    monkeypatch.setenv("OTHER_VAR", "False")
    ed = fc.EmotionDetector({}, accept_disclaimer="OTHER_VAR")
    actions = ed._define_actions({"wears_mask": True})
    assert actions == []
    actions = ed._define_actions({"wears_mask": False})
    assert actions == ["emotion"]


def test_ensure_deepface_models(monkeypatch):
    monkeypatch.setenv("OTHER_VAR", "True")
    ed = fc.EmotionDetector({}, accept_disclaimer="OTHER_VAR")
    ed._ensure_deepface_models(["age", "gender", "race", "emotion"])


def test_analyse_faces(get_path, monkeypatch):
    mydict = {
        "filename": get_path + "pexels-pixabay-415829.jpg",
    }
    monkeypatch.setenv("OTHER_VAR", "True")
    mydict.update(
        fc.EmotionDetector(mydict, accept_disclaimer="OTHER_VAR").analyse_image()
    )

    with open(get_path + "example_faces.json", "r") as file:
        out_dict = json.load(file)
    # delete the filename key
    mydict.pop("filename", None)
    # do not test for age, as this is not a reliable metric
    mydict.pop("age", None)
    for key in mydict.keys():
        assert mydict[key] == out_dict[key]
