import ammico.faces as fc
import json
import pytest
import os


def test_init_EmotionDetector(monkeypatch):
    # standard input
    monkeypatch.setattr("builtins.input", lambda _: "yes")
    ed = fc.EmotionDetector({}, accept_disclosure="OTHER_VAR")
    assert ed.subdict["face"] == "No"
    assert ed.subdict["multiple_faces"] == "No"
    assert ed.subdict["wears_mask"] == ["No"]
    assert ed.emotion_threshold == 50
    assert ed.race_threshold == 50
    assert ed.gender_threshold == 50
    assert ed.emotion_categories["angry"] == "Negative"
    assert ed.emotion_categories["happy"] == "Positive"
    assert ed.emotion_categories["surprise"] == "Neutral"
    assert ed.accepted
    monkeypatch.delenv("OTHER_VAR", raising=False)
    # different thresholds
    ed = fc.EmotionDetector(
        {},
        emotion_threshold=80,
        race_threshold=30,
        gender_threshold=60,
        accept_disclosure="OTHER_VAR",
    )
    assert ed.emotion_threshold == 80
    assert ed.race_threshold == 30
    assert ed.gender_threshold == 60
    monkeypatch.delenv("OTHER_VAR", raising=False)
    # do not accept disclosure
    monkeypatch.setattr("builtins.input", lambda _: "no")
    ed = fc.EmotionDetector({}, accept_disclosure="OTHER_VAR")
    assert os.environ.get("OTHER_VAR") == "False"
    assert not ed.accepted
    monkeypatch.delenv("OTHER_VAR", raising=False)
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
    # test pre-set variables: disclosure
    monkeypatch.delattr("builtins.input", raising=False)
    monkeypatch.setenv("OTHER_VAR", "something")
    ed = fc.EmotionDetector({}, accept_disclosure="OTHER_VAR")
    assert not ed.accepted
    monkeypatch.setenv("OTHER_VAR", "False")
    ed = fc.EmotionDetector({}, accept_disclosure="OTHER_VAR")
    assert not ed.accepted
    monkeypatch.setenv("OTHER_VAR", "True")
    ed = fc.EmotionDetector({}, accept_disclosure="OTHER_VAR")
    assert ed.accepted


def test_define_actions(monkeypatch):
    monkeypatch.setenv("OTHER_VAR", "True")
    ed = fc.EmotionDetector({}, accept_disclosure="OTHER_VAR")
    ed._define_actions({"wears_mask": True})
    assert ed.actions == ["age"]
    ed._define_actions({"wears_mask": False})
    assert ed.actions == ["age", "gender", "race", "emotion"]
    monkeypatch.setenv("OTHER_VAR", "False")
    ed = fc.EmotionDetector({}, accept_disclosure="OTHER_VAR")
    ed._define_actions({"wears_mask": True})
    assert ed.actions == []
    ed._define_actions({"wears_mask": False})
    assert ed.actions == ["emotion"]


def test_ensure_deepface_models(monkeypatch):
    monkeypatch.setenv("OTHER_VAR", "True")
    ed = fc.EmotionDetector({}, accept_disclosure="OTHER_VAR")
    ed.actions = ["age", "gender", "race", "emotion"]
    ed._ensure_deepface_models()


def test_analyse_faces(get_path, monkeypatch):
    mydict = {
        # one face, no mask
        "pexels-pixabay-415829": {"filename": get_path + "pexels-pixabay-415829.jpg"},
        # two faces, no mask
        "pexels-1000990-1954659": {"filename": get_path + "pexels-1000990-1954659.jpg"},
        # one face, mask
        "pexels-maksgelatin-4750169": {
            "filename": get_path + "pexels-maksgelatin-4750169.jpg"
        },
    }
    monkeypatch.setenv("OTHER_VAR", "True")
    for key in mydict.keys():
        mydict[key].update(
            fc.EmotionDetector(
                mydict[key], emotion_threshold=80, accept_disclosure="OTHER_VAR"
            ).analyse_image()
        )

    with open(get_path + "example_faces.json", "r") as file:
        out_dict = json.load(file)

    for key in mydict.keys():
        # delete the filename key
        mydict[key].pop("filename", None)
        # do not test for age, as this is not a reliable metric
        mydict[key].pop("age", None)
        for subkey in mydict[key].keys():
            assert mydict[key][subkey] == out_dict[key][subkey]
