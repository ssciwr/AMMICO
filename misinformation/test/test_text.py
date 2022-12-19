import os
import pytest
import spacy
import misinformation.text as tt

TESTDICT = {
    "IMG_3755": {
        "filename": "./test/data/IMG_3755.jpg",
    },
    "IMG_3756": {
        "filename": "./test/data/IMG_3756.jpg",
    },
    "IMG_3757": {
        "filename": "./test/data/IMG_3757.jpg",
    },
}

os.environ[
    "GOOGLE_APPLICATION_CREDENTIALS"
] = "../data/seismic-bonfire-329406-412821a70264.json"


def test_TextDetector():
    for item in TESTDICT:
        test_obj = tt.TextDetector(TESTDICT[item])
        assert test_obj.subdict["text"] is None
        assert test_obj.subdict["text_language"] is None
        assert test_obj.subdict["text_english"] is None
        assert test_obj.subdict["text_cleaned"] is None


@pytest.mark.gcv
def test_get_text_from_image():
    for item in TESTDICT:
        test_obj = tt.TextDetector(TESTDICT[item])
        test_obj.get_text_from_image()
        ref_file = "./test/data/text_" + item + ".txt"
        with open(ref_file, "r") as file:
            reference_text = file.read()
        assert test_obj.subdict["text"] == reference_text


def test_translate_text():
    for item in TESTDICT:
        test_obj = tt.TextDetector(TESTDICT[item])
        ref_file = "./test/data/text_" + item + ".txt"
        with open(ref_file, "r") as file:
            reference_text = file.read()
        test_obj.subdict["text"] = reference_text
        test_obj.translate_text()
        print("-----")
        print(test_obj.subdict["text_language"])
        print("-----")
        print(test_obj.subdict["text_english"])
        print("-----")


def test_init_spacy():
    test_obj = tt.TextDetector(TESTDICT["IMG_3755"])
    ref_file = "./test/data/text_IMG_3755.txt"
    with open(ref_file, "r") as file:
        reference_text = file.read()
    test_obj.subdict["text_english"] = reference_text
    test_obj._init_spacy()
    assert isinstance(test_obj.doc, spacy.tokens.doc.Doc)


def test_clean_text():
    nlp = spacy.load("en_core_web_md")
    doc = nlp("I like cats and fjejg")
    test_obj = tt.TextDetector(TESTDICT["IMG_3755"])
    test_obj.doc = doc
    test_obj.clean_text()
    result = "I like cats and"
    assert test_obj.subdict["text_clean"] == result
