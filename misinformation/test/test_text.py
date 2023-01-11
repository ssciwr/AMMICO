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

LANGUAGES = ["de", "om", "en"]

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
        assert not test_obj.analyse_text
        assert not test_obj.analyse_topic


@pytest.mark.gcv
def test_get_text_from_image():
    for item in TESTDICT:
        test_obj = tt.TextDetector(TESTDICT[item])
        test_obj.get_text_from_image()
        ref_file = "./test/data/text_" + item + ".txt"
        with open(ref_file, "r", encoding="utf8") as file:
            reference_text = file.read()
        assert test_obj.subdict["text"] == reference_text


def test_translate_text():
    for item, lang in zip(TESTDICT, LANGUAGES):
        test_obj = tt.TextDetector(TESTDICT[item])
        ref_file = "./test/data/text_" + item + ".txt"
        trans_file = "./test/data/text_translated_" + item + ".txt"
        with open(ref_file, "r", encoding="utf8") as file:
            reference_text = file.read()
        with open(trans_file, "r", encoding="utf8") as file:
            translated_text = file.read()
        test_obj.subdict["text"] = reference_text
        test_obj.translate_text()
        assert test_obj.subdict["text_language"] == lang
        assert test_obj.subdict["text_english"] == translated_text


def test_init_spacy():
    test_obj = tt.TextDetector(TESTDICT["IMG_3755"], analyse_text=True)
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


def test_correct_spelling():
    mydict = {}
    test_obj = tt.TextDetector(mydict, analyse_text=True)
    test_obj.subdict["text_english"] = "I lik cats ad dogs."
    test_obj.correct_spelling()
    result = "I like cats ad dogs."
    assert test_obj.subdict["text_english_correct"] == result
