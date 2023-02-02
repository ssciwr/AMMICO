import os
import pytest
import spacy
import misinformation.text as tt
import misinformation
import pandas as pd

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
        assert not test_obj.analyse_text


@pytest.mark.gcv
def test_analyse_image():
    for item in TESTDICT:
        test_obj = tt.TextDetector(TESTDICT[item])
        test_obj.analyse_image()
        test_obj = tt.TextDetector(TESTDICT[item], analyse_text=True)
        test_obj.analyse_image()


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


def test_remove_linebreaks():
    test_obj = tt.TextDetector({})
    test_obj.subdict["text"] = "This is \n a test."
    test_obj.subdict["text_english"] = "This is \n another\n test."
    test_obj.remove_linebreaks()
    assert test_obj.subdict["text"] == "This is   a test."
    assert test_obj.subdict["text_english"] == "This is   another  test."


def test_run_spacy():
    test_obj = tt.TextDetector(TESTDICT["IMG_3755"], analyse_text=True)
    ref_file = "./test/data/text_IMG_3755.txt"
    with open(ref_file, "r") as file:
        reference_text = file.read()
    test_obj.subdict["text_english"] = reference_text
    test_obj._run_spacy()
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


def test_sentiment_analysis():
    mydict = {}
    test_obj = tt.TextDetector(mydict, analyse_text=True)
    test_obj.subdict["text_english"] = "I love cats and dogs."
    test_obj._run_spacy()
    test_obj.correct_spelling()
    test_obj.sentiment_analysis()
    assert test_obj.subdict["polarity"] == 0.5
    assert test_obj.subdict["subjectivity"] == 0.6


def test_PostprocessText():
    reference_dict = [
        None,
        "SCATTERING THEORY\nThe Quantum Theory of\nNonrelativistic Collisions\nJOHN R. TAYLOR\nUniversity of Colorado\nostaliga Lanbidean\n1 ilde\nballoons big goin\ngdĐOL, SIVI 23 TL\nthere in obl\noch yd change\na\nBer\nook Sy-RW isn't going anywhere",
        "THE\nALGEBRAIC\nEIGENVALUE\nPROBLEM\nDOM\nNVS TIO\nMINA\nMonographs\non Numerical Analysis\nJ.. H. WILKINSON",
    ]
    reference_df = [
        "Mathematische Formelsammlung\nfür Ingenieure und Naturwissenschaftler\nMit zahlreichen Abbildungen und Rechenbeispielen\nund einer ausführlichen Integraltafel\n3., verbesserte Auflage",
        "SCATTERING THEORY\nThe Quantum Theory of\nNonrelativistic Collisions\nJOHN R. TAYLOR\nUniversity of Colorado\nostaliga Lanbidean\n1 ilde\nballoons big goin\ngdĐOL, SIVI 23 TL\nthere in obl\noch yd change\na\nBer\nook Sy-RW isn't going anywhere",
        "THE\nALGEBRAIC\nEIGENVALUE\nPROBLEM\nDOM\nNVS TIO\nMINA\nMonographs\non Numerical Analysis\nJ.. H. WILKINSON",
    ]
    obj = tt.PostprocessText(mydict=TESTDICT)
    # make sure test works on windows where end-of-line character is \r\n
    test_dict = obj.list_text_english
    for i in test_dict:
        i.replace("\r", "") if i else None
    print("******")
    print(TESTDICT)
    print("******")
    print(reference_dict)
    print("******")
    assert test_dict == reference_dict
    for key in TESTDICT.keys():
        TESTDICT[key].pop("text_english")
    with pytest.raises(ValueError):
        tt.PostprocessText(mydict=TESTDICT)
    obj = tt.PostprocessText(use_csv=True, csv_path="./test/data/test_data_out.csv")
    # make sure test works on windows where end-of-line character is \r\n
    test_df = obj.list_text_english
    for i in test_df:
        i.replace("\r", "") if i else None
    assert test_df == reference_df
    with pytest.raises(ValueError):
        tt.PostprocessText(use_csv=True, csv_path="./test/data/test_data_out_nokey.csv")
    with pytest.raises(ValueError):
        tt.PostprocessText()


def test_analyse_topic():
    _, topic_df, most_frequent_topics = tt.PostprocessText(
        use_csv=True, csv_path="./test/data/topic_analysis_test.csv"
    ).analyse_topic()
    # since this is not deterministic we cannot be sure we get the same result twice
    assert len(topic_df) == 2
    assert topic_df["Name"].iloc[0] == "0_the_feat_of_is"
    assert most_frequent_topics[0][0][0] == "the"
