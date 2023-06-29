import pytest
import ammico.text as tt


@pytest.fixture
def set_testdict(get_path):
    testdict = {
        "IMG_3755": {
            "filename": get_path + "IMG_3755.jpg",
        },
        "IMG_3756": {
            "filename": get_path + "IMG_3756.jpg",
        },
        "IMG_3757": {
            "filename": get_path + "IMG_3757.jpg",
        },
    }
    return testdict


LANGUAGES = ["de", "en", "en"]


def test_TextDetector(set_testdict):
    for item in set_testdict:
        test_obj = tt.TextDetector(set_testdict[item])
        assert test_obj.subdict["text"] is None
        assert test_obj.subdict["text_language"] is None
        assert test_obj.subdict["text_english"] is None
        assert not test_obj.analyse_text


def test_init_revision_numbers_and_models():
    test_obj = tt.TextDetector({})
    # check the default options
    assert test_obj.model_summary == "sshleifer/distilbart-cnn-12-6"
    assert test_obj.model_sentiment == "distilbert-base-uncased-finetuned-sst-2-english"
    assert test_obj.model_ner == "dbmdz/bert-large-cased-finetuned-conll03-english"
    assert test_obj.revision_summary == "a4f8f3e"
    assert test_obj.revision_sentiment == "af0f99b"
    assert test_obj.revision_ner == "f2482bf"
    # provide non-default options
    model_names = ["facebook/bart-large-cnn", None, None]
    test_obj = tt.TextDetector({}, model_names=model_names)
    assert test_obj.model_summary == "facebook/bart-large-cnn"
    assert test_obj.model_sentiment == "distilbert-base-uncased-finetuned-sst-2-english"
    assert test_obj.model_ner == "dbmdz/bert-large-cased-finetuned-conll03-english"
    assert not test_obj.revision_summary
    assert test_obj.revision_sentiment == "af0f99b"
    assert test_obj.revision_ner == "f2482bf"
    revision_numbers = ["3d22493", None, None]
    test_obj = tt.TextDetector(
        {},
        model_names=model_names,
        revision_numbers=revision_numbers,
    )
    assert test_obj.model_summary == "facebook/bart-large-cnn"
    assert test_obj.model_sentiment == "distilbert-base-uncased-finetuned-sst-2-english"
    assert test_obj.model_ner == "dbmdz/bert-large-cased-finetuned-conll03-english"
    assert test_obj.revision_summary == "3d22493"
    assert test_obj.revision_sentiment == "af0f99b"
    assert test_obj.revision_ner == "f2482bf"
    # now test the exceptions
    with pytest.raises(ValueError):
        tt.TextDetector({}, analyse_text=1.0)
    with pytest.raises(ValueError):
        tt.TextDetector({}, model_names=1.0)
    with pytest.raises(ValueError):
        tt.TextDetector({}, revision_numbers=1.0)
    with pytest.raises(ValueError):
        tt.TextDetector({}, model_names=["something"])
    with pytest.raises(ValueError):
        tt.TextDetector({}, revision_numbers=["something"])


@pytest.mark.gcv
def test_analyse_image(set_testdict, set_environ):
    for item in set_testdict:
        test_obj = tt.TextDetector(set_testdict[item])
        test_obj.analyse_image()
        test_obj = tt.TextDetector(set_testdict[item], analyse_text=True)
        test_obj.analyse_image()


@pytest.mark.gcv
def test_get_text_from_image(set_testdict, get_path, set_environ):
    for item in set_testdict:
        test_obj = tt.TextDetector(set_testdict[item])
        test_obj.get_text_from_image()
        ref_file = get_path + "text_" + item + ".txt"
        with open(ref_file, "r", encoding="utf8") as file:
            reference_text = file.read()
        assert test_obj.subdict["text"] == reference_text


def test_translate_text(set_testdict, get_path):
    for item, lang in zip(set_testdict, LANGUAGES):
        test_obj = tt.TextDetector(set_testdict[item])
        ref_file = get_path + "text_" + item + ".txt"
        trans_file = get_path + "text_translated_" + item + ".txt"
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


@pytest.mark.win_skip
def test_text_summary(get_path):
    mydict = {}
    test_obj = tt.TextDetector(mydict, analyse_text=True)
    ref_file = get_path + "example_summary.txt"
    with open(ref_file, "r", encoding="utf8") as file:
        reference_text = file.read()
    mydict["text_english"] = reference_text
    test_obj.text_summary()
    reference_summary = " I’m sorry, but I don’t want to be an emperor"
    assert mydict["text_summary"] == reference_summary


def test_text_sentiment_transformers():
    mydict = {}
    test_obj = tt.TextDetector(mydict, analyse_text=True)
    mydict["text_english"] = "I am happy that the CI is working again."
    test_obj.text_sentiment_transformers()
    assert mydict["sentiment"] == "POSITIVE"
    assert mydict["sentiment_score"] == pytest.approx(0.99, 0.02)


@pytest.mark.win_skip
def test_text_ner():
    mydict = {}
    test_obj = tt.TextDetector(mydict, analyse_text=True)
    mydict["text_english"] = "Bill Gates was born in Seattle."
    test_obj.text_ner()
    assert mydict["entity"] == ["Bill Gates", "Seattle"]
    assert mydict["entity_type"] == ["PER", "LOC"]


@pytest.mark.win_skip
def test_PostprocessText(set_testdict, get_path):
    reference_dict = "THE\nALGEBRAIC\nEIGENVALUE\nPROBLEM\nDOM\nNVS TIO\nMINA\nMonographs\non Numerical Analysis\nJ.. H. WILKINSON"
    reference_df = "Mathematische Formelsammlung\nfür Ingenieure und Naturwissenschaftler\nMit zahlreichen Abbildungen und Rechenbeispielen\nund einer ausführlichen Integraltafel\n3., verbesserte Auflage"
    img_numbers = ["IMG_3755", "IMG_3756", "IMG_3757"]
    for image_ref in img_numbers:
        ref_file = get_path + "text_" + image_ref + ".txt"
        with open(ref_file, "r") as file:
            reference_text = file.read()
        set_testdict[image_ref]["text_english"] = reference_text
    obj = tt.PostprocessText(mydict=set_testdict)
    test_dict = obj.list_text_english[2].replace("\r", "")
    assert test_dict == reference_dict
    for key in set_testdict.keys():
        set_testdict[key].pop("text_english")
    with pytest.raises(ValueError):
        tt.PostprocessText(mydict=set_testdict)
    obj = tt.PostprocessText(use_csv=True, csv_path=get_path + "test_data_out.csv")
    # make sure test works on windows where end-of-line character is \r\n
    test_df = obj.list_text_english[0].replace("\r", "")
    assert test_df == reference_df
    with pytest.raises(ValueError):
        tt.PostprocessText(use_csv=True, csv_path=get_path + "test_data_out_nokey.csv")
    with pytest.raises(ValueError):
        tt.PostprocessText()


def test_analyse_topic(get_path):
    _, topic_df, most_frequent_topics = tt.PostprocessText(
        use_csv=True, csv_path=get_path + "topic_analysis_test.csv"
    ).analyse_topic()
    # since this is not deterministic we cannot be sure we get the same result twice
    assert len(topic_df) == 2
    assert topic_df["Name"].iloc[0] == "0_the_feat_of_is"
    assert most_frequent_topics[0][0][0] == "the"
