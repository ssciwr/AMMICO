import pytest
import ammico.text as tt
import spacy
import json
import sys


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


@pytest.fixture
def accepted(monkeypatch):
    monkeypatch.setenv("OTHER_VAR", "True")
    tt.TextDetector({}, accept_privacy="OTHER_VAR")
    return "OTHER_VAR"


def test_privacy_statement(monkeypatch):
    # test pre-set variables: privacy
    monkeypatch.delattr("builtins.input", raising=False)
    monkeypatch.setenv("OTHER_VAR", "something")
    with pytest.raises(ValueError):
        tt.TextDetector({}, accept_privacy="OTHER_VAR")
    monkeypatch.setenv("OTHER_VAR", "False")
    with pytest.raises(ValueError):
        tt.TextDetector({}, accept_privacy="OTHER_VAR")
    with pytest.raises(ValueError):
        tt.TextDetector({}, accept_privacy="OTHER_VAR").get_text_from_image()
    with pytest.raises(ValueError):
        tt.TextDetector({}, accept_privacy="OTHER_VAR").translate_text()
    monkeypatch.setenv("OTHER_VAR", "True")
    pd = tt.TextDetector({}, accept_privacy="OTHER_VAR")
    assert pd.accepted


def test_TextDetector(set_testdict, accepted):
    for item in set_testdict:
        test_obj = tt.TextDetector(set_testdict[item], accept_privacy=accepted)
        assert not test_obj.analyse_text
        assert not test_obj.skip_extraction
        assert test_obj.subdict["filename"] == set_testdict[item]["filename"]
        assert test_obj.model_summary == "sshleifer/distilbart-cnn-12-6"
        assert (
            test_obj.model_sentiment
            == "distilbert-base-uncased-finetuned-sst-2-english"
        )
        assert test_obj.model_ner == "dbmdz/bert-large-cased-finetuned-conll03-english"
        assert test_obj.revision_summary == "a4f8f3e"
        assert test_obj.revision_sentiment == "af0f99b"
        assert test_obj.revision_ner == "f2482bf"
    test_obj = tt.TextDetector(
        {}, analyse_text=True, skip_extraction=True, accept_privacy=accepted
    )
    assert test_obj.analyse_text
    assert test_obj.skip_extraction
    with pytest.raises(ValueError):
        tt.TextDetector({}, analyse_text=1.0, accept_privacy=accepted)
    with pytest.raises(ValueError):
        tt.TextDetector({}, skip_extraction=1.0, accept_privacy=accepted)


def test_run_spacy(set_testdict, get_path, accepted):
    test_obj = tt.TextDetector(
        set_testdict["IMG_3755"], analyse_text=True, accept_privacy=accepted
    )
    ref_file = get_path + "text_IMG_3755.txt"
    with open(ref_file, "r") as file:
        reference_text = file.read()
    test_obj.subdict["text_english"] = reference_text
    test_obj._run_spacy()
    assert isinstance(test_obj.doc, spacy.tokens.doc.Doc)


def test_clean_text(set_testdict, accepted):
    nlp = spacy.load("en_core_web_md")
    doc = nlp("I like cats and fjejg")
    test_obj = tt.TextDetector(set_testdict["IMG_3755"], accept_privacy=accepted)
    test_obj.doc = doc
    test_obj.clean_text()
    result = "I like cats and"
    assert test_obj.subdict["text_clean"] == result


def test_init_revision_numbers_and_models(accepted):
    test_obj = tt.TextDetector({}, accept_privacy=accepted)
    # check the default options
    assert test_obj.model_summary == "sshleifer/distilbart-cnn-12-6"
    assert test_obj.model_sentiment == "distilbert-base-uncased-finetuned-sst-2-english"
    assert test_obj.model_ner == "dbmdz/bert-large-cased-finetuned-conll03-english"
    assert test_obj.revision_summary == "a4f8f3e"
    assert test_obj.revision_sentiment == "af0f99b"
    assert test_obj.revision_ner == "f2482bf"
    # provide non-default options
    model_names = ["facebook/bart-large-cnn", None, None]
    test_obj = tt.TextDetector({}, model_names=model_names, accept_privacy=accepted)
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
        accept_privacy=accepted,
    )
    assert test_obj.model_summary == "facebook/bart-large-cnn"
    assert test_obj.model_sentiment == "distilbert-base-uncased-finetuned-sst-2-english"
    assert test_obj.model_ner == "dbmdz/bert-large-cased-finetuned-conll03-english"
    assert test_obj.revision_summary == "3d22493"
    assert test_obj.revision_sentiment == "af0f99b"
    assert test_obj.revision_ner == "f2482bf"
    # now test the exceptions
    with pytest.raises(ValueError):
        tt.TextDetector({}, analyse_text=1.0, accept_privacy=accepted)
    with pytest.raises(ValueError):
        tt.TextDetector({}, model_names=1.0, accept_privacy=accepted)
    with pytest.raises(ValueError):
        tt.TextDetector({}, revision_numbers=1.0, accept_privacy=accepted)
    with pytest.raises(ValueError):
        tt.TextDetector({}, model_names=["something"], accept_privacy=accepted)
    with pytest.raises(ValueError):
        tt.TextDetector({}, revision_numbers=["something"], accept_privacy=accepted)


def test_check_add_space_after_full_stop(accepted):
    test_obj = tt.TextDetector({}, accept_privacy=accepted)
    test_obj.subdict["text"] = "I like cats. I like dogs."
    test_obj._check_add_space_after_full_stop()
    assert test_obj.subdict["text"] == "I like cats. I like dogs."
    test_obj.subdict["text"] = "I like cats."
    test_obj._check_add_space_after_full_stop()
    assert test_obj.subdict["text"] == "I like cats."
    test_obj.subdict["text"] = "www.icanhascheezburger.com"
    test_obj._check_add_space_after_full_stop()
    assert test_obj.subdict["text"] == "www. icanhascheezburger. com"


@pytest.mark.gcv
def test_analyse_image(set_testdict, set_environ, accepted):
    for item in set_testdict:
        test_obj = tt.TextDetector(set_testdict[item], accept_privacy=accepted)
        test_obj.analyse_image()
        test_obj = tt.TextDetector(
            set_testdict[item], analyse_text=True, accept_privacy=accepted
        )
        test_obj.analyse_image()


@pytest.mark.gcv
def test_get_text_from_image(set_testdict, get_path, set_environ, accepted):
    for item in set_testdict:
        test_obj = tt.TextDetector(set_testdict[item], accept_privacy=accepted)
        test_obj.get_text_from_image()
        ref_file = get_path + "text_" + item + ".txt"
        with open(ref_file, "r", encoding="utf8") as file:
            reference_text = file.read().replace("\n", " ")
        assert test_obj.subdict["text"].replace("\n", " ") == reference_text


def test_translate_text(set_testdict, get_path, accepted):
    for item, lang in zip(set_testdict, LANGUAGES):
        test_obj = tt.TextDetector(set_testdict[item], accept_privacy=accepted)
        ref_file = get_path + "text_" + item + ".txt"
        trans_file = get_path + "text_translated_" + item + ".txt"
        with open(ref_file, "r", encoding="utf8") as file:
            reference_text = file.read().replace("\n", " ")
        with open(trans_file, "r", encoding="utf8") as file:
            true_translated_text = file.read().replace("\n", " ")
        test_obj.subdict["text"] = reference_text
        test_obj.translate_text()
        assert test_obj.subdict["text_language"] == lang
        translated_text = test_obj.subdict["text_english"].lower().replace("\n", " ")
        for word in true_translated_text.lower():
            assert word in translated_text


def test_remove_linebreaks(accepted):
    test_obj = tt.TextDetector({}, accept_privacy=accepted)
    test_obj.subdict["text"] = "This is \n a test."
    test_obj.subdict["text_english"] = "This is \n another\n test."
    test_obj.remove_linebreaks()
    assert test_obj.subdict["text"] == "This is   a test."
    assert test_obj.subdict["text_english"] == "This is   another  test."


def test_text_summary(get_path, accepted):
    mydict = {}
    test_obj = tt.TextDetector(mydict, analyse_text=True, accept_privacy=accepted)
    ref_file = get_path + "example_summary.txt"
    with open(ref_file, "r", encoding="utf8") as file:
        reference_text = file.read()
    mydict["text_english"] = reference_text
    test_obj.text_summary()
    reference_summary = " I’m sorry, but I don’t want to be an emperor"
    assert mydict["text_summary"] == reference_summary


def test_text_sentiment_transformers(accepted):
    mydict = {}
    test_obj = tt.TextDetector(mydict, analyse_text=True, accept_privacy=accepted)
    mydict["text_english"] = "I am happy that the CI is working again."
    test_obj.text_sentiment_transformers()
    assert mydict["sentiment"] == "POSITIVE"
    assert mydict["sentiment_score"] == pytest.approx(0.99, 0.02)


def test_text_ner(accepted):
    mydict = {}
    test_obj = tt.TextDetector(mydict, analyse_text=True, accept_privacy=accepted)
    mydict["text_english"] = "Bill Gates was born in Seattle."
    test_obj.text_ner()
    assert mydict["entity"] == ["Bill Gates", "Seattle"]
    assert mydict["entity_type"] == ["PER", "LOC"]


def test_init_csv_option(get_path):
    test_obj = tt.TextAnalyzer(csv_path=get_path + "test.csv")
    assert test_obj.csv_path == get_path + "test.csv"
    assert test_obj.column_key == "text"
    assert test_obj.csv_encoding == "utf-8"
    test_obj = tt.TextAnalyzer(
        csv_path=get_path + "test.csv", column_key="mytext", csv_encoding="utf-16"
    )
    assert test_obj.column_key == "mytext"
    assert test_obj.csv_encoding == "utf-16"
    with pytest.raises(ValueError):
        tt.TextAnalyzer(csv_path=1.0)
    with pytest.raises(ValueError):
        tt.TextAnalyzer(csv_path="something")
    with pytest.raises(FileNotFoundError):
        tt.TextAnalyzer(csv_path=get_path + "test_no.csv")
    with pytest.raises(ValueError):
        tt.TextAnalyzer(csv_path=get_path + "test.csv", column_key=1.0)
    with pytest.raises(ValueError):
        tt.TextAnalyzer(csv_path=get_path + "test.csv", csv_encoding=1.0)


@pytest.mark.skipif(sys.platform == "win32", reason="Encoding different on Window")
def test_read_csv(get_path):
    test_obj = tt.TextAnalyzer(csv_path=get_path + "test.csv")
    test_obj.read_csv()
    with open(get_path + "test_read_csv_ref.json", "r") as file:
        ref_dict = json.load(file)
    # we are assuming the order did not get jungled up
    for (_, value_test), (_, value_ref) in zip(
        test_obj.mydict.items(), ref_dict.items()
    ):
        assert value_test["text"] == value_ref["text"]
    # test with different encoding
    test_obj = tt.TextAnalyzer(
        csv_path=get_path + "test-utf16.csv", csv_encoding="utf-16"
    )
    test_obj.read_csv()
    # we are assuming the order did not get jungled up
    for (_, value_test), (_, value_ref) in zip(
        test_obj.mydict.items(), ref_dict.items()
    ):
        assert value_test["text"] == value_ref["text"]


def test_PostprocessText(set_testdict, get_path):
    reference_dict = "THE ALGEBRAIC EIGENVALUE PROBLEM"
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
