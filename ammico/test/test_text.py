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


def test_truncate_text(accepted):
    test_obj = tt.TextDetector({}, accept_privacy=accepted)
    test_obj.subdict["text"] = "I like cats and dogs."
    test_obj._truncate_text()
    assert test_obj.subdict["text"] == "I like cats and dogs."
    assert "text_truncated" not in test_obj.subdict
    test_obj.subdict["text"] = 20000 * "m"
    test_obj._truncate_text()
    assert test_obj.subdict["text_truncated"] == 5000 * "m"
    assert test_obj.subdict["text"] == 20000 * "m"


@pytest.mark.gcv
def test_analyse_image(set_testdict, set_environ, accepted):
    for item in set_testdict:
        test_obj = tt.TextDetector(set_testdict[item], accept_privacy=accepted)
        test_obj.analyse_image()
        test_obj = tt.TextDetector(
            set_testdict[item], analyse_text=True, accept_privacy=accepted
        )
        test_obj.analyse_image()
    testdict = {}
    testdict["text"] = 20000 * "m"
    test_obj = tt.TextDetector(
        testdict, skip_extraction=True, analyse_text=True, accept_privacy=accepted
    )
    test_obj.analyse_image()
    assert test_obj.subdict["text_truncated"] == 5000 * "m"
    assert test_obj.subdict["text"] == 20000 * "m"


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
