import os
import pytest
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
