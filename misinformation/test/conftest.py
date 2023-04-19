import os
import pytest


@pytest.fixture
def get_path(request):
    mypath = os.path.dirname(request.module.__file__)
    mypath = mypath + "/data/"
    return mypath


@pytest.fixture
def set_environ(request):
    mypath = os.path.dirname(request.module.__file__)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
        mypath + "/../../data/seismic-bonfire-329406-412821a70264.json"
    )
    print(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))


@pytest.fixture
def get_testdict(get_path):
    testdict = {
        "IMG_2746": {"filename": get_path + "IMG_2746.png"},
        "IMG_2809": {"filename": get_path + "IMG_2809.png"},
    }
    return testdict


@pytest.fixture
def get_test_my_dict(get_path):
    test_my_dict = {
        "IMG_2746": {
            "filename": get_path + "IMG_2746.png",
            "rank A bus": 1,
            "A bus": 0.15640679001808167,
            "rank " + get_path + "IMG_3758.png": 1,
            get_path + "IMG_3758.png": 0.7533495426177979,
        },
        "IMG_2809": {
            "filename": get_path + "IMG_2809.png",
            "rank A bus": 0,
            "A bus": 0.1970970332622528,
            "rank " + get_path + "IMG_3758.png": 0,
            get_path + "IMG_3758.png": 0.8907483816146851,
        },
    }
    return test_my_dict
