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
