import os
import pytest
from ammico.model import MultimodalSummaryModel
from ammico.video_summary import VideoSummaryDetector

import torch


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
def get_video_testdict(get_path):
    testdict = {
        "video1": {"filename": get_path + "video1.mp4"},
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


@pytest.fixture(scope="session")
def model():
    m = MultimodalSummaryModel(device="cpu")
    try:
        yield m
    finally:
        m.close()


@pytest.fixture
def mock_model():
    """
    Mock model fixture that doesn't load the actual model.
    Useful for faster unit tests that don't need actual model inference.
    """

    class MockProcessor:
        """Mock processor that mimics AutoProcessor behavior."""

        def apply_chat_template(self, messages, **kwargs):
            return "processed_text"

        def __call__(self, text, images, **kwargs):
            """Mock processing that returns tensor-like inputs."""
            batch_size = len(text) if isinstance(text, list) else 1
            return {
                "input_ids": torch.randint(0, 1000, (batch_size, 10)),
                "pixel_values": torch.randn(batch_size, 3, 224, 224),
                "attention_mask": torch.ones(batch_size, 10),
            }

    class MockTokenizer:
        """Mock tokenizer that mimics AutoTokenizer behavior."""

        def batch_decode(self, ids, **kwargs):
            """Return mock captions for the given batch size."""
            batch_size = ids.shape[0] if hasattr(ids, "shape") else len(ids)
            return ["mock caption" for _ in range(batch_size)]

    class MockModelObj:
        """Mock model object that mimics the model.generate behavior."""

        def __init__(self):
            self.device = "cpu"

        def eval(self):
            return self

        def generate(self, input_ids=None, **kwargs):
            """Generate mock token IDs."""
            batch_size = input_ids.shape[0] if hasattr(input_ids, "shape") else 1
            return torch.randint(0, 1000, (batch_size, 20))

    class MockMultimodalSummaryModel:
        """Mock MultimodalSummaryModel that doesn't load actual models."""

        def __init__(self):
            self.model = MockModelObj()
            self.processor = MockProcessor()
            self.tokenizer = MockTokenizer()
            self.device = "cpu"

        def close(self):
            """Mock close method - no actual cleanup needed."""
            pass

    return MockMultimodalSummaryModel()


@pytest.fixture(scope="session")
def video_summary_model(model):
    vsm = VideoSummaryDetector(summary_model=model)
    try:
        yield vsm
    finally:
        vsm.summary_model.close()
