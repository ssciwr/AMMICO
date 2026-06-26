import gc
import os
import pytest
from ammico.model import MultimodalEmbeddingsModel
from ammico.inference import InferenceModel
from ammico.video_summary import VideoSummaryDetector
from ammico.multimodal_search import MultimodalSearch
from unittest.mock import MagicMock
import numpy as np
import torch


def _api_env_configured() -> bool:
    """True if the live inference endpoint is configured via environment."""
    return all(
        os.environ.get(var)
        for var in ("AMMICO_API_BASE_URL", "AMMICO_API_KEY", "AMMICO_MODEL_ID")
    )


def _release_torch_memory() -> None:
    gc.collect(2)
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


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


@pytest.fixture(scope="module")
def model():
    # Live inference endpoint (vLLM/OpenAI/Gemini), used by @pytest.mark.long tests.
    # Skipped unless AMMICO_API_* environment variables are configured.
    if not _api_env_configured():
        pytest.skip("AMMICO_API_* env vars not set; skipping live inference tests")
    m = InferenceModel()
    try:
        yield m
    finally:
        m.close()


class MockInferenceModel:
    """Mock InferenceModel that returns canned text without any network call."""

    def __init__(self, chat_return="mock caption"):
        self.model_id = "mock-model"
        self.chat_return = chat_return

    def build_messages(self, images, text):
        content = []
        if images is not None:
            image_list = images if isinstance(images, (list, tuple)) else [images]
            content.extend({"type": "image_url"} for _ in image_list)
        content.append({"type": "text", "text": text})
        return [{"role": "user", "content": content}]

    def chat(self, messages, max_new_tokens=256, n=1):
        return [self.chat_return for _ in range(n)]

    def chat_batch(self, messages_batch, max_new_tokens=256):
        return [self.chat_return for _ in messages_batch]

    def close(self):
        pass


@pytest.fixture
def mock_model():
    """Mock inference model for fast unit tests (no network, no weights)."""
    return MockInferenceModel()


class MockAudioTranscriptionModel:
    """Mock audio transcription model returning a fixed segment."""

    def __init__(self):
        self.closed = False

    def transcribe(self, audio_path, language=None):
        return [{"start_time": 0.0, "end_time": 1.0, "text": "hello", "duration": 1.0}]

    def close(self):
        self.closed = True


@pytest.fixture
def mock_audio_model():
    return MockAudioTranscriptionModel()


@pytest.fixture(scope="module")
def video_summary_model(model):
    vsm = VideoSummaryDetector(summary_model=model)
    try:
        yield vsm
    finally:
        vsm.summary_model.close()


@pytest.fixture(scope="module")
def multimodal_embeddings_model_cpu():
    mem = MultimodalEmbeddingsModel(device="cpu")
    try:
        yield mem
    finally:
        mem.close()
        _release_torch_memory()


@pytest.fixture(scope="module")
def multimodal_embeddings_model_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available for multimodal embeddings tests")
    mem = MultimodalEmbeddingsModel(device="cuda")
    try:
        yield mem
    finally:
        mem.close()
        _release_torch_memory()


@pytest.fixture
def mock_multimodal_cpu_model():
    model = MagicMock(spec=MultimodalEmbeddingsModel)
    model.device = "cpu"
    model.encode_image.return_value = np.random.rand(10, 128)
    model.encode_text.return_value = np.random.rand(1, 128)
    return model


@pytest.fixture
def multimodal_search_mock(mock_multimodal_cpu_model):
    return MultimodalSearch(model=mock_multimodal_cpu_model)
