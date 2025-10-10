import pytest
import torch
from ammico.model import MultimodalSummaryModel


def test_model_init(model):
    assert model.model is not None
    assert model.processor is not None
    assert model.tokenizer is not None
    assert model.device is not None


def test_model_invalid_device():
    with pytest.raises(ValueError):
        MultimodalSummaryModel(device="invalid_device")


def test_model_invalid_model_id():
    with pytest.raises(ValueError):
        MultimodalSummaryModel(model_id="non_existent_model", device="cpu")


def test_free_resources():
    model = MultimodalSummaryModel(device="cpu")
    model.close()
    assert model.model is None
    assert model.processor is None
