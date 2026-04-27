import pytest
from ammico.model import (
    MultimodalSummaryModel,
    AudioToTextModel,
)


@pytest.mark.long
def test_model_init_and_close_releases_resources(model):
    assert model.model is not None
    assert model.processor is not None
    assert model.tokenizer is not None
    assert model.device is not None
    model.close()
    assert model.model is None
    assert model.processor is None
    assert model.tokenizer is None


@pytest.mark.long
def test_model_invalid_device():
    with pytest.raises(ValueError):
        MultimodalSummaryModel(device="invalid_device")


@pytest.mark.long
def test_model_invalid_model_id():
    with pytest.raises(ValueError):
        MultimodalSummaryModel(model_id="non_existent_model", device="cpu")


def test_audio_to_text_model_invalid_language():
    with pytest.raises(ValueError):
        AudioToTextModel(language="xyz")

    with pytest.raises(ValueError):
        AudioToTextModel(language="e4")

    with pytest.raises(ValueError):
        AudioToTextModel(language="ha")  # not supported by whisperx align model
