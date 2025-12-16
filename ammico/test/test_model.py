import pytest
from ammico.model import MultimodalSummaryModel, AudioToTextModel


@pytest.mark.long
def test_model_init(model):
    assert model.model is not None
    assert model.processor is not None
    assert model.tokenizer is not None
    assert model.device is not None


@pytest.mark.long
def test_model_invalid_device():
    with pytest.raises(ValueError):
        MultimodalSummaryModel(device="invalid_device")


@pytest.mark.long
def test_model_invalid_model_id():
    with pytest.raises(ValueError):
        MultimodalSummaryModel(model_id="non_existent_model", device="cpu")


@pytest.mark.long
def test_free_resources():
    model = MultimodalSummaryModel(device="cpu")
    model.close()
    assert model.model is None
    assert model.processor is None


def test_audio_to_text_model_invalid_language():
    with pytest.raises(ValueError):
        AudioToTextModel(language="xyz")

    with pytest.raises(ValueError):
        AudioToTextModel(language="e4")

    with pytest.raises(ValueError):
        AudioToTextModel(language="ha")  # not supported by whisperx align model
