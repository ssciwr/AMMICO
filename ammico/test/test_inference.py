"""Unit tests for the external inference backends (no network calls)."""

import types
import pytest
from PIL import Image

from ammico.inference import (
    InferenceModel,
    AudioTranscriptionModel,
    encode_image_to_data_url,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clear_env(monkeypatch):
    for var in (
        "AMMICO_API_BASE_URL",
        "AMMICO_API_KEY",
        "AMMICO_MODEL_ID",
        "OPENAI_API_KEY",
        "AMMICO_AUDIO_BASE_URL",
        "AMMICO_AUDIO_API_KEY",
        "AMMICO_AUDIO_MODEL_ID",
    ):
        monkeypatch.delenv(var, raising=False)
    # Avoid picking up a developer .env during tests.
    monkeypatch.setattr("ammico.inference._maybe_load_dotenv", lambda: None)


def _fake_chat_client(captured):
    """Return a fake OpenAI-like client capturing the create() kwargs."""

    def create(**kwargs):
        captured.append(kwargs)
        n = kwargs.get("n", 1)
        choices = [
            types.SimpleNamespace(message=types.SimpleNamespace(content=f"answer {i}"))
            for i in range(n)
        ]
        return types.SimpleNamespace(choices=choices)

    completions = types.SimpleNamespace(create=create)
    chat = types.SimpleNamespace(completions=completions)
    return types.SimpleNamespace(chat=chat)


def _make_model(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setattr(
        "openai.OpenAI", lambda **kwargs: types.SimpleNamespace(**kwargs)
    )
    return InferenceModel(
        base_url="http://localhost:8000/v1",
        api_key="secret",
        model_id="test-model",
    )


# ---------------------------------------------------------------------------
# Image encoding / message building
# ---------------------------------------------------------------------------
def test_encode_image_to_data_url():
    img = Image.new("RGB", (8, 8), color=(255, 0, 0))
    url = encode_image_to_data_url(img, "JPEG")
    assert url.startswith("data:image/jpeg;base64,")
    assert len(url) > len("data:image/jpeg;base64,")


def test_build_messages_with_image(monkeypatch):
    model = _make_model(monkeypatch)
    img = Image.new("RGB", (8, 8), color=(0, 128, 0))
    messages = model.build_messages(img, "What is this?")
    assert len(messages) == 1
    content = messages[0]["content"]
    # image block(s) precede the text block
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/")
    assert content[-1] == {"type": "text", "text": "What is this?"}


def test_build_messages_text_only(monkeypatch):
    model = _make_model(monkeypatch)
    messages = model.build_messages(None, "summarise")
    content = messages[0]["content"]
    assert content == [{"type": "text", "text": "summarise"}]


def test_build_messages_multiple_images(monkeypatch):
    model = _make_model(monkeypatch)
    imgs = [Image.new("RGB", (4, 4)) for _ in range(3)]
    messages = model.build_messages(imgs, "describe")
    content = messages[0]["content"]
    image_blocks = [c for c in content if c["type"] == "image_url"]
    assert len(image_blocks) == 3


# ---------------------------------------------------------------------------
# chat / chat_batch
# ---------------------------------------------------------------------------
def test_chat_returns_choice_text(monkeypatch):
    model = _make_model(monkeypatch)
    captured = []
    model.client = _fake_chat_client(captured)

    out = model.chat([{"role": "user", "content": "hi"}], max_new_tokens=42, n=1)
    assert out == ["answer 0"]
    assert captured[0]["model"] == "test-model"
    assert captured[0]["max_tokens"] == 42
    assert captured[0]["temperature"] == 0


def test_chat_multiple_choices(monkeypatch):
    model = _make_model(monkeypatch)
    model.client = _fake_chat_client([])
    out = model.chat([{"role": "user", "content": "hi"}], n=3)
    assert out == ["answer 0", "answer 1", "answer 2"]


def test_chat_batch_preserves_order(monkeypatch):
    model = _make_model(monkeypatch)
    calls = []

    def fake_chat(messages, max_new_tokens=256, n=1):
        # echo the prompt text so we can assert ordering
        calls.append(messages)
        return [messages]

    monkeypatch.setattr(model, "chat", fake_chat)
    batch = [[f"q{i}"] for i in range(5)]
    out = model.chat_batch(batch, max_new_tokens=16)
    assert out == batch


def test_chat_batch_empty(monkeypatch):
    model = _make_model(monkeypatch)
    assert model.chat_batch([]) == []


# ---------------------------------------------------------------------------
# Config / error handling
# ---------------------------------------------------------------------------
def test_missing_base_url_raises(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setattr(
        "openai.OpenAI", lambda **kwargs: types.SimpleNamespace(**kwargs)
    )
    with pytest.raises(ValueError, match="endpoint"):
        InferenceModel(api_key="k", model_id="m")


def test_missing_api_key_raises(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setattr(
        "openai.OpenAI", lambda **kwargs: types.SimpleNamespace(**kwargs)
    )
    with pytest.raises(ValueError, match="API key"):
        InferenceModel(base_url="http://x/v1", model_id="m")


def test_missing_model_id_raises(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setattr(
        "openai.OpenAI", lambda **kwargs: types.SimpleNamespace(**kwargs)
    )
    with pytest.raises(ValueError, match="model id"):
        InferenceModel(base_url="http://x/v1", api_key="k")


def test_external_endpoint_warns(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setattr(
        "openai.OpenAI", lambda **kwargs: types.SimpleNamespace(**kwargs)
    )
    with pytest.warns(RuntimeWarning, match="external inference endpoint"):
        InferenceModel(
            base_url="https://api.openai.com/v1", api_key="k", model_id="gpt-4o"
        )


def test_local_endpoint_no_warn(monkeypatch, recwarn):
    _clear_env(monkeypatch)
    monkeypatch.setattr(
        "openai.OpenAI", lambda **kwargs: types.SimpleNamespace(**kwargs)
    )
    InferenceModel(base_url="http://localhost:8000/v1", api_key="k", model_id="m")
    assert not any(
        "external inference endpoint" in str(w.message) for w in recwarn.list
    )


def test_env_fallback(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("AMMICO_PRIVACY_ACK", "True")
    monkeypatch.setenv("AMMICO_API_BASE_URL", "http://env/v1")
    monkeypatch.setenv("AMMICO_API_KEY", "envkey")
    monkeypatch.setenv("AMMICO_MODEL_ID", "envmodel")
    monkeypatch.setattr(
        "openai.OpenAI", lambda **kwargs: types.SimpleNamespace(**kwargs)
    )
    model = InferenceModel()
    assert model.base_url == "http://env/v1"
    assert model.api_key == "envkey"
    assert model.model_id == "envmodel"


# ---------------------------------------------------------------------------
# Audio transcription
# ---------------------------------------------------------------------------
def _make_audio_model(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setattr(
        "openai.OpenAI", lambda **kwargs: types.SimpleNamespace(**kwargs)
    )
    return AudioTranscriptionModel(
        base_url="http://localhost:9000/v1",
        api_key="secret",
        model_id="whisper-1",
    )


def test_audio_invalid_language(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setattr(
        "openai.OpenAI", lambda **kwargs: types.SimpleNamespace(**kwargs)
    )
    for bad in ("xyz", "e4", "1"):
        with pytest.raises(ValueError):
            AudioTranscriptionModel(
                base_url="http://x/v1",
                api_key="k",
                model_id="whisper-1",
                language=bad,
            )


def test_audio_map_segments_object():
    seg = types.SimpleNamespace(start=1.0, end=2.5, text=" hi ")
    resp = types.SimpleNamespace(segments=[seg])
    out = AudioTranscriptionModel._map_segments(resp)
    assert out == [{"start_time": 1.0, "end_time": 2.5, "text": "hi", "duration": 1.5}]


def test_audio_map_segments_dict():
    resp = {"segments": [{"start": 0.0, "end": 3.0, "text": "world"}]}
    out = AudioTranscriptionModel._map_segments(resp)
    assert out[0]["start_time"] == 0.0
    assert out[0]["end_time"] == 3.0
    assert out[0]["duration"] == 3.0
    assert out[0]["text"] == "world"


def test_audio_map_segments_empty():
    assert (
        AudioTranscriptionModel._map_segments(types.SimpleNamespace(segments=None))
        == []
    )


def test_audio_map_segments_skips_missing_timestamps():
    """Segments lacking start/end are skipped with a warning, not a crash.

    Regression test: ``getattr(seg, "start")`` with no default raised
    AttributeError (and ``float(None)`` raised TypeError) on nonconforming
    responses.
    """
    good = types.SimpleNamespace(start=1.0, end=2.0, text="ok")
    missing_attr = types.SimpleNamespace(text="no timestamps")
    missing_key = {"text": "also no timestamps"}
    none_value = {"start": None, "end": 3.0, "text": "null start"}
    resp = types.SimpleNamespace(segments=[good, missing_attr, missing_key, none_value])

    with pytest.warns(RuntimeWarning, match="without start/end timestamps"):
        out = AudioTranscriptionModel._map_segments(resp)

    assert out == [{"start_time": 1.0, "end_time": 2.0, "text": "ok", "duration": 1.0}]


def test_audio_transcribe(monkeypatch, tmp_path):
    model = _make_audio_model(monkeypatch)
    audio_file = tmp_path / "a.wav"
    audio_file.write_bytes(b"RIFFfake")

    captured = {}

    def create(file, **kwargs):
        captured.update(kwargs)
        return {"segments": [{"start": 0.0, "end": 1.2, "text": "spoken"}]}

    model.client = types.SimpleNamespace(
        audio=types.SimpleNamespace(transcriptions=types.SimpleNamespace(create=create))
    )
    out = model.transcribe(str(audio_file))
    assert out == [
        {"start_time": 0.0, "end_time": 1.2, "text": "spoken", "duration": 1.2}
    ]
    assert captured["model"] == "whisper-1"
    assert captured["response_format"] == "verbose_json"


def test_audio_transcribe_missing_file(monkeypatch):
    model = _make_audio_model(monkeypatch)
    with pytest.raises(ValueError, match="does not exist"):
        model.transcribe("/no/such/file.wav")
