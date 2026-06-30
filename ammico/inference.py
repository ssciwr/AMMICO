"""External inference backends for AMMICO.

Vision-language summarisation/VQA and audio transcription are no longer run
in-process. Instead they are served by an externally hosted model reached over
an OpenAI-compatible HTTP API. The same code path works against:

* a self-hosted server (e.g. vLLM serving ``Qwen/Qwen2.5-VL-7B-Instruct``),
* the OpenAI API (``https://api.openai.com/v1``),
* Google Gemini via its OpenAI-compatibility endpoint
  (``https://generativelanguage.googleapis.com/v1beta/openai/``).

Only configuration (base url, api key, model id) changes between them.

The ``openai`` SDK is an optional dependency; install it with
``pip install ammico[api]``.
"""

import base64
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence, Union

from PIL import Image


# ---------------------------------------------------------------------------
# Optional dependency handling
# ---------------------------------------------------------------------------
def _require_openai():
    """Import and return the openai module, with an actionable error message."""
    try:
        import openai  # noqa: F401

        return openai
    except ImportError as e:
        raise ImportError(
            "The 'openai' package is required for external inference but is not "
            "installed. Install the API extra with:\n\n"
            "    pip install ammico[api]\n"
        ) from e


def _maybe_load_dotenv() -> None:
    """Load a local .env file if python-dotenv is available. No-op otherwise."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def _first_env(*names: str) -> Optional[str]:
    """Return the first set, non-empty environment variable among names."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _warn_if_external(base_url: str) -> None:
    """Warn when sending data to a non-local endpoint (privacy disclosure).

    Suppress by setting the environment variable AMMICO_PRIVACY_ACK=True.
    """
    if os.environ.get("AMMICO_PRIVACY_ACK") == "True":
        return
    host = base_url.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0].lower()
    if host in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
        return
    warnings.warn(
        "AMMICO is sending image/audio data to an external inference endpoint "
        f"('{host}'). The data leaves your machine and is processed by a third party. "
        "Set AMMICO_PRIVACY_ACK=True to silence this warning, or use a self-hosted "
        "endpoint (e.g. a local vLLM server).",
        RuntimeWarning,
        stacklevel=3,
    )


def encode_image_to_data_url(image: Image.Image, image_format: str = "JPEG") -> str:
    """Encode a PIL image as a base64 ``data:`` URL for an OpenAI image block.

    Args:
        image: PIL image.
        image_format: Encoding format ("JPEG" is compact and lossy, "PNG" lossless).

    Returns:
        A ``data:image/...;base64,...`` URL string.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    buffer = BytesIO()
    save_kwargs = {"quality": 90} if image_format.upper() == "JPEG" else {}
    image.save(buffer, format=image_format, **save_kwargs)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = "jpeg" if image_format.upper() == "JPEG" else image_format.lower()
    return f"data:image/{mime};base64,{encoded}"


class InferenceModel:
    """OpenAI-compatible chat client for vision-language summary and VQA.

    The object exposes a small, provider-agnostic surface used by the detector
    classes: :meth:`build_messages`, :meth:`chat` and :meth:`chat_batch`.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_id: Optional[str] = None,
        timeout: float = 120.0,
        max_concurrency: int = 8,
        image_format: str = "JPEG",
    ) -> None:
        """
        Args:
            base_url: OpenAI-compatible endpoint. Falls back to ``AMMICO_API_BASE_URL``.
            api_key: Bearer token. Falls back to ``AMMICO_API_KEY`` / ``OPENAI_API_KEY``.
            model_id: Served model name. Falls back to ``AMMICO_MODEL_ID``.
            timeout: Per-request timeout in seconds.
            max_concurrency: Maximum number of concurrent requests in :meth:`chat_batch`.
            image_format: Image encoding for inline blocks ("JPEG" or "PNG").
        """
        _maybe_load_dotenv()
        openai = _require_openai()

        self.base_url = base_url or _first_env("AMMICO_API_BASE_URL")
        self.api_key = api_key or _first_env("AMMICO_API_KEY", "OPENAI_API_KEY")
        self.model_id = model_id or _first_env("AMMICO_MODEL_ID")
        self.timeout = timeout
        self.max_concurrency = max(1, int(max_concurrency))
        self.image_format = image_format

        if not self.base_url:
            raise ValueError(
                "No inference endpoint configured. Set AMMICO_API_BASE_URL "
                "(e.g. http://localhost:8000/v1) or pass base_url=..."
            )
        if not self.api_key:
            raise ValueError(
                "No API key configured. Set AMMICO_API_KEY or pass api_key=... "
                "(for a self-hosted vLLM server this is the value of --api-key)."
            )
        if not self.model_id:
            raise ValueError(
                "No model id configured. Set AMMICO_MODEL_ID "
                "(e.g. Qwen/Qwen2.5-VL-7B-Instruct, gpt-4o, gemini-2.0-flash) "
                "or pass model_id=..."
            )

        _warn_if_external(self.base_url)
        self.client = openai.OpenAI(
            base_url=self.base_url, api_key=self.api_key, timeout=timeout
        )

    def build_messages(
        self,
        images: Union[Image.Image, Sequence[Image.Image], None],
        text: str,
    ) -> List[Dict[str, Any]]:
        """Build a single-turn OpenAI chat message from images and a text prompt.

        Args:
            images: A PIL image, a sequence of PIL images, or None for text-only.
            text: The instruction / question text.

        Returns:
            A messages list with one user message; images precede the text block.
        """
        if images is None:
            image_list: List[Image.Image] = []
        elif isinstance(images, Image.Image):
            image_list = [images]
        else:
            image_list = list(images)

        content: List[Dict[str, Any]] = []
        for img in image_list:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": encode_image_to_data_url(img, self.image_format)
                    },
                }
            )
        content.append({"type": "text", "text": text})
        return [{"role": "user", "content": content}]

    def chat(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = 256,
        n: int = 1,
    ) -> List[str]:
        """Run one chat completion and return the text of each choice.

        ``temperature=0`` reproduces the previous greedy (``do_sample=False``)
        decoding behaviour.

        Args:
            messages: OpenAI-format messages (see :meth:`build_messages`).
            max_new_tokens: Upper bound on generated tokens.
            n: Number of completions to return.

        Returns:
            List of completion strings (length ``n``).
        """
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=max_new_tokens,
            n=n,
            temperature=0,
        )
        return [(choice.message.content or "").strip() for choice in response.choices]

    def chat_batch(
        self,
        messages_batch: Sequence[List[Dict[str, Any]]],
        max_new_tokens: int = 256,
    ) -> List[str]:
        """Run many independent chat completions concurrently (one choice each).

        API calls are I/O-bound, so requests are fanned out over a bounded thread
        pool. This replaces the in-GPU batching of the previous local backend.

        Args:
            messages_batch: A sequence of messages lists, one per request.
            max_new_tokens: Upper bound on generated tokens per request.

        Returns:
            List of completion strings, in the same order as ``messages_batch``.
        """
        if not messages_batch:
            return []
        if len(messages_batch) == 1:
            return [self.chat(messages_batch[0], max_new_tokens=max_new_tokens)[0]]

        workers = min(self.max_concurrency, len(messages_batch))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(
                ex.map(
                    lambda m: self.chat(m, max_new_tokens=max_new_tokens)[0],
                    messages_batch,
                )
            )
        return results

    def close(self) -> None:
        """Close the underlying HTTP client (preserves the .close() contract)."""
        try:
            close = getattr(self.client, "close", None)
            if callable(close):
                close()
        except Exception as e:  # pragma: no cover - best effort cleanup
            warnings.warn(f"Failed to close inference client: {e!r}", RuntimeWarning)


class AudioTranscriptionModel:
    """OpenAI-compatible audio transcription client (replaces local WhisperX).

    Points at any ``/v1/audio/transcriptions`` endpoint: a self-hosted Whisper
    server (e.g. Speaches / faster-whisper-server), vLLM audio support, or the
    OpenAI API. Configured independently from :class:`InferenceModel` since the
    audio and VL models usually live on different hosts.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_id: Optional[str] = None,
        language: Optional[str] = None,
        timeout: float = 300.0,
    ) -> None:
        """
        Args:
            base_url: Endpoint. Falls back to ``AMMICO_AUDIO_BASE_URL`` then
                ``AMMICO_API_BASE_URL``.
            api_key: Bearer token. Falls back to ``AMMICO_AUDIO_API_KEY`` then
                ``AMMICO_API_KEY`` / ``OPENAI_API_KEY``.
            model_id: Whisper model name. Falls back to ``AMMICO_AUDIO_MODEL_ID``.
            language: Optional ISO-639-1 code. If None, the server auto-detects.
            timeout: Per-request timeout in seconds.
        """
        _maybe_load_dotenv()
        openai = _require_openai()

        self.base_url = base_url or _first_env(
            "AMMICO_AUDIO_BASE_URL", "AMMICO_API_BASE_URL"
        )
        self.api_key = api_key or _first_env(
            "AMMICO_AUDIO_API_KEY", "AMMICO_API_KEY", "OPENAI_API_KEY"
        )
        self.model_id = model_id or _first_env("AMMICO_AUDIO_MODEL_ID")
        self.language = self._validate_language(language)
        self.timeout = timeout

        if not self.base_url:
            raise ValueError(
                "No audio endpoint configured. Set AMMICO_AUDIO_BASE_URL "
                "(or AMMICO_API_BASE_URL) or pass base_url=..."
            )
        if not self.api_key:
            raise ValueError(
                "No audio API key configured. Set AMMICO_AUDIO_API_KEY "
                "(or AMMICO_API_KEY) or pass api_key=..."
            )
        if not self.model_id:
            raise ValueError(
                "No audio model id configured. Set AMMICO_AUDIO_MODEL_ID "
                "(e.g. whisper-1, Systran/faster-whisper-large-v3) or pass model_id=..."
            )

        _warn_if_external(self.base_url)
        self.client = openai.OpenAI(
            base_url=self.base_url, api_key=self.api_key, timeout=timeout
        )

    @staticmethod
    def _validate_language(language: Optional[str]) -> Optional[str]:
        """Lightly validate an ISO-639-1 code (no whisperx dependency)."""
        if not language:
            return None
        language = language.strip().lower()
        if len(language) != 2 or not language.isalpha():
            raise ValueError(
                f"Invalid language code: '{language}'. Use a 2-letter ISO-639-1 code."
            )
        return language

    def transcribe(
        self, audio_path: str, language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Transcribe an audio file into timestamped segments.

        Returns the same contract the video pipeline consumed from WhisperX:
        a list of ``{start_time, end_time, text, duration}`` dicts.

        Args:
            audio_path: Path to an audio file (16 kHz mono wav from ffmpeg).
            language: Optional override of the instance language.

        Returns:
            List of transcribed segments. Empty if the audio has no speech.
        """
        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file {audio_path} does not exist.")

        lang = self._validate_language(language) if language else self.language

        request_kwargs: Dict[str, Any] = {
            "model": self.model_id,
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment"],
        }
        if lang:
            request_kwargs["language"] = lang

        try:
            with open(audio_path, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    file=audio_file, **request_kwargs
                )
        except Exception as e:
            raise RuntimeError(f"Failed to transcribe audio: {e}") from e

        return self._map_segments(response)

    @staticmethod
    def _map_segments(response: Any) -> List[Dict[str, Any]]:
        """Map a verbose_json transcription response to the segment contract."""
        segments = getattr(response, "segments", None)
        if segments is None and isinstance(response, dict):
            segments = response.get("segments")
        if not segments:
            return []

        def _field(seg: Any, name: str) -> Any:
            if isinstance(seg, dict):
                return seg.get(name)
            return getattr(seg, name, None)

        descriptions: List[Dict[str, Any]] = []
        for seg in segments:
            start = _field(seg, "start")
            end = _field(seg, "end")
            if start is None or end is None:
                warnings.warn(
                    "Skipping transcription segment without start/end timestamps.",
                    RuntimeWarning,
                )
                continue
            start = float(start)
            end = float(end)
            text = _field(seg, "text")
            descriptions.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "text": (text or "").strip(),
                    "duration": end - start,
                }
            )
        return descriptions

    def close(self) -> None:
        """Close the underlying HTTP client (preserves the .close() contract)."""
        try:
            close = getattr(self.client, "close", None)
            if callable(close):
                close()
        except Exception as e:  # pragma: no cover - best effort cleanup
            warnings.warn(
                f"Failed to close transcription client: {e!r}", RuntimeWarning
            )
