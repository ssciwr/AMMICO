"""Integration tests against live external endpoints.

These are opt-in: marked ``long`` and skipped unless the relevant
``AMMICO_API_*`` / ``AMMICO_AUDIO_*`` environment variables are configured.
Run a real OpenAI-compatible server (e.g. vLLM for the VL model, a Whisper
server for audio) and point the variables at it, then::

    pytest -m long ammico/test/test_integration_inference.py
"""

import os
import pytest

from ammico.inference import InferenceModel, AudioTranscriptionModel
from ammico.image_summary import ImageSummaryDetector


def _vl_configured() -> bool:
    return all(
        os.environ.get(v)
        for v in ("AMMICO_API_BASE_URL", "AMMICO_API_KEY", "AMMICO_MODEL_ID")
    )


def _audio_configured() -> bool:
    return all(
        os.environ.get(v) or os.environ.get(fallback)
        for v, fallback in (
            ("AMMICO_AUDIO_BASE_URL", "AMMICO_API_BASE_URL"),
            ("AMMICO_AUDIO_API_KEY", "AMMICO_API_KEY"),
            ("AMMICO_AUDIO_MODEL_ID", "AMMICO_AUDIO_MODEL_ID"),
        )
    )


@pytest.mark.long
@pytest.mark.skipif(not _vl_configured(), reason="AMMICO_API_* not configured")
def test_live_image_summary_and_vqa(get_testdict):
    model = InferenceModel()
    try:
        detector = ImageSummaryDetector(summary_model=model, subdict=get_testdict)
        results = detector.analyse_images_from_dict(
            analysis_type="summary_and_questions",
            list_of_questions=["What is in the image?"],
        )
        for key in get_testdict:
            assert isinstance(results[key]["caption"], str)
            assert len(results[key]["caption"]) > 0
            assert isinstance(results[key]["vqa"], list)
            assert len(results[key]["vqa"]) == 1
    finally:
        model.close()


@pytest.mark.long
@pytest.mark.skipif(not _audio_configured(), reason="AMMICO_AUDIO_* not configured")
def test_live_audio_transcription(get_path):
    # Extract a wav from the test video and transcribe it over the live endpoint.
    import subprocess
    import tempfile

    video = os.path.join(get_path, "video1.mp4")
    model = AudioTranscriptionModel()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            wav = os.path.join(tmp, "audio.wav")
            subprocess.run(
                ["ffmpeg", "-i", video, "-vn", "-ar", "16000", "-ac", "1", "-y", wav],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            segments = model.transcribe(wav)
        assert isinstance(segments, list)
        for seg in segments:
            assert set(seg) == {"start_time", "end_time", "text", "duration"}
            assert seg["end_time"] >= seg["start_time"]
    finally:
        model.close()
