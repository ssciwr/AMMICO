from ammico.video_summary import VideoSummaryDetector
from ammico.inference import AudioTranscriptionModel

import pytest
from PIL import Image


def test_analyse_videos_from_dict_invalid_call():
    video_summ = VideoSummaryDetector()

    with pytest.raises(ValueError):
        video_summ.analyse_videos_from_dict(analysis_type="invalid_type")

    with pytest.raises(TypeError):
        video_summ.analyse_videos_from_dict(list_of_questions="not_a_list")

    with pytest.raises(ValueError):
        video_summ.analyse_videos_from_dict(list_of_questions=[None])

    with pytest.raises(ValueError):
        video_summ.analyse_videos_from_dict(list_of_questions=[123, "valid question"])


def test_analyse_videos_incorrect_subdict_type():
    with pytest.raises(TypeError):
        VideoSummaryDetector(subdict="not_a_dict")

    with pytest.raises(ValueError):
        VideoSummaryDetector(subdict={"vid1": {"wrong_key": "value"}})


@pytest.mark.long
def test_analyse_videos_from_dict_summary(video_summary_model, get_video_testdict):
    video_summary_model.subdict = get_video_testdict
    results = video_summary_model.analyse_videos_from_dict(analysis_type="summary")

    assert "video1" in results
    assert "summary" in results["video1"]
    assert isinstance(results["video1"]["summary"], str)


@pytest.mark.long
def test_analyse_videos_from_dict_questions(video_summary_model, get_video_testdict):
    video_summary_model.subdict = get_video_testdict
    questions = ["When and where the video was recorded?"]
    results = video_summary_model.analyse_videos_from_dict(
        analysis_type="questions", list_of_questions=questions
    )

    assert "video1" in results
    assert "vqa_answers" in results["video1"]
    vqa_texts = [vqa.lower() for vqa in results["video1"]["vqa_answers"]]
    assert not any("heidelberg" in text for text in vqa_texts)
    assert not any("november" in text for text in vqa_texts)


@pytest.mark.long
def test_analyse_videos_from_dict_summary_and_questions(
    video_summary_model, get_video_testdict
):
    video_summary_model.subdict = get_video_testdict
    video_summary_model.audio_model = AudioTranscriptionModel()
    questions = ["When and where the video was recorded?"]
    results = video_summary_model.analyse_videos_from_dict(
        analysis_type="summary_and_questions", list_of_questions=questions
    )
    assert "video1" in results
    assert "summary" in results["video1"]
    assert "vqa_answers" in results["video1"]
    vqa_texts = [res.lower() for res in results["video1"]["vqa_answers"]]
    assert any(("heidelberg" in text) or ("urban" in text) for text in vqa_texts)
    assert any(("november" in text) or ("autumn" in text) for text in vqa_texts)


def test_make_captions_for_subclips_invalid_dict(mock_model):
    video_summ = VideoSummaryDetector(summary_model=mock_model)

    with pytest.raises(ValueError):
        video_summ.make_captions_for_subclips(
            entry={"filename": "non_existent_file.mp4"}
        )


def test_analyse_videos_from_dict_continues_on_failure(mock_model):
    """A video whose processing raises does not abort the rest of the batch."""

    class _FailingDetector(VideoSummaryDetector):
        def make_captions_for_subclips(self, entry, list_of_questions=None):
            raise RuntimeError("model 'mock-model' not found")

    detector = _FailingDetector(
        summary_model=mock_model,
        subdict={
            "video1": {"filename": "a.mp4"},
            "video2": {"filename": "b.mp4"},
        },
    )
    with pytest.warns(RuntimeWarning, match="Video analysis failed for .*mock-model"):
        results = detector.analyse_videos_from_dict(
            analysis_type="summary_and_questions",
            list_of_questions=["What happens?"],
        )

    # both videos processed; requested keys present with empty fallbacks
    assert len(results) == 2
    for key in ("video1", "video2"):
        assert results[key]["summary"] == ""
        assert results[key]["vqa_answers"] == []


class _FailingAudioModel:
    """Audio model stub whose transcribe always fails."""

    model_id = "broken-whisper"

    def transcribe(self, audio_path, language=None):
        raise RuntimeError("model 'broken-whisper' not found")


def test_audio_failure_degrades_to_visual_only(mock_model, monkeypatch, tmp_path):
    """A failing audio endpoint warns and continues with visual-only analysis."""
    video_file = tmp_path / "v.mp4"
    video_file.write_bytes(b"not a real video")

    detector = VideoSummaryDetector(
        summary_model=mock_model, audio_model=_FailingAudioModel()
    )

    # force the audio path to fail without invoking ffmpeg
    def _raise_audio(filename):
        raise RuntimeError("model 'broken-whisper' not found")

    monkeypatch.setattr(detector, "_extract_transcribe_audio_part", _raise_audio)

    # stub the local visual pipeline so no ffmpeg/opencv is needed
    monkeypatch.setattr(
        detector,
        "_extract_frame_timestamps_from_clip",
        lambda filename: {
            "segments": [{"start_time": 0.0, "end_time": 1.0, "duration": 1.0}],
            "video_meta": {"width": 10, "height": 10},
        },
    )

    def _fake_merge(audio_segs, video_segs, **kwargs):
        return [
            {
                "start_time": 0.0,
                "end_time": 1.0,
                "audio_phrases": [],
                "video_frame_timestamps": [0.0],
            }
        ]

    monkeypatch.setattr(detector, "merge_audio_visual_boundaries", _fake_merge)

    def _fake_captions(filename, merged_segments, video_meta, list_of_questions=None):
        for seg in merged_segments:
            seg["summary_bullets"] = []
            seg["vqa_bullets"] = []

    monkeypatch.setattr(
        detector, "_make_captions_from_extracted_frames", _fake_captions
    )

    entry = {"filename": str(video_file)}
    with pytest.warns(
        RuntimeWarning, match="Audio transcription failed.*broken-whisper"
    ):
        results = detector.make_captions_for_subclips(entry)

    assert isinstance(results, list)
    assert entry["audio_descriptions"] == []
    # audio model stays alive so later videos in a batch can still use it
    assert detector.audio_model is not None


def test_audio_model_used_for_every_video_in_batch(
    mock_model, mock_audio_model, monkeypatch, tmp_path
):
    """The shared audio model is reused for every video, not consumed after the first.

    Regression test: previously the audio model was closed and set to None after
    the first video, so every subsequent video in the batch silently skipped audio.
    """
    v1 = tmp_path / "v1.mp4"
    v2 = tmp_path / "v2.mp4"
    v1.write_bytes(b"not a real video")
    v2.write_bytes(b"not a real video")

    detector = VideoSummaryDetector(
        summary_model=mock_model,
        audio_model=mock_audio_model,
        subdict={
            "video1": {"filename": str(v1)},
            "video2": {"filename": str(v2)},
        },
    )

    transcribe_calls = []

    def _fake_audio(filename):
        transcribe_calls.append(filename)
        return mock_audio_model.transcribe(filename)

    monkeypatch.setattr(detector, "_extract_transcribe_audio_part", _fake_audio)

    # stub the visual pipeline so no ffmpeg/opencv is needed
    monkeypatch.setattr(
        detector,
        "_extract_frame_timestamps_from_clip",
        lambda filename: {
            "segments": [{"start_time": 0.0, "end_time": 1.0, "duration": 1.0}],
            "video_meta": {"width": 10, "height": 10},
        },
    )
    monkeypatch.setattr(
        detector,
        "merge_audio_visual_boundaries",
        lambda audio_segs, video_segs, **kwargs: [
            {
                "start_time": 0.0,
                "end_time": 1.0,
                "audio_phrases": list(audio_segs),
                "video_frame_timestamps": [0.0],
            }
        ],
    )

    def _fake_captions(filename, merged_segments, video_meta, list_of_questions=None):
        for seg in merged_segments:
            seg["summary_bullets"] = ["a bullet"]
            seg["vqa_bullets"] = []

    monkeypatch.setattr(
        detector, "_make_captions_from_extracted_frames", _fake_captions
    )

    detector.analyse_videos_from_dict(analysis_type="summary")

    # audio transcription attempted for BOTH videos (regression: was only the first)
    assert transcribe_calls == [str(v1), str(v2)]
    # the shared audio model survives the whole batch
    assert detector.audio_model is mock_audio_model


def test_frame_extraction_workers_at_least_one_on_single_core(mock_model, monkeypatch):
    """Frame extraction must not request a zero-sized thread pool on a 1-core host.

    Regression test: ``min(8, cpu_count() // 2)`` is 0 on a single-core machine,
    which made ThreadPoolExecutor raise ``max_workers must be greater than 0``.
    """
    detector = VideoSummaryDetector(summary_model=mock_model)

    monkeypatch.setattr("ammico.video_summary.os.cpu_count", lambda: 1)

    captured = {}

    def _fake_extract(filename, frame_timestamps, original_w, original_h, workers):
        captured["workers"] = workers
        return [(t, Image.new("RGB", (4, 4))) for t in frame_timestamps]

    monkeypatch.setattr(detector, "_extract_frames_ffmpeg", _fake_extract)

    merged_segments = [
        {"start_time": 0.0, "end_time": 1.0, "video_frame_timestamps": [0.0, 0.5]}
    ]
    detector._make_captions_from_extracted_frames(
        "x.mp4", merged_segments, {"width": 10, "height": 10}
    )

    assert captured["workers"] >= 1


@pytest.mark.long
def test_make_captions_for_subclips_valid_output(
    video_summary_model, get_video_testdict
):
    entry = get_video_testdict["video1"]
    questions = ["When and where the video was recorded?"]
    results = video_summary_model.make_captions_for_subclips(
        entry, list_of_questions=questions
    )

    assert isinstance(results, list)
    for segment in results:
        assert "start_time" in segment
        assert "end_time" in segment
        assert "summary_bullets" in segment
        assert isinstance(segment["summary_bullets"], list)
        assert "vqa_bullets" in segment
        assert isinstance(segment["vqa_bullets"], list)


def test_extract_transcribe_audio_part(
    mock_model, mock_audio_model, get_video_testdict
):
    video_summ = VideoSummaryDetector(
        summary_model=mock_model, audio_model=mock_audio_model
    )
    filename = get_video_testdict["video1"]["filename"]

    audio_captions = video_summ._extract_transcribe_audio_part(filename)
    assert isinstance(audio_captions, list)
    for caption in audio_captions:
        assert "start_time" in caption
        assert "end_time" in caption
        assert "text" in caption
        assert "duration" in caption
    # the shared audio model is not consumed/closed by a single transcription
    assert video_summ.audio_model is mock_audio_model


def test_extract_frame_timestamps_from_clip(mock_model, get_video_testdict):
    video_summ = VideoSummaryDetector(summary_model=mock_model)
    filename = get_video_testdict["video1"]["filename"]

    result = video_summ._extract_frame_timestamps_from_clip(filename)
    assert isinstance(result, dict)
    assert "segments" in result
    assert "video_meta" in result
    segments = result["segments"]
    assert isinstance(segments, list)
    for seg in segments:
        assert "start_time" in seg
        assert "end_time" in seg
        assert "frame_timestamps" in seg
        assert isinstance(seg["frame_timestamps"], list)
        assert all(isinstance(t, float) for t in seg["frame_timestamps"])
        assert len(seg["frame_timestamps"]) >= 1
        eps = 1e-5
        for timestamp in seg["frame_timestamps"]:
            assert seg["start_time"] - eps <= timestamp <= seg["end_time"] + eps


def test_merge_audio_visual_boundaries_type_and_size(mock_model):
    video_summ = VideoSummaryDetector(summary_model=mock_model)

    audio_segs = [
        {"start_time": 0.0, "end_time": 5.0, "text": "Hello", "duration": 5.0},
        {"start_time": 10.0, "end_time": 15.0, "text": "World", "duration": 5.0},
    ]
    video_segs = [
        {
            "start_time": 0.0,
            "end_time": 8.0,
            "duration": 8.0,
            "frame_timestamps": [1.0, 3.0, 6.0],
        },
        {
            "start_time": 12.0,
            "end_time": 20.0,
            "duration": 8.0,
            "frame_timestamps": [13.0, 14.0, 18.0],
        },
    ]

    merged = video_summ.merge_audio_visual_boundaries(audio_segs, video_segs)
    assert isinstance(merged, list)
    assert len(merged) > 0
    for seg in merged:
        assert "start_time" in seg
        assert "end_time" in seg
        assert "audio_phrases" in seg
        assert isinstance(seg["audio_phrases"], list)
        assert "video_scenes" in seg
        assert isinstance(seg["video_scenes"], list)


def test_merge_audio_visual_boundaries_no_segments(mock_model):
    video_summ = VideoSummaryDetector(summary_model=mock_model)

    audio_segs = []
    video_segs = []

    with pytest.raises(ValueError):
        video_summ.merge_audio_visual_boundaries(audio_segs, video_segs)


def test_merge_audio_visual_boundaries_empty_audio(mock_model):
    video_summ = VideoSummaryDetector(summary_model=mock_model)

    audio_segs = []
    video_segs = [
        {
            "start_time": 0.0,
            "end_time": 8.0,
            "duration": 8.0,
            "frame_timestamps": [1.0, 3.0, 6.0],
        },
        {
            "start_time": 12.0,
            "end_time": 20.0,
            "duration": 8.0,
            "frame_timestamps": [13.0, 14.0, 18.0],
        },
    ]

    merged = video_summ.merge_audio_visual_boundaries(audio_segs, video_segs)
    assert isinstance(merged, list)
    assert len(merged) > 0
    for seg in merged:
        assert "start_time" in seg
        assert "end_time" in seg
        assert "audio_phrases" in seg
        assert isinstance(seg["audio_phrases"], list)
        assert "video_scenes" in seg
        assert isinstance(seg["video_scenes"], list)


def test_reassign_video_timestamps_to_segments(mock_model):
    video_summ = VideoSummaryDetector(summary_model=mock_model)

    segments = [
        {"start_time": 0.0, "end_time": 5.0},
        {"start_time": 10.0, "end_time": 15.0},
    ]
    video_segs = [
        {"start_time": 0.0, "end_time": 8.0, "frame_timestamps": [1.0, 3.0, 6.0]},
        {"start_time": 12.0, "end_time": 20.0, "frame_timestamps": [13.0, 14.0, 18.0]},
    ]

    video_summ._reassign_video_timestamps_to_segments(segments, video_segs)

    assert isinstance(segments, list)
    assert len(segments) == 2
    assert "video_frame_timestamps" in segments[0]
    assert isinstance(segments[0]["video_frame_timestamps"], list)
    assert segments[0]["video_frame_timestamps"] == [1.0, 3.0]
    assert "video_frame_timestamps" in segments[1]
    assert isinstance(segments[1]["video_frame_timestamps"], list)
    assert segments[1]["video_frame_timestamps"] == [13.0, 14.0]


def test_combine_visual_frames_by_time(mock_model):
    video_summ = VideoSummaryDetector(summary_model=mock_model)

    video_segs = [
        {
            "start_time": 0.0,
            "end_time": 15.0,
            "duration": 15.0,
            "frame_timestamps": [1.0, 5.0, 13.0],
        },
        {
            "start_time": 20.0,
            "end_time": 40.0,
            "duration": 20.0,
            "frame_timestamps": [20.0, 25.0, 38.0],
        },
        {
            "start_time": 40.0,
            "end_time": 80.0,
            "duration": 40.0,
            "frame_timestamps": [49.0, 60.0, 75.0],
        },
    ]
    combined = video_summ._combine_visual_frames_by_time(video_segs)
    assert isinstance(combined, list)
    assert len(combined) == 4
    assert combined[0]["start_time"] == 0.0
    assert combined[0]["end_time"] == 15.0
    assert combined[1]["start_time"] == 20.0
    assert combined[1]["end_time"] == 40.0
    assert combined[2]["start_time"] == 40.0
    assert combined[2]["end_time"] == 60.0
    assert combined[3]["start_time"] == 60.0
    assert combined[3]["end_time"] == 80.0


def test_extract_frame_timestamps_long_scene(
    mock_model,
    monkeypatch,
):
    video_summ = VideoSummaryDetector(summary_model=mock_model)

    def fake_detect_scene_cuts(filename):
        return {
            "segments": [
                {
                    "type": "video_scene",
                    "start_time": 0.0,
                    "end_time": 300.0,
                    "duration": 300.0,
                }
            ],
            "video_meta": {
                "width": 1920,
                "height": 1080,
            },
        }

    monkeypatch.setattr(
        video_summ,
        "_detect_scene_cuts",
        fake_detect_scene_cuts,
    )

    result = video_summ._extract_frame_timestamps_from_clip("fake.mp4")

    timestamps = result["segments"][0]["frame_timestamps"]

    assert len(timestamps) == 11
    assert timestamps[0] == pytest.approx(0.0)
    assert timestamps[-1] == pytest.approx(300.0)

    gaps = [right - left for left, right in zip(timestamps, timestamps[1:])]

    assert all(gap <= 30.0 + 1e-4 for gap in gaps)


def test_extract_frame_timestamps_short_scene_keeps_existing_behavior(
    mock_model,
    monkeypatch,
):
    video_summ = VideoSummaryDetector(summary_model=mock_model)

    def fake_detect_scene_cuts(filename):
        return {
            "segments": [
                {
                    "type": "video_scene",
                    "start_time": 10.0,
                    "end_time": 20.0,
                    "duration": 10.0,
                }
            ],
            "video_meta": {
                "width": 1280,
                "height": 720,
            },
        }

    monkeypatch.setattr(
        video_summ,
        "_detect_scene_cuts",
        fake_detect_scene_cuts,
    )

    result = video_summ._extract_frame_timestamps_from_clip("fake.mp4")

    timestamps = result["segments"][0]["frame_timestamps"]

    assert len(timestamps) == 4
    assert timestamps[0] == pytest.approx(10.0)
    assert timestamps[-1] == pytest.approx(20.0)
    assert result["video_meta"] == {
        "width": 1280,
        "height": 720,
    }


def test_reassign_video_timestamps_segment_has_no_sampled_frame(
    mock_model,
):
    """This tests the case where a segment overlaps with a video scene, but the scene's frame timestamps do not include any frames that fall within the segment's time range. In this case, the method should add a fallback timestamp at the midpoint of the segment."""
    video_summ = VideoSummaryDetector(summary_model=mock_model)

    segments = [
        {
            "start_time": 5.0,
            "end_time": 7.0,
            "duration": 2.0,
            "audio_phrases": [
                {
                    "start_time": 5.2,
                    "end_time": 6.8,
                    "text": "Short phrase.",
                    "duration": 1.6,
                }
            ],
            "video_scenes": [],
        }
    ]

    video_segs = [
        {
            "start_time": 0.0,
            "end_time": 60.0,
            "duration": 60.0,
            "frame_timestamps": [0.0, 30.0, 60.0],
        }
    ]

    video_summ._reassign_video_timestamps_to_segments(segments, video_segs)

    assert segments[0]["video_frame_timestamps"] == [6.0]


def test_reassign_video_timestamps_segment_does_not_overlap_video(
    mock_model,
):
    """This tests the case where a segment does not overlap with any video scenes at all. In this case, the method should leave the "video_frame_timestamps" list empty, since there are no relevant frames to assign to the segment."""
    video_summ = VideoSummaryDetector(summary_model=mock_model)

    segments = [
        {
            "start_time": 70.0,
            "end_time": 75.0,
            "duration": 5.0,
            "audio_phrases": [],
            "video_scenes": [],
        }
    ]

    video_segs = [
        {
            "start_time": 0.0,
            "end_time": 60.0,
            "duration": 60.0,
            "frame_timestamps": [0.0, 30.0, 60.0],
        }
    ]

    video_summ._reassign_video_timestamps_to_segments(segments, video_segs)

    assert segments[0]["video_frame_timestamps"] == []


def test_make_captions_for_subclips_with_frame_timestamp(
    mock_model,
    monkeypatch,
    tmp_path,
):
    """This tests the case where the segment has frame timestamps that fall within its time range.
    In this case, the method should use those frame timestamps to generate captions for the segment, and the resulting summary bullets should reference the specific timestamps of the frames."""

    video_file = tmp_path / "video.mp4"
    video_file.write_bytes(b"fake video content")

    video_summ = VideoSummaryDetector(
        summary_model=mock_model,
        audio_model=None,
        subdict={},
    )

    def fake_extract_frame_timestamps_from_clip(filename):
        return {
            "segments": [
                {
                    "type": "video_scene",
                    "start_time": 10.0,
                    "end_time": 18.0,
                    "duration": 8.0,
                    "frame_timestamps": [10.0, 14.0, 18.0],
                }
            ],
            "video_meta": {
                "width": 1920,
                "height": 1080,
            },
        }

    def fake_merge_audio_visual_boundaries(audio_segs, video_segs):
        return [
            {
                "start_time": 10.0,
                "end_time": 18.0,
                "duration": 8.0,
                "audio_phrases": [],
                "video_scenes": video_segs,
                "video_frame_timestamps": [10.0, 14.0, 18.0],
                "summary_bullets": [
                    "- [10.000s] A presenter is visible.",
                ],
                "vqa_bullets": [],
            }
        ]

    def fake_make_captions_from_extracted_frames(
        filename,
        merged_segments,
        video_meta,
        list_of_questions=None,
    ):
        merged_segments[0]["summary_bullets"] = [
            "- [10.000s] A presenter is visible.",
        ]
        merged_segments[0]["vqa_bullets"] = []

    def fake_chat(messages, max_new_tokens=256, n=1):
        return [
            "A presenter explains the project timeline.",
        ]

    monkeypatch.setattr(
        video_summ,
        "_extract_frame_timestamps_from_clip",
        fake_extract_frame_timestamps_from_clip,
    )
    monkeypatch.setattr(
        video_summ,
        "merge_audio_visual_boundaries",
        fake_merge_audio_visual_boundaries,
    )
    monkeypatch.setattr(
        video_summ,
        "_make_captions_from_extracted_frames",
        fake_make_captions_from_extracted_frames,
    )
    monkeypatch.setattr(video_summ.summary_model, "chat", fake_chat)

    result = video_summ.make_captions_for_subclips(
        {"filename": str(video_file)},
    )

    assert result == [
        {
            "start_time": 10.0,
            "end_time": 18.0,
            "summary_bullets": [
                "- [10.000s] A presenter explains the project timeline.",
            ],
            "vqa_bullets": [],
        }
    ]


def test_make_captions_for_subclips_without_frame_timestamp(
    mock_model,
    monkeypatch,
    tmp_path,
):
    """This tests the case where the segment does not have any visual timestamps that fall within its time range.
    In this case, the method should not fail, but it should still generate captions based on the audio content."""
    video_file = tmp_path / "video.mp4"
    video_file.write_bytes(b"fake video content")

    video_summ = VideoSummaryDetector(
        summary_model=mock_model,
        audio_model=None,
        subdict={},
    )

    def fake_extract_frame_timestamps_from_clip(filename):
        return {
            "segments": [
                {
                    "type": "video_scene",
                    "start_time": 35.0,
                    "end_time": 38.0,
                    "duration": 3.0,
                    "frame_timestamps": [],
                }
            ],
            "video_meta": {
                "width": 1920,
                "height": 1080,
            },
        }

    def fake_merge_audio_visual_boundaries(audio_segs, video_segs):
        return [
            {
                "start_time": 35.0,
                "end_time": 38.0,
                "duration": 3.0,
                "audio_phrases": [
                    {
                        "start_time": 35.2,
                        "end_time": 37.7,
                        "text": "All stages of the project timeline are shown in a chart.",
                        "duration": 2.5,
                    }
                ],
                "video_scenes": video_segs,
                "video_frame_timestamps": [],
                "summary_bullets": [],
                "vqa_bullets": [],
            }
        ]

    def fake_make_captions_from_extracted_frames(
        filename,
        merged_segments,
        video_meta,
        list_of_questions=None,
    ):
        return None

    def fake_chat(messages, max_new_tokens=256, n=1):
        return [
            "The presenter introduces a chart about the project timeline.",
        ]

    monkeypatch.setattr(
        video_summ,
        "_extract_frame_timestamps_from_clip",
        fake_extract_frame_timestamps_from_clip,
    )
    monkeypatch.setattr(
        video_summ,
        "merge_audio_visual_boundaries",
        fake_merge_audio_visual_boundaries,
    )
    monkeypatch.setattr(
        video_summ,
        "_make_captions_from_extracted_frames",
        fake_make_captions_from_extracted_frames,
    )
    monkeypatch.setattr(video_summ.summary_model, "chat", fake_chat)

    result = video_summ.make_captions_for_subclips(
        {"filename": str(video_file)},
    )

    assert result == [
        {
            "start_time": 35.0,
            "end_time": 38.0,
            "summary_bullets": [
                "- [36.500s] The presenter introduces a chart about the project timeline.",
            ],
            "vqa_bullets": [],
        }
    ]
