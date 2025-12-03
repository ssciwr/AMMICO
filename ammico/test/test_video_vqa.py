from ammico.video_summary import VideoSummaryDetector
from ammico.model import AudioToTextModel

import pytest


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
    video_summary_model.audio_model = AudioToTextModel(model_size="large", device="cpu")
    questions = ["When and where the video was recorded?"]
    results = video_summary_model.analyse_videos_from_dict(
        analysis_type="summary_and_questions", list_of_questions=questions
    )
    assert "video1" in results
    assert "summary" in results["video1"]
    assert "vqa_answers" in results["video1"]
    vqa_texts = [res.lower() for res in results["video1"]["vqa_answers"]]
    assert any("heidelberg" or "urban" in text for text in vqa_texts)
    assert any("november" or "autumn" in text for text in vqa_texts)


def test_make_captions_for_subclips_invalid_dict(mock_model):
    video_summ = VideoSummaryDetector(summary_model=mock_model)

    with pytest.raises(ValueError):
        video_summ.make_captions_for_subclips(
            entry={"filename": "non_existent_file.mp4"}
        )


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


def test_extract_transcribe_audio_part(mock_model, get_video_testdict):
    audio_model = AudioToTextModel(model_size="small", device="cpu")
    video_summ = VideoSummaryDetector(summary_model=mock_model, audio_model=audio_model)
    filename = get_video_testdict["video1"]["filename"]

    audio_captions = video_summ._extract_transcribe_audio_part(filename)
    assert isinstance(audio_captions, list)
    for caption in audio_captions:
        assert "start_time" in caption
        assert "end_time" in caption
        assert "text" in caption
        assert "duration" in caption
    assert video_summ.audio_model is None


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
        assert seg["end_time"] - seg["start_time"] <= 20.0


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
