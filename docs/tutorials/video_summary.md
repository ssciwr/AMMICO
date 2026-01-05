### Video summary and VQA module
Also the `MultimodalSummaryDetector` can be used to generate video captions (`summary`) as well as visual question answering (`VQA`) for visual part of video file. 

```
model = ammico.MultimodalSummaryModel(model_id=model_id)
```
For more accurate results for videos, you can take the audio part into account for your tasks. The `AudioToTextModel` model is responsible for this in `ammico`. By default, it loads a small model on the GPU (if your device supports CUDA), also you can specify size of the audio model ("small", "base", "large"), or device ("cuda" or "cpu") if you want. Increasing the model size can improve the result of converting an audio track to text, but consumes more RAM or VRAM.
```
audio_model = ammico.AudioToTextModel(model_size="small", device="cuda")
```

#### Read your video data into AMMICO
The ammico package reads in one or several input video files given in a folder for processing. The user can select to read in all videos in a folder, to include subfolders via the `recursive` option, and can select the file extension that should be considered (i.e. "mp4"). For reading in the files, the ammico function `find_videos` is used, with supported extentions supported:

| input key | input type | possible input values |
| --------- | ---------- | --------------------- |
`path` | `str` | the directory containing the video files (defaults to the location set by environment variable `AMMICO_DATA_HOME`) |
| `pattern` | `str\|list` | the file extensions to consider (defaults to "mp4", "mov", "avi", "mkv", "webm") |
| `recursive` | `bool` | include subdirectories recursively (defaults to `True`) |
| `limit` | `int` | maximum number of files to read (defaults to `5`, for all videos set to `None` or `-1`) |
| `random_seed` | `str` | the random seed for shuffling the videos; applies when only a few videos are read and the selection should be preserved (defaults to `None`) |

The `find_videos` function returns a nested dictionary that contains the file ids and the paths to the files and is empty otherwise. 
```
video_dict = ammico.find_videos(
    path=str("/insert/your/path/here/"),  # path to the folder with videos
    limit=-1,  # -1 means no limit on the number of files, by default it is set to 20
    pattern="mp4",  # file extensions to look for
)
```
#### Video summary and VQA

To instantiate class it is required to provide `MultimodalSummaryModel` and `video_dict`. Optionally you may provide `AudioToTextModel` for more precise results.
```
vid_summary_model = ammico.VideoSummaryDetector(
    summary_model=model, audio_model=audio_model, subdict=video_dict
)
```
To perform video analysis, use the `analyse_videos_from_dict()` method.
This function provides flexible options for generating summaries and performing visual question answering. 
1. `analysis_type` – defines the type of analysis to perform. Setting it to `summary` will generate a caption (summary), `questions` will prepare answers (VQA) to a list of questions as set by the user, `summary_and_questions` will do both.
2. `list_of_questions` a list of text questions to be answered by the model. This parameter is required when analysis_type is set to "questions" or "summary_and_questions".

**Example Usage**
To generate a concise video summary only:

```
summary_dict = vid_summary_model.analyse_videos_from_dict(analysis_type="summary")
```
To generate detailed summaries and answer multiple questions:

First, define a list of questions:
```
questions = ["What did people in the frame say?"]
```

Then call the function:
```
vqa_results = vid_summary_model.analyse_videos_from_dict(
    analysis_type="questions",
    list_of_questions=questions,
)
```
or, in case of both summary and VQA:
```
vqa_results = vid_summary_model.analyse_videos_from_dict(
    analysis_type="summary_and_questions",
    list_of_questions=questions,
)
```