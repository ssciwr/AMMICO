# Multimodal Summary and Visual Question Anwering

[![Open this tutorial on Google colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ssciwr/ammico/blob/docs/migrate-to-mkdocs/docs/tutorials/ammico_demo_getting_started.ipynb)

This detector is built on the [QWEN 2.5 Vision-Language model family](https://huggingface.co/collections/Qwen/qwen25-vl). In this project, two model variants are supported: 

1. `Qwen2.5-VL-3B-Instruct`, which requires approximately 3 GB of video memory to load.
2. `Qwen2.5-VL-7B-Instruct`, which requires up to 8 GB of VRAM for initialization.

Each version can be run on the CPU, but this will significantly increase the operating time, so we cannot recommend it, but we retain this option. 
The model type can be specified when initializing the `MultimodalSummaryModel` class:
```
model_id = (
    "Qwen/Qwen2.5-VL-7B-Instruct"  # or "Qwen/Qwen2.5-VL-3B-Instruct" respectively
)
model = ammico.MultimodalSummaryModel(model_id=model_id)
```
You can also define the preferred device type ("cpu" or "cuda") explicitly during initialization:
```
model = ammico.MultimodalSummaryModel(model_id=model_id, device="cuda")
```
By default, the initialization follows this logic:

If a GPU is available, it is automatically detected and the model defaults to Qwen2.5-VL-7B-Instruct on "cuda".

If no GPU is detected, the system falls back to the Qwen2.5-VL-3B-Instruct model on the "cpu" device.
```
model = ammico.MultimodalSummaryModel()
```
To instantiate class it is required to provide `MultimodalSummaryModel` and dictionary
```
image_summary_vqa = ammico.ImageSummaryDetector(summary_model=model, subdict=image_dict)
```
To perform image analysis, use the `analyse_images_from_dict()` method.
This function provides flexible options for generating summaries and performing visual question answering. 

1. `analysis_type` – defines the type of analysis to perform. Setting it to `summary` will generate a caption (summary), `questions` will prepare answers (VQA) to a list of questions as set by the user, `summary_and_questions` will do both.
2. `list_of_questions` a list of text questions to be answered by the model. This parameter is required when analysis_type is set to "questions" or "summary_and_questions".
3. `keys_batch_size` controls the number of images processed per batch. Increasing this value may slightly improve performance, depending on your system.
The default is `16`, which provides a good balance between speed and stability on most setups.
4. `is_concise_summary` – determines the level of detail in generated captions:
    * `True` → produces short, concise summaries.
    * `False` → produces longer, more descriptive captions that may include additional context or atmosphere, but take more time to compute.
5. `is_concise_answer`– similar to the previous flag, but for controlling the level of detail in question answering responses.

## Read your image data into `ammico`

`ammico` reads in files from a directory. You can iterate through directories in a recursive manner and filter by extensions. Note that the order of the files may vary on different OS. Reading in these files creates a dictionary `image_dict`, with one entry per image file, containing the file path and filename. This dictionary is the main data structure that ammico operates on and is extended successively with each detector run as explained below.

For reading in the files, the ammico function `find_files` is used, with optional keywords:

| input key | input type | possible input values |
| --------- | ---------- | --------------------- |
`path` | `str` | the directory containing the image files (defaults to the location set by environment variable `AMMICO_DATA_HOME`) |
| `pattern` | `str\|list` | the file extensions to consider (defaults to "png", "jpg", "jpeg", "gif", "webp", "avif", "tiff") |
| `recursive` | `bool` | include subdirectories recursively (defaults to `True`) |
| `limit` | `int` | maximum number of files to read (defaults to `20`, for all images set to `None` or `-1`) |
| `random_seed` | `int` | the random seed for shuffling the images; applies when only a few images are read and the selection should be preserved (defaults to `None`) |

## Example usage

To generate a concise image summary only:
```
summary = image_summary_vqa.analyse_images_from_dict(
    analysis_type="summary", is_concise_summary=True
)
```
To generate detailed summaries and answer multiple questions:

First, define a list of questions:
```
list_of_questions = [
    "How many persons on the picture?",
    "Are there any politicians in the picture?",
    "Does the picture show something from medicine?",
]
```
Then call the function:
```
summary_and_answers = ammico.analyse_images_from_dict(
    analysis_type="summary_and_questions",
    list_of_questions=list_of_questions,
    is_concise_summary=False,
    is_concise_answer=False,
)
```
The output of the `analyse_images_from_dict()` method is a dictionary, where each key corresponds to an input image identifier. Each entry in this dictionary contains the processed results for that image.

| output key | output type | output value |
| ---------- | ----------- | ------------ |
| `caption` | `str` | when `analysis_type="summary"` or `"summary_and_questions"`, constant image caption |
| `vqa` | `list[str]` | when `analysis_type="questions"` or `summary_and_questions`, the answers to the user-defined input question |
