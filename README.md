# AMMICO - AI-based Media and Misinformation Content Analysis Tool

![License: MIT](https://img.shields.io/github/license/ssciwr/AMMICO)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ssciwr/AMMICO/ci.yml?branch=main)
![codecov](https://img.shields.io/codecov/c/github/ssciwr/AMMICO)
![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=ssciwr_ammico&metric=alert_status)
![Language](https://img.shields.io/github/languages/top/ssciwr/AMMICO)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ssciwr/ammico/blob/main/ammico/notebooks/DemoNotebook_ammico.ipynb)

This package extracts data from images such as social media posts that contain an image part and a text part. The analysis can generate a very large number of features, depending on the user input. See [our paper](https://dx.doi.org/10.31235/osf.io/v8txj) for a more in-depth description.

**_This project is currently under development!_**

Use pre-processed image files such as social media posts with comments and process to collect information:
1. Text extraction from the images
    1. Language detection
    1. Translation into English or other languages
1. Content extraction from the images
    1. Textual summary of the image content ("image caption") that can be analyzed further using the above tools
    1. Feature extraction from the images: User inputs query and images are matched to that query (both text and image query)
    1. Question answering about image content
1. Content extractioni from the videos
    1. Textual summary of the video content that can be analyzed further
    1. Question answering about video content
1. Color analysis
    1. Analyse hue and percentage of color on image
1. Multimodal analysis
    1. Find best matches for image content or image similarity
1. Cropping images to remove comments from posts
 
## Installation

The `AMMICO` package can be installed using pip: 
```
pip install ammico 
```
This will install the package and its dependencies locally. If after installation you get some errors when running some modules, please follow the instructions in the [FAQ](https://ssciwr.github.io/AMMICO/build/html/faq_link.html). 

## Usage

Demonstration notebooks can be found in the `docs/tutorials` folder and also on google colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ssciwr/ammico/blob/docs/migrate-to-mkdocs/docs/tutorials/ammico_demo_getting_started.ipynb)

## Features
### Text extraction
The text is extracted from the images using [google-cloud-vision](https://cloud.google.com/vision). For this, you need an API key. Set up your google account following the instructions on the google Vision AI website or as described [here](https://ssciwr.github.io/AMMICO/build/html/create_API_key_link.html).
You then need to export the location of the API key as an environment variable:
```
export GOOGLE_APPLICATION_CREDENTIALS="location of your .json"
```
The extracted text is then stored under the `text` key (column when exporting a csv).

[Googletrans](https://py-googletrans.readthedocs.io/en/latest/) is used to recognize the language automatically and translate into English. The text language and translated text is then stored under the `text_language` and `text_english` key (column when exporting a csv).

### Content extraction from images and videos

The image and video content ("caption") is extracted using [QWEN 2.5 Vision-Language model family](https://huggingface.co/collections/Qwen/qwen25-vl). Qwen2.5-VL is a multimodal large language model capable of understanding and generating content from both images and videos. With its help, `ammico` supports tasks such as image/video summarization and image/video visual question answering, where the model answers users' questions about the context of a media file.

The audio transcription, language detection and translation is carried out using the [WhisperX model family](https://github.com/m-bain/whisperX) for audio transcription as [developed by OpenAI](https://arxiv.org/abs/2303.00747).

### Color/hue detection

Color detection is carried out using [colorgram.py](https://github.com/obskyr/colorgram.py) and [colour](https://github.com/vaab/colour) for the distance metric. The colors can be classified into the main named colors/hues in the English language, that are red, green, blue, yellow, cyan, orange, purple, pink, brown, grey, white, black.

## Contributing

We welcome contributions to the ammico project! If you'd like to help improve the tool, add new features, or report and fix bugs, please follow [these guidelines](CONTRIBUTING.md).

## Reporting Issues

Please use the [issues tab](https://github.com/ssciwr/AMMICO/issues) to report bugs, request features, or start discussions.

## License

ammico is licensed under the [MIT license](LICENSE).

## Citing ammico

Ammico has been published in Comp. Comm. Res., please cite the paper as specified in the [Citations](CITATION.cff) file.