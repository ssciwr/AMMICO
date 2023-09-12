# AMMICO - AI Media and Misinformation Content Analysis Tool

![License: MIT](https://img.shields.io/github/license/ssciwr/AMMICO)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ssciwr/AMMICO/ci.yml?branch=main)
![codecov](https://img.shields.io/codecov/c/github/ssciwr/AMMICO)
![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=ssciwr_ammico&metric=alert_status)
![Language](https://img.shields.io/github/languages/top/ssciwr/AMMICO)

This package extracts data from images such as social media posts that contain an image part and a text part. The analysis can generate a very large number of features, depending on the user input.

**_This project is currently under development!_**

Use pre-processed image files such as social media posts with comments and process to collect information:
1. Text extraction from the images
    1. Language detection
    1. Translation into English or other languages
    1. Cleaning of the text, spell-check
    1. Sentiment analysis
    1. Subjectivity analysis
    1. Named entity recognition
    1. Topic analysis
1. Content extraction from the images
    1. Textual summary of the image content ("image caption") that can be analyzed further using the above tools
    1. Feature extraction from the images: User inputs query and images are matched to that query (both text and image query)
    1. Question answering   
1. Performing person and face recognition in images
    1. Face mask detection
    1. Age, gender and race detection
    1. Emotion recognition
1. Object detection in images
    1. Detection of position and number of objects in the image; currently  person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, cell phone
1. Cropping images to remove comments from posts
 

## Installation

The `AMMICO` package can be installed using pip: 
```
pip install ammico
```
This will install the package and its dependencies locally.

To make pycocotools work on Windows OS you may need to install `vs_BuildTools.exe` from https://visualstudio.microsoft.com/visual-cpp-build-tools/ and choose following elements:
- `Visual Studio extension development`
- `MSVC v143 - VS 2022 C++ x64/x86 build tools`
- `Windows 11 SDK` for Windows 11 (or `Windows 10 SDK` for Windows 10)

Be careful, it requires around 7 GB of disk space.

![Screenshot 2023-06-01 165712](https://github.com/ssciwr/AMMICO/assets/8105097/3dfb302f-c390-46a7-a700-4e044f56c30f)

## Usage

The main demonstration notebook can be found in the `notebooks` folder and also on [google colab](https://colab.research.google.com/github/ssciwr/ammico/blob/main/ammico/notebooks/DemoNotebook_ammico.ipynb)

There are further sample notebooks in the `notebooks` folder for the more experimental features:
1. Topic analysis: Use the notebook `get-text-from-image.ipynb` to analyse the topics of the extraced text.\
**You can run this notebook on google colab: [Here](https://colab.research.google.com/github/ssciwr/ammico/blob/main/ammico/notebooks/get-text-from-image.ipynb)**  
Place the data files and google cloud vision API key in your google drive to access the data.
1. Multimodal content: Use the notebook `multimodal_search.ipynb` to find the best fitting images to an image or text query.
**You can run this notebook on google colab: [Here](https://colab.research.google.com/github/ssciwr/ammico/blob/main/ammico/notebooks/multimodal_search.ipynb)**
1. Color analysis: Use the notebook `color_analysis.ipynb` to identify colors the image. The colors are then classified into the main named colors in the English language.
**You can run this notebook on google colab: [Here](https://colab.research.google.com/github/ssciwr/ammico/blob/main/ammico/notebooks/colors_analysis.ipynb)**
1. To crop social media posts use the `cropposts.ipynb` notebook. 
**You can run this notebook on google colab: [Here](https://colab.research.google.com/github/ssciwr/ammico/blob/main/ammico/notebooks/cropposts.ipynb)**

## Features
### Text extraction
The text is extracted from the images using [google-cloud-vision](https://cloud.google.com/vision). For this, you need an API key. Set up your google account following the instructions on the google Vision AI website.
You then need to export the location of the API key as an environment variable:
```
export GOOGLE_APPLICATION_CREDENTIALS="location of your .json"
```
The extracted text is then stored under the `text` key (column when exporting a csv).

[Googletrans](https://py-googletrans.readthedocs.io/en/latest/) is used to recognize the language automatically and translate into English. The text language and translated text is then stored under the `text_language` and `text_english` key (column when exporting a csv).

If you further want to analyse the text, you have to set the `analyse_text` keyword to `True`. In doing so, the text is then processed using [spacy](https://spacy.io/) (tokenized, part-of-speech, lemma, ...). The English text is cleaned from numbers and unrecognized words (`text_clean`), spelling of the English text is corrected (`text_english_correct`), and further sentiment and subjectivity analysis are carried out (`polarity`, `subjectivity`). The latter two steps are carried out using [TextBlob](https://textblob.readthedocs.io/en/dev/index.html). For more information on the sentiment analysis using TextBlob see [here](https://towardsdatascience.com/my-absolute-go-to-for-sentiment-analysis-textblob-3ac3a11d524).

The [Hugging Face transformers library](https://huggingface.co/) is used to perform another sentiment analysis, a text summary, and named entity recognition, using the `transformers` pipeline.

### Content extraction

The image content ("caption") is extracted using the [LAVIS](https://github.com/salesforce/LAVIS) library. This library enables vision intelligence extraction using several state-of-the-art models, depending on the task. Further, it allows feature extraction from the images, where users can input textual and image queries, and the images in the database are matched to that query (multimodal search). Another option is question answering, where the user inputs a text question and the library finds the images that match the query.

### Emotion recognition

Emotion recognition is carried out using the [deepface](https://github.com/serengil/deepface) and [retinaface](https://github.com/serengil/retinaface) libraries. These libraries detect the presence of faces, and their age, gender, emotion and race based on several state-of-the-art models. It is also detected if the person is wearing a face mask - if they are, then no further detection is carried out as the mask prevents an accurate prediction.

### Object detection

Object detection is carried out using [cvlib](https://github.com/arunponnusamy/cvlib) and the [YOLOv4](https://github.com/AlexeyAB/darknet) model. This library detects faces, people, and several inanimate objects; we currently have restricted the output to person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, cell phone.

### Color/hue detection

Color detection is carried out using [colorgram.py](https://github.com/obskyr/colorgram.py) and [colour](https://github.com/vaab/colour) for the distance metric. The colors can be classified into the main named colors/hues in the English language, that are red, green, blue, yellow, cyan, orange, purple, pink, brown, grey, white, black.

### Cropping of posts

Social media posts can automatically be cropped to remove further comments on the page and restrict the textual content to the first comment only.