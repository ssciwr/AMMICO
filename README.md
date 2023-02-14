# Misinformation campaign analysis

![License: MIT](https://img.shields.io/github/license/ssciwr/misinformation)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ssciwr/misinformation/ci.yml?branch=main)
![codecov](https://img.shields.io/codecov/c/github/ssciwr/misinformation)
![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=ssciwr_misinformation&metric=alert_status)
![Language](https://img.shields.io/github/languages/top/ssciwr/misinformation)

Extract data from from social media images and texts in disinformation campaigns.

**_This project is currently under development!_**

Use the pre-processed social media posts (image files) and process to collect information:
1. Cropping images to remove comments from posts
1. Text extraction from the images
1. Language recognition, translation into English, cleaning of the text/spell-check
1. Sentiment and subjectivity analysis
1. Performing person and face recognition in images, emotion recognition
1. Extraction of other non-human objects in the image
 
This development will serve the fight to combat misinformation, by providing more comprehensive data about its content and techniques. 
The ultimate goal of this project is to develop a computer-assisted toolset to investigate the content of disinformation campaigns worldwide. 

# Installation

The `misinformation` package can be installed using pip: Navigate into your package folder `misinformation/` and execute
```
pip install .
```
This will install the package and its dependencies locally.

## Installation on Windows

Some modules use [lavis]() to anaylse image content. To enable this functionality on Windows OS, you need to install some dependencies that are not available by default or can be obtained from the command line:
1. Download [Visual C++](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170) and install (see also [here](https://github.com/philferriere/cocoapi)).
1. Then install the coco API from Github
```
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
```
1. Now you can install the package by navigating to the misinformation directory and typing
```
pip install .
```
in the command prompt.

# Usage

There are sample notebooks in the `misinformation/notebooks` folder for you to explore the package:
1. Text analysis: Use the notebook `get-text-from-image.ipynb` to extract any text from the images. The text is directly translated into English. If the text should be further analysed, set the keyword `analyse_text` to `True` as demonstrated in the notebook.\
**You can run this notebook on google colab: [Here](https://colab.research.google.com/github/ssciwr/misinformation/blob/main/notebooks/get-text-from-image.ipynb)**  
Place the data files and google cloud vision API key in your google drive to access the data.
1. Facial analysis: Use the notebook `facial_expressions.ipynb` to identify if there are faces on the image, if they are wearing masks, and if they are not wearing masks also the race, gender and dominant emotion.
**You can run this notebook on google colab: [Here](https://colab.research.google.com/github/ssciwr/misinformation/blob/main/notebooks/facial_expressions.ipynb)**   
Place the data files in your google drive to access the data.**
1. Object analysis: Use the notebook `ojects_expression.ipynb` to identify certain objects in the image. Currently, the following objects are being identified: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, cell phone.

There are further notebooks that are currently of exploratory nature (`colors_expression.ipynb` to identify certain colors on the image).

# Features
## Text extraction
The text is extracted from the images using [`google-cloud-vision`](https://cloud.google.com/vision). For this, you need an API key. Set up your google account following the instructions on the google Vision AI website.
You then need to export the location of the API key as an environment variable:
`export GOOGLE_APPLICATION_CREDENTIALS="location of your .json"`
The extracted text is then stored under the `text` key (column when exporting a csv).

[Googletrans](https://py-googletrans.readthedocs.io/en/latest/) is used to recognize the language automatically and translate into English. The text language and translated text is then stored under the `text_language` and `text_english` key (column when exporting a csv).

If you further want to analyse the text, you have to set the `analyse_text` keyword to `True`. In doing so, the text is then processed using [spacy](https://spacy.io/) (tokenized, part-of-speech, lemma, ...). The English text is cleaned from numbers and unrecognized words (`text_clean`), spelling of the English text is corrected (`text_english_correct`), and further sentiment and subjectivity analysis are carried out (`polarity`, `subjectivity`). The latter two steps are carried out using [TextBlob](https://textblob.readthedocs.io/en/dev/index.html). For more information on the sentiment analysis using TextBlob see [here](https://towardsdatascience.com/my-absolute-go-to-for-sentiment-analysis-textblob-3ac3a11d524).

## Emotion recognition

## Object detection

## Cropping of posts