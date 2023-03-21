# AMMICO - AI Media and Misinformation Content Analysis Tool

![License: MIT](https://img.shields.io/github/license/ssciwr/misinformation)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ssciwr/misinformation/ci.yml?branch=main)
![codecov](https://img.shields.io/codecov/c/github/ssciwr/misinformation)
![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=ssciwr_misinformation&metric=alert_status)
![Language](https://img.shields.io/github/languages/top/ssciwr/misinformation)

This package extracts data from images such as social media images, and the accompanying text/text that is included in the image. The analysis can extract a very large number of features, depending on the user input.

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

The `AMMICO` package can be installed using pip: Navigate into your package folder `misinformation/` and execute
```
pip install .
```
This will install the package and its dependencies locally.


## Usage

There are sample notebooks in the `misinformation/notebooks` folder for you to explore the package:
1. Text analysis: Use the notebook `get-text-from-image.ipynb` to extract any text from the images. The text is directly translated into English. If the text should be further analysed, set the keyword `analyse_text` to `True` as demonstrated in the notebook.\
**You can run this notebook on google colab: [Here](https://colab.research.google.com/github/ssciwr/misinformation/blob/main/notebooks/get-text-from-image.ipynb)**  
Place the data files and google cloud vision API key in your google drive to access the data.
1. Facial analysis: Use the notebook `facial_expressions.ipynb` to identify if there are faces on the image, if they are wearing masks, and if they are not wearing masks also the race, gender and dominant emotion.
**You can run this notebook on google colab: [Here](https://colab.research.google.com/github/ssciwr/misinformation/blob/main/notebooks/facial_expressions.ipynb)**   
Place the data files in your google drive to access the data.**
1. Object analysis: Use the notebook `ojects_expression.ipynb` to identify certain objects in the image. Currently, the following objects are being identified: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, cell phone.

There are further notebooks that are currently of exploratory nature (`colors_expression.ipynb` to identify certain colors on the image).

## Features
### Text extraction
The text is extracted from the images using [`google-cloud-vision`](https://cloud.google.com/vision). For this, you need an API key. Set up your google account following the instructions on the google Vision AI website.
You then need to export the location of the API key as an environment variable:
`export GOOGLE_APPLICATION_CREDENTIALS="location of your .json"`
The extracted text is then stored under the `text` key (column when exporting a csv).

[Googletrans](https://py-googletrans.readthedocs.io/en/latest/) is used to recognize the language automatically and translate into English. The text language and translated text is then stored under the `text_language` and `text_english` key (column when exporting a csv).

If you further want to analyse the text, you have to set the `analyse_text` keyword to `True`. In doing so, the text is then processed using [spacy](https://spacy.io/) (tokenized, part-of-speech, lemma, ...). The English text is cleaned from numbers and unrecognized words (`text_clean`), spelling of the English text is corrected (`text_english_correct`), and further sentiment and subjectivity analysis are carried out (`polarity`, `subjectivity`). The latter two steps are carried out using [TextBlob](https://textblob.readthedocs.io/en/dev/index.html). For more information on the sentiment analysis using TextBlob see [here](https://towardsdatascience.com/my-absolute-go-to-for-sentiment-analysis-textblob-3ac3a11d524).

### Emotion recognition

### Object detection

### Cropping of posts
