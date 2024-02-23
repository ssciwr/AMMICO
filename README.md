# AMMICO - AI Media and Misinformation Content Analysis Tool

![License: MIT](https://img.shields.io/github/license/ssciwr/AMMICO)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ssciwr/AMMICO/ci.yml?branch=main)
![codecov](https://img.shields.io/codecov/c/github/ssciwr/AMMICO)
![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=ssciwr_ammico&metric=alert_status)
![Language](https://img.shields.io/github/languages/top/ssciwr/AMMICO)

This package extracts data from images such as social media posts that contain an image part and a text part. The analysis can generate a very large number of features, depending on the user input. See [our paper](https://dx.doi.org/10.31235/osf.io/v8txj) for a more in-depth description.

**_This project is currently under development!_**

Use pre-processed image files such as social media posts with comments and process to collect information:
1. Text extraction from the images
    1. Language detection
    1. Translation into English or other languages
    1. Cleaning of the text, spell-check
    1. Sentiment analysis
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
This will install the package and its dependencies locally. If after installation you get some errors when running some modules, please follow the instructions below. 

## Compatibility problems solving

Some ammico components require `tensorflow` (e.g. Emotion detector), some `pytorch` (e.g. Summary detector). Sometimes there are compatibility problems between these two frameworks. To avoid these problems on your machines, you can prepare proper environment before installing the package (you need conda on your machine):

### 1. First, install tensorflow (https://www.tensorflow.org/install/pip)
- create a new environment with python and activate it

    ```conda create -n ammico_env python=3.10```

    ```conda activate ammico_env```
- install cudatoolkit from conda-forge

    ``` conda install -c conda-forge cudatoolkit=11.8.0```
- install nvidia-cudnn-cu11 from pip

    ```python -m pip install nvidia-cudnn-cu11==8.6.0.163```
- add script that runs when conda environment `ammico_env` is activated to put the right libraries on your LD_LIBRARY_PATH

    ```
    mkdir -p $CONDA_PREFIX/etc/conda/activate.d
    echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    echo 'export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    ```
- deactivate and re-activate conda environment to call script above

    ```conda deactivate```

    ```conda activate ammico_env ```

- install tensorflow

    ```python -m pip install tensorflow==2.12.1```

### 2. Second, install pytorch

-   install pytorch for same cuda version as above

    ```python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```
    
### 3. After we prepared right environment we can install the ```ammico``` package

-    ```python -m pip install ammico``` 

It is done.
    
### Micromamba
If you are using micromamba you can prepare environment with just one command: 

```micromamba create --no-channel-priority -c nvidia -c pytorch -c conda-forge -n ammico_env "python=3.10" pytorch torchvision torchaudio pytorch-cuda "tensorflow-gpu<=2.12.3" "numpy<=1.23.4"```  
   
### Windows

To make pycocotools work on Windows OS you may need to install `vs_BuildTools.exe` from https://visualstudio.microsoft.com/visual-cpp-build-tools/ and choose following elements:
- `Visual Studio extension development`
- `MSVC v143 - VS 2022 C++ x64/x86 build tools`
- `Windows 11 SDK` for Windows 11 (or `Windows 10 SDK` for Windows 10)

Be careful, it requires around 7 GB of disk space.

![Screenshot 2023-06-01 165712](https://github.com/ssciwr/AMMICO/assets/8105097/3dfb302f-c390-46a7-a700-4e044f56c30f)

## Usage

The main demonstration notebook can be found in the `notebooks` folder and also on [google colab](https://colab.research.google.com/github/ssciwr/ammico/blob/main/ammico/notebooks/DemoNotebook_ammico.ipynb).

There are further sample notebooks in the `notebooks` folder for the more experimental features:
1. Topic analysis: Use the notebook `get-text-from-image.ipynb` to analyse the topics of the extraced text.\
**You can run this notebook on google colab: [Here](https://colab.research.google.com/github/ssciwr/ammico/blob/main/ammico/notebooks/get-text-from-image.ipynb)**  
Place the data files and google cloud vision API key in your google drive to access the data.
1. To crop social media posts use the `cropposts.ipynb` notebook. 
**You can run this notebook on google colab: [Here](https://colab.research.google.com/github/ssciwr/ammico/blob/main/ammico/notebooks/cropposts.ipynb)**

## Features
### Text extraction
The text is extracted from the images using [google-cloud-vision](https://cloud.google.com/vision). For this, you need an API key. Set up your google account following the instructions on the google Vision AI website or as described [here](docs/source/set_up_credentials.md).
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

### Color/hue detection

Color detection is carried out using [colorgram.py](https://github.com/obskyr/colorgram.py) and [colour](https://github.com/vaab/colour) for the distance metric. The colors can be classified into the main named colors/hues in the English language, that are red, green, blue, yellow, cyan, orange, purple, pink, brown, grey, white, black.

### Cropping of posts

Social media posts can automatically be cropped to remove further comments on the page and restrict the textual content to the first comment only.


# FAQ

## What happens to the images that are sent to google Cloud Vision?

According to the [google Vision API](https://cloud.google.com/vision/docs/data-usage), the images that are uploaded and analysed are not stored and not shared with third parties:

> We won't make the content that you send available to the public. We won't share the content with any third party. The content is only used by Google as necessary to provide the Vision API service. Vision API complies with the Cloud Data Processing Addendum.

> For online (immediate response) operations (`BatchAnnotateImages` and `BatchAnnotateFiles`), the image data is processed in memory and not persisted to disk.
For asynchronous offline batch operations (`AsyncBatchAnnotateImages` and `AsyncBatchAnnotateFiles`), we must store that image for a short period of time in order to perform the analysis and return the results to you. The stored image is typically deleted right after the processing is done, with a failsafe Time to live (TTL) of a few hours.
Google also temporarily logs some metadata about your Vision API requests (such as the time the request was received and the size of the request) to improve our service and combat abuse.

## What happens to the text that is sent to google Translate?

According to [google Translate](https://cloud.google.com/translate/data-usage), the data is not stored after processing and not made available to third parties:

> We will not make the content of the text that you send available to the public. We will not share the content with any third party. The content of the text is only used by Google as necessary to provide the Cloud Translation API service. Cloud Translation API complies with the Cloud Data Processing Addendum.

> When you send text to Cloud Translation API, text is held briefly in-memory in order to perform the translation and return the results to you.

## What happens if I don't have internet access - can I still use ammico?

Some features of ammico require internet access; a general answer to this question is not possible, some services require an internet connection, others can be used offline:

- Text extraction: To extract text from images, and translate the text, the data needs to be processed by google Cloud Vision and google Translate, which run in the cloud. Without internet access, text extraction and translation is not possible.
- Image summary and query: After an initial download of the models, the `summary` module does not require an internet connection.
- Facial expressions: After an initial download of the models, the `faces` module does not require an internet connection.
- Multimodal search: After an initial download of the models, the `multimodal_search` module does not require an internet connection.
- Color analysis: The `color` module does not require an internet connection.
