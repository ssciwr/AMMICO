# AMMICO - AI-based Media and Misinformation Content Analysis Tool

![License: MIT](https://img.shields.io/github/license/ssciwr/AMMICO)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ssciwr/AMMICO/ci.yml?branch=main)
![codecov](https://img.shields.io/codecov/c/github/ssciwr/AMMICO)
![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=ssciwr_ammico&metric=alert_status)
![Language](https://img.shields.io/github/languages/top/ssciwr/AMMICO)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ssciwr/ammico/blob/main/docs/tutorials/ammico_demo_getting_started.ipynb)

This package extracts data from images such as social media posts that contain an image part and a text part. The analysis can generate a very large number of features, depending on the user input. See [our paper](https://dx.doi.org/10.31235/osf.io/v8txj) for a more in-depth description.

**_This project is currently under development!_**

Use pre-processed image files such as social media posts with comments and process to collect information:

1. Text extraction from the images
    - Language detection
    - Translation into English or other languages
2. Content extraction from the images
    - Textual summary of the image content ("image caption")
    - Question answering about image content
3. Content extraction from videos
    - Textual summary of the video content 
    - Question answering about video content
    - Extraction and translation of audio from the video
4. Color analysis
    - Analyse hue and percentage of color on image
5. Multimodal analysis
    - Find best matches for image content or image similarity  


## Installation

The `AMMICO` package can be installed using pip: 
```
pip install ammico
```

Or install the development version from GitHub (currently recommended for the new features):

```bash
pip install git+https://github.com/ssciwr/AMMICO.git
```
This will install the package and its dependencies locally. 


## Usage

The main demonstration notebook can be found in the `docs/tutorials` folder and also on google colab: 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ssciwr/ammico/blob/main/docs/tutorials/ammico_demo_getting_started.ipynb)

## Contributing

We welcome contributions to the ammico project! If you'd like to help improve the tool, add new features, or report or fix bugs, please follow [these guidelines](CONTRIBUTING.md).

## Reporting Issues

Please use the [issues tab](https://github.com/ssciwr/AMMICO/issues) to report bugs, request features, or start discussions.

## License

ammico is licensed under the [MIT license](LICENSE).

## Citing ammico

Ammico has been published in Comp. Comm. Res., please cite the paper as specified in the [Citations](CITATION.cff) file.

