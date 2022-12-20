# Misinformation campaign analysis

![License: MIT](https://img.shields.io/github/license/ssciwr/misinformation)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ssciwr/misinformation/ci.yml?branch=main)
![codecov](https://img.shields.io/codecov/c/github/ssciwr/misinformation)
![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=ssciwr_misinformation&metric=alert_status)
![Language](https://img.shields.io/github/languages/top/ssciwr/misinformation)

Extract data from from social media images and texts in disinformation campaigns.

**_This project is currently under development!_**

Use the pre-processed social media posts (image files) and process to collect information:
1. Text extraction from the images
1. Improving the preparation of the text for the data analysis (e.g., text cleaning)
1. Performing person and face recognition in images, facial expressions recognition, as well as the extraction of any other available individual characteristics (e.g., gender, clothes) 
1. Extraction of other non-human objects in the image
1. 5-Color analysis of the images
  
This development will serve the fight to combat misinformation, by providing more comprehensive data about its content and techniques. 
The ultimate goal of this project is to develop a computer-assisted toolset to investigate the content of disinformation campaigns worldwide. 

# Installation

The `misinformation` package can be installed using pip: Navigate into your package folder `misinformation/` and execute
```
pip install .
```
This will install the package and its dependencies locally.

# Usage

There are sample notebooks in the `misinformation/notebooks` folder for you to explore the package usage:
1. Facial analysis: Use the notebook `facial_expressions.ipynb` to identify if there are faces on the image, if they are wearing masks, and if they are not wearing masks also the race, gender and dominant emotion.
1. Object analysis: Use the notebook `ojects_expression.ipynb` to identify certain objects in the image. Currently, the following objects are being identified: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, cell phone.

There are further notebooks that are currently of exploratory nature (`colors_expression` to identify certain colors on the image, `get-text-from-image` to extract text that is contained in an image.)
