# Tutorial

This tutorial demonstrates how to use AMMICO to analyze text on images and image content.

[![Open this tutorial on Google colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ssciwr/ammico/blob/main/docs/tutorials/ammico_demo_getting_started.ipynb)

[Tutorial notebook](https://github.com/ssciwr/AMMICO/tree/main/docs/tutorials/ammico_demo_getting_started.ipynb)

## Installation

First, install the package using pip:

```bash
pip install ammico
```

Or install the development version from GitHub (currently recommended for the new features):

```bash
pip install git+https://github.com/ssciwr/AMMICO.git
```

## Step 0: Set up Credentials

For text extraction using the Google Cloud Vision API, you need to set your API key environment variable.

```python
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/your/key.json"
```
How to obtain this key is described in [setting up credentials](../set_up_credentials.md). However, you only need this if you plan to extract text from images, not for image summary, VQA or video analysis.

## Step 1: Read Data

AMMICO reads in files from a directory. You can iterate through directories in a recursive manner and filter by extensions. Note that the order of the files may vary on different OS. Reading in these files creates a dictionary `image_dict`, with one entry per image file, containing the file path and filename. This dictionary is the main data structure that ammico operates on and is extended successively with each detector run as explained below.

```python
import ammico

# Define your data path
data_path = "./data-test"

# Find files and create the image dictionary
image_dict = ammico.find_files(
    path=data_path,
    limit=20, # Limit the number of files to process (optional)
)
```

## Step 2: Interactive Analysis (Optional)

You can launch an interactive Dash interface to inspect your data and test different detector settings before running a full analysis. This is mostly useful if you want to try out different models or settings for the analysis.

```python
# Launch the explorer
analysis_explorer = ammico.AnalysisExplorer(image_dict)
analysis_explorer.run_server(port=8055)
```

## Step 3: Run Detectors

You can run various detectors on your images. The results are stored in the `image_dict`. For example, running the text detector will add the text to the dictionary:
```
for key in image_dict:
    image_dict[key] = ammico.TextDetector(
        image_dict[key],
    ).analyse_image()
```
This will iterate over all images in the dictionary and run the text detector on each one.
For batching mode, there are also advanced options that are described in the more focused tutorials. You can run all detectors on the `image_dict`, and the order does not matter.
```
for key in image_dict:
    image_dict[key] = ammico.TextDetector(
        image_dict[key],
    ).analyse_image()
    image_dict[key] = ammico.ColorDetector(
        image_dict[key],
    ).analyse_image()
```
Note that for the image summary detector, you need to initialize the model first and create an instance of the detector class:
```
# Initialize the model once
model = ammico.MultimodalSummaryModel(device="cuda") # Use "cpu" if no GPU available

# Initialize the detector
image_summary_detector = ammico.ImageSummaryDetector(
    subdict=image_dict, 
    summary_model=model
)
```
Now you can run the inference task:
```
summary = image_summary_detector.analyse_images_from_dict(
    analysis_type="summary", is_concise_summary=True
)
```
For more information, consult the [demonstration notebook](ammico_demo_getting_started.ipynb) [![on Google colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ssciwr/ammico/blob/docs/migrate-to-mkdocs/docs/tutorials/ammico_demo_getting_started.ipynb) or the more in-depth sections about each analysis tool.

### Privacy Disclosure

The text detector requires you to accept a disclosure statement, since it sends data to the Google Cloud for processing. You will be prompted for this automatically when you invoke the detector, or you may run it separately using

```python
# For TextDetector (uses Google Cloud)
ammico.privacy_disclosure(accept_privacy="PRIVACY_AMMICO")

```

## Step 4: Export Results

Convert the results dictionary to a pandas DataFrame and save it to a CSV file.

```python
# Convert to DataFrame
image_df = ammico.get_dataframe(image_dict)

# Save to CSV
image_df.to_csv("ammico_results.csv")
```

You can now inspect `ammico_results.csv` to see all extracted features, including text, translations, image summaries and so on, and perform further analysis on the data.
