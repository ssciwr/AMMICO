# Tutorial

This tutorial demonstrates how to use AMMICO to analyze text on images and image content.

## Installation

First, install the package using pip:

```bash
pip install ammico
```

Or install the development version from GitHub:

```bash
pip install git+https://github.com/ssciwr/AMMICO.git
```

## Step 0: Set up Credentials

For text extraction using the Google Cloud Vision API, you need to set your API key environment variable.

```python
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/your/key.json"
```

## Step 1: Read Data

AMMICO reads in files from a directory. You can iterate through files in a recursive manner and filter by extensions.

```python
import ammico
from pathlib import Path

# Define your data path
data_path = "./data-test"

# Find files and create the image dictionary
image_dict = ammico.find_files(
    path=data_path,
    limit=20, # Limit the number of files to process (optional)
)
```

## Step 2: Interactive Analysis (Optional)

You can launch an interactive Dash interface to inspect your data and test different detector settings before running a full analysis.

```python
# Launch the explorer
analysis_explorer = ammico.AnalysisExplorer(image_dict)
analysis_explorer.run_server(port=8055)
```

## Step 3: Run Detectors

You can run various detectors on your images. The results are stored in the `image_dict`.

### Privacy and Ethical Disclosures

Some detectors require you to accept disclosure statements.

```python
# For TextDetector (uses Google Cloud)
ammico.privacy_disclosure(accept_privacy="PRIVACY_AMMICO")

# For EmotionDetector (uses DeepFace)
ammico.ethical_disclosure(accept_disclosure="DISCLOSURE_AMMICO")
```

### Text Detection

Extract text, translate it, and optionally analyze it for sentiment and named entities.

```python
for key in image_dict:
    image_dict[key] = ammico.TextDetector(
        image_dict[key],
        analyse_text=True # Optional: Perform advanced text analysis
    ).analyse_image()
```

### Emotion Detection

Detect faces and emotions.

```python
for key in image_dict:
    image_dict[key] = ammico.EmotionDetector(
        image_dict[key],
        emotion_threshold=50
    ).analyse_image()
```

### Image Summary and VQA

Use multimodal models (Qwen2.5-VL) to summarize image content or answer questions about it.

```python
# Initialize the model once
model = ammico.MultimodalSummaryModel(device="cuda") # Use "cpu" if no GPU available

# Initialize the detector
image_summary_detector = ammico.ImageSummaryDetector(
    subdict=image_dict, 
    summary_model=model
)

# Run analysis
image_summary_detector.analyse_images_from_dict(
    analysis_type="summary_and_questions",
    list_of_questions=["Describe the image.", "Is there a person in the image?"]
)
```

## Step 4: Export Results

Convert the results dictionary to a pandas DataFrame and save it to a CSV file.

```python
# Convert to DataFrame
image_df = ammico.get_dataframe(image_dict)

# Save to CSV
image_df.to_csv("ammico_results.csv")
```

You can now inspect `ammico_results.csv` to see all extracted features, including text, translations, sentiment scores, image summaries, and emotion data.
