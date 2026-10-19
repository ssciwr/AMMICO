# Comparing vision-language models with OpenRouter

[![Open this tutorial on Google colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ssciwr/ammico/blob/main/docs/tutorials/ammico_demo_openrouter_model_comparison.ipynb)

[Tutorial notebook](https://github.com/ssciwr/AMMICO/tree/main/docs/tutorials/ammico_demo_openrouter_model_comparison.ipynb)

Which vision-language model works best depends on your images and your research question. This tutorial runs the [image summary and VQA](image_summary.md) analysis with several models and compares the results side by side.

The models are accessed through [OpenRouter](https://openrouter.ai), which provides many hosted models (OpenAI, Google, Anthropic, Qwen, Meta, Mistral, ...) through a single **OpenAI-compatible API** and a single API key. Any model that accepts images as input can be used; see the [OpenRouter model list](https://openrouter.ai/models?input_modalities=image).

**Privacy note:** with OpenRouter, your images are sent to a third party (OpenRouter and the provider of the selected model).

## Connect to OpenRouter

Create an API key at [openrouter.ai/keys](https://openrouter.ai/keys) and set it as the environment variable `OPENROUTER_API_KEY`. Then point `ammico` to the OpenRouter endpoint:
```
import os
os.environ["AMMICO_API_BASE_URL"] = "https://openrouter.ai/api/v1"
os.environ["AMMICO_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
os.environ["AMMICO_PRIVACY_ACK"] = "True"  # acknowledge sending data externally
```
Only the model id changes between the models, for example:
```
model = ammico.InferenceModel(model_id="openai/gpt-4o-mini")
```

## Run and save the results per model

In the notebook, you list the models you want to compare:
```
models = [
    "openai/gpt-4o-mini",
    "google/gemini-2.5-flash",
    "qwen/qwen2.5-vl-72b-instruct",
    "meta-llama/llama-4-maverick",
]
```
Each model is run on a copy of the same image dictionary, and its results are saved to their own csv file, `openrouter_results/<provider>__<model>.csv`. Models that already have a result file are skipped, so you can add models later without running (and paying for) the previous ones again. Set `overwrite = True` to re-run all models.

## Compare the results

The saved results are read back in and combined into a table with one row per image and one column per model, for either the captions or the answers to your questions. The notebook also displays each image together with the output of all models, and exports both the comparison table and a combined table of all results (with a `model` column) to csv.
