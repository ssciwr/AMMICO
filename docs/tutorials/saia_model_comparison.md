# Comparing vision-language models with GWDG SAIA

[![Open this tutorial on Google colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ssciwr/ammico/blob/main/docs/tutorials/ammico_demo_saia_model_comparison.ipynb)

[Tutorial notebook](https://github.com/ssciwr/AMMICO/tree/main/docs/tutorials/ammico_demo_saia_model_comparison.ipynb)

Which vision-language model works best depends on your images and your research question. This tutorial runs the [image summary and VQA](image_summary.md) analysis with several models and compares the results side by side, in the same way as the [model comparison with OpenRouter](model_comparison.md).

The models are accessed through [SAIA](https://docs.hpc.gwdg.de/services/ai-services/saia/index.html), the Scalable AI Accelerator of the [GWDG](https://gwdg.de), which provides the models of the Chat AI service through an **OpenAI-compatible API**. SAIA is available to researchers at German universities and research institutions via [KISSKI](https://kisski.gwdg.de). Any model with the capability "image" can be used; see the [list of available models](https://docs.hpc.gwdg.de/services/ai-services/saia/index.html).

**Privacy note:** with SAIA, your images are sent to the GWDG. According to the GWDG, the prompts and message contents sent to the models hosted at the GWDG are not stored. For the external models offered by Chat AI (for example from OpenAI and Anthropic), your data is passed on to the external provider; the tutorial only uses models hosted at the GWDG.

## Connect to SAIA

Request an API key on the [KISSKI LLM service page](https://kisski.gwdg.de/en/leistungen/2-02-llm-service) and set it as the environment variable `SAIA_API_KEY`. Then point `ammico` to the SAIA endpoint:
```
import os
os.environ["AMMICO_API_BASE_URL"] = "https://chat-ai.academiccloud.de/v1"
os.environ["AMMICO_API_KEY"] = os.environ["SAIA_API_KEY"]
os.environ["AMMICO_PRIVACY_ACK"] = "True"  # acknowledge sending data externally
```
Only the model id changes between the models, for example:
```
model = ammico.InferenceModel(model_id="gemma-4-31b-it")
```

## Request limits

The number of requests per API key is limited per minute, hour, day and month (for example 10 requests per minute). The notebook shows you how to check the limits of your key. Since `ammico` sends many requests at the same time by default, the notebook uses a small subclass of `InferenceModel` that sends the requests one after the other and spaces them out to stay below the per-minute limit. Each image needs one request for the caption and one request per question, so start with a small number of images.

## Run and save the results per model

In the notebook, you list the models you want to compare:
```
models = [
    "gemma-4-31b-it",
    "qwen3.6-35b-a3b",
    "qwen3.8-27b",
    "glm-5.3-flash",
]
```
Each model is run on a copy of the same image dictionary, and its results are saved to their own csv file, `saia_results/<model>.csv`. Models that already have a result file are skipped, so you can add models later without running the previous ones again. Set `overwrite = True` to re-run all models.

## Compare the results

The saved results are read back in and combined into a table with one row per image and one column per model, for either the captions or the answers to your questions. The notebook also displays each image together with the output of all models, and exports both the comparison table and a combined table of all results (with a `model` column) to csv.
