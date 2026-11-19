"""Helpers for running the ammico image summary / VQA with several OpenRouter models.

Used by ``workshop_multi_model_112026.ipynb``. Every model works on its own copy of
the image dictionary, so the results of the different models do not overwrite each
other. Results are kept in a dictionary ``{model_id: image_dict}``.
"""

from __future__ import annotations

import copy
import os
import random
import textwrap
import time
from getpass import getpass
from pathlib import Path

import matplotlib.pyplot as plt
import openai
import pandas as pd
from PIL import Image

import ammico

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# the result key each analysis type writes into the image dictionary
RESULT_KEYS = {"summary": ["caption"], "questions": ["vqa"]}
RESULT_KEYS["summary_and_questions"] = RESULT_KEYS["summary"] + RESULT_KEYS["questions"]


def connect(new_key: bool = False):
    """Point ammico to OpenRouter and check that the API key is accepted.

    Asks for the API key if ``OPENROUTER_API_KEY`` is not set, or always if
    ``new_key=True`` (use this to replace a key entered earlier in the session).
    """
    key = "" if new_key else os.environ.get("OPENROUTER_API_KEY", "")
    if not key.strip():
        key = getpass("Enter your OpenRouter API key: ")
    key = key.strip().strip("'\"")
    os.environ["OPENROUTER_API_KEY"] = key
    os.environ["AMMICO_API_BASE_URL"] = OPENROUTER_BASE_URL
    os.environ["AMMICO_API_KEY"] = key
    os.environ["AMMICO_PRIVACY_ACK"] = "True"  # acknowledge sending data externally
    _check_key(key)


def _check_key(key: str):
    """Fail early with a clear message if OpenRouter does not accept the key."""
    client = openai.OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)
    try:
        client.get("/key", cast_to=object)  # key info; free, rejects unknown keys
    except openai.AuthenticationError as error:
        raise RuntimeError(
            f"OpenRouter rejected the API key (starts with {key[:8]!r}, "
            f"length {len(key)}): {error}. Create a new key at "
            "https://openrouter.ai/keys and run OR.connect(new_key=True)."
        ) from None
    print(f"Connected to OpenRouter (key starts with {key[:8]!r}).")


class OpenRouterModel(ammico.InferenceModel):
    """InferenceModel that retries rate-limited requests and optionally switches off reasoning.

    Rate-limited (429) or temporarily unavailable (5xx, connection, timeout) requests
    are retried up to ``max_retries`` times, with a waiting time that doubles after each
    attempt (about 5, 10, 20, 40, 80, 120 s). Reasoning models spend tokens on
    "thinking", which can leave the caption empty, so reasoning is switched off by default.
    """

    def __init__(self, *args, disable_reasoning=True, max_retries=6, **kwargs):
        super().__init__(*args, **kwargs)
        self.disable_reasoning = disable_reasoning
        self.max_retries = max_retries

    def chat(self, messages, max_new_tokens=256, n=1):
        extra_body = (
            {"reasoning": {"enabled": False}} if self.disable_reasoning else None
        )
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    n=n,
                    temperature=0,
                    extra_body=extra_body,
                )
                break
            except (
                openai.RateLimitError,  # 429: rate-limited
                openai.InternalServerError,  # 5xx: provider temporarily unavailable
                openai.APIConnectionError,
                openai.APITimeoutError,
            ) as e:
                if attempt == self.max_retries:
                    raise  # give up, ammico skips the image with a warning
                wait = min(5 * 2**attempt, 120) + random.uniform(0, 2)
                print(
                    f"{self.model_id}: {type(e).__name__}, "
                    f"retry {attempt + 1}/{self.max_retries} in {wait:.0f} s"
                )
                time.sleep(wait)
        return [(choice.message.content or "").strip() for choice in response.choices]


class MultiModelDetector:
    """One ``ammico.ImageSummaryDetector`` per model, each on its own copy of ``image_dict``.

    Args:
        models: OpenRouter model ids, e.g. ``"openai/gpt-4o-mini"``.
        image_dict: the image dictionary from ``ammico.find_files``.
        max_parallel_images: images processed at the same time (per model).
        max_parallel_requests: requests (questions) per image at the same time.
        **model_kwargs: passed on to :class:`OpenRouterModel`
            (``disable_reasoning``, ``max_retries``).
    """

    def __init__(
        self,
        models,
        image_dict,
        max_parallel_images=4,
        max_parallel_requests=2,
        **model_kwargs,
    ):
        self.max_parallel_images = max_parallel_images
        self.results = {}
        self.detectors = {}
        for model_id in models:
            model = OpenRouterModel(
                model_id=model_id, max_concurrency=max_parallel_requests, **model_kwargs
            )
            self.results[model_id] = copy.deepcopy(image_dict)
            self.detectors[model_id] = ammico.ImageSummaryDetector(
                summary_model=model, subdict=self.results[model_id]
            )

    def analyse_images_from_dict(self, analysis_type, overwrite=False, **kwargs):
        """Run the analysis with every model; returns ``{model_id: image_dict}``.

        Only images without a result for this analysis are sent to a model, so that
        running the cell again retries just the images that failed (e.g. because of a
        rate limit) without paying again for the others. Set ``overwrite=True`` to
        re-run all images. Further keyword arguments are passed on to
        ``ImageSummaryDetector.analyse_images_from_dict``.
        """
        keys = RESULT_KEYS[str(analysis_type)]
        for model_id, detector in self.detectors.items():
            model_dict = self.results[model_id]
            todo = {
                key: entry
                for key, entry in model_dict.items()
                if overwrite or any(_is_missing(entry.get(k)) for k in keys)
            }
            if not todo:
                print(f"{model_id}: all {len(model_dict)} images done, skipped")
                continue
            print(f"{model_id}: running on {len(todo)} images ...")
            start = time.time()
            # the entries in todo are the entries of model_dict, updated in place
            detector.subdict = todo
            detector.analyse_images_from_dict(
                analysis_type=analysis_type,
                keys_batch_size=self.max_parallel_images,
                **kwargs,
            )
            detector.subdict = model_dict
            n_missing = sum(
                any(_is_missing(entry.get(k)) for k in keys)
                for entry in model_dict.values()
            )
            print(f"{model_id}: done in {time.time() - start:.1f} s")
            if n_missing:
                print(
                    f"{model_id}: {n_missing} images without results - "
                    "run the cell again later to retry them"
                )
        return self.results


def _is_missing(value):
    """Failed requests leave an empty caption or an empty list of answers."""
    if value is None:
        return True
    if isinstance(value, (list, tuple)):
        return len(value) == 0
    return str(value).strip() == ""


def _show_image(image_path):
    img = Image.open(image_path)
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis("off")  # Hide axes
    plt.tight_layout()
    plt.show()


def _print_indented(text):
    print(textwrap.indent(textwrap.fill(str(text), width=100), "    "))


def show_summaries(results):
    """Display each image together with the captions of all models."""
    first = next(iter(results.values()))
    for key in first:
        _show_image(first[key]["filename"])
        for model_id, model_dict in results.items():
            print(f"{model_id}:")
            _print_indented(model_dict[key].get("caption", ""))
        print()


def show_answers(results, list_of_questions, n_images=20):
    """Display images together with the answers of all models to each question.

    Only the first ``n_images`` images are shown (``None`` shows all), since
    displaying many images at once puts a high load on the Jupyter server.
    """
    first = next(iter(results.values()))
    keys = list(first)[:n_images]
    print(f"Showing {len(keys)} of {len(first)} images.\n")
    for key in keys:
        _show_image(first[key]["filename"])
        for i, question in enumerate(list_of_questions):
            print(f"Q{i + 1}: {question}")
            for model_id, model_dict in results.items():
                answers = model_dict[key].get("vqa") or []
                answer = answers[i] if i < len(answers) else ""
                print(f"  {model_id}:")
                _print_indented(answer)
        print()


def get_dataframe(results):
    """One long table: one row per image and model, the column ``model`` names the model."""
    frames = []
    for model_id, model_dict in results.items():
        df = ammico.get_dataframe(model_dict)
        df.insert(0, "model", model_id)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def save_per_model(image_df, results_dir):
    """Save the results of each model to ``<results_dir>/<provider>__<model>.csv``."""
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    for model_id, df in image_df.groupby("model", sort=False):
        outfile = Path(results_dir) / f"{model_id.replace('/', '__')}.csv"
        df.reset_index(drop=True).to_csv(outfile)
        print(f"{model_id}: saved to {outfile}")
