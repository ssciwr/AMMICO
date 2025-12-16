import glob
import os
from pandas import DataFrame, read_csv
import pooch
import torch
import importlib_resources
import collections
import random
from enum import Enum
from typing import List, Tuple, Optional, Union
import re
import warnings
from whisperx.alignment import DEFAULT_ALIGN_MODELS_HF, DEFAULT_ALIGN_MODELS_TORCH

pkg = importlib_resources.files("ammico")


def iterable(arg):
    return isinstance(arg, collections.abc.Iterable) and not isinstance(arg, str)


class DownloadResource:
    """A remote resource that needs on demand downloading.

    We use this as a wrapper to the pooch library. The wrapper registers
    each data file and allows prefetching through the CLI entry point
    ammico_prefetch_models.
    """

    # We store a list of defined resouces in a class variable, allowing
    # us prefetching from a CLI e.g. to bundle into a Docker image
    resources = []

    def __init__(self, **kwargs):
        DownloadResource.resources.append(self)
        self.kwargs = kwargs

    def get(self):
        return pooch.retrieve(**self.kwargs)


def ammico_prefetch_models():
    """Prefetch all the download resources"""
    for res in DownloadResource.resources:
        res.get()


class AnalysisType(str, Enum):
    SUMMARY = "summary"
    QUESTIONS = "questions"
    SUMMARY_AND_QUESTIONS = "summary_and_questions"

    @classmethod
    def _validate_analysis_type(
        cls,
        analysis_type: Union["AnalysisType", str],
        list_of_questions: Optional[List[str]],
    ) -> Tuple[str, bool, bool]:
        max_questions_per_image = 15  # safety cap to avoid too many questions
        if isinstance(analysis_type, AnalysisType):
            analysis_type = analysis_type.value

        allowed = {item.value for item in AnalysisType}
        if analysis_type not in allowed:
            raise ValueError(f"analysis_type must be one of {allowed}")

        if analysis_type in ("questions", "summary_and_questions"):
            if not list_of_questions:
                raise ValueError(
                    "list_of_questions must be provided for QUESTIONS analysis type."
                )

            if len(list_of_questions) > max_questions_per_image:
                raise ValueError(
                    f"Number of questions per image ({len(list_of_questions)}) exceeds safety cap ({max_questions_per_image}). Reduce questions or increase max_questions_per_image."
                )

        is_summary = analysis_type in ("summary", "summary_and_questions")
        is_questions = analysis_type in ("questions", "summary_and_questions")
        return analysis_type, is_summary, is_questions


class AnalysisMethod:
    """Base class to be inherited by all analysis methods."""

    def __init__(self, subdict: dict) -> None:
        self.subdict = subdict
        # define keys that will be set by the analysis

    def set_keys(self):
        raise NotImplementedError()

    def analyse_image(self):
        raise NotImplementedError()


def _validate_subdict(mydict: dict) -> None:
    """Validate the nested dictionary for analysis.

    Args:
        mydict(dict): The nested dictionary to validate.
    Returns:
        dict: The validated dictionary.
    """
    if not isinstance(mydict, dict):
        raise TypeError(
            f"Please provide a nested dictionary - you provided {type(mydict)}"
        )
    # check that we actually got a nested dict with filenames
    for key in mydict.keys():
        if not isinstance(mydict[key], dict):
            raise ValueError(
                "Please provide a nested dictionary - you provided {}".format(key)
            )
        if "filename" not in mydict[key].keys():
            raise ValueError(
                f"Each sub-dictionary must contain a 'filename' key - missing in {key}"
            )


def _match_pattern(path, pattern, recursive):
    # helper function for find_files
    # find all matches for a single pattern.

    if pattern.startswith("."):
        pattern = pattern[1:]
    if recursive:
        search_path = f"{path}/**/*.{pattern}"
    else:
        search_path = f"{path}/*.{pattern}"
    return list(glob.glob(search_path, recursive=recursive))


def _limit_results(results, limit):
    # helper function for find_files
    # use -1 or None to return all images
    if limit == -1 or limit is None:
        limit = len(results)

    # limit or batch the images
    if isinstance(limit, int):
        if limit < -1:
            raise ValueError("limit must be an integer greater than 0 or equal to -1")
        results = results[:limit]

    elif iterable(limit):
        if len(limit) == 2:
            results = results[limit[0] : limit[1]]
        else:
            raise ValueError(
                f"limit must be an integer or a tuple of length 2, but is {limit}"
            )
    else:
        raise ValueError(
            f"limit must be an integer or a tuple of length 2, but is {limit}"
        )

    return results


def _extract_summary_vqa(content: str, pattern: re.Pattern) -> Tuple[str, str]:
    m = pattern.search(content)
    if not m:
        raise ValueError(
            f"Failed to parse summary and VQA answers from model output: {content}"
        )

    summary_text = m.group(1).replace("\n", " ").strip() if m.group(1) else None
    vqa_text = m.group(2).strip() if m.group(2) else None

    if not summary_text or not vqa_text:
        raise ValueError(
            f"Model output is missing either summary or VQA answers: {content}"
        )

    return summary_text, vqa_text


def _categorize_outputs(
    collected: List[Tuple[float, str]],
    include_questions: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Categorize collected outputs into summary bullets and VQA bullets.
    Args:
        collected (List[Tuple[float, str]]): List of tuples containing timestamps and generated texts.
    Returns:
        Tuple[List[str], List[str]]: A tuple containing two lists - summary bullets and VQA bullets.
    """
    MAX_CAPTIONS_FOR_SUMMARY = 600  # TODO For now, this is a constant value, but later we need to make it adjustable, with the idea of cutting out the most similar frames to reduce the load on the system.
    caps_for_summary_vqa = (
        collected[-MAX_CAPTIONS_FOR_SUMMARY:]
        if len(collected) > MAX_CAPTIONS_FOR_SUMMARY
        else collected
    )
    bullets_summary = []
    bullets_vqa = []

    if include_questions:
        pattern = re.compile(
            r"Summary\s*:\s*(.*?)(?:\s*VQA\s+Answers\s*:\s*(.*))?$",
            flags=re.IGNORECASE | re.DOTALL,
        )

        for t, c in caps_for_summary_vqa:
            summary_text, vqa_text = _extract_summary_vqa(c.strip(), pattern)
            bullets_summary.append(f"- [{t:.3f}s] {summary_text}")
            bullets_vqa.append(f"- [{t:.3f}s] {vqa_text}")
    else:
        for t, c in caps_for_summary_vqa:
            snippet = c.replace("\n", " ").strip()
            bullets_summary.append(f"- [{t:.3f}s] {snippet}")

    return bullets_summary, bullets_vqa


def _normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _remove_prefix_by_normalized_search(decoded: str, p_norm: str):
    running: list[str] = []
    for i, ch in enumerate(decoded):
        running.append(ch if not ch.isspace() else " ")
        cur_norm = _normalize_whitespace("".join(running))
        if cur_norm.endswith(p_norm):
            return decoded[i + 1 :].lstrip() if i + 1 < len(decoded) else ""
    return None


def _strip_role_prefix(decoded: str):
    m = re.match(
        r"^(?:\s*(system|user|assistant)[:\s-]*\n?)+", decoded, flags=re.IGNORECASE
    )
    if m:
        return decoded[m.end() :].lstrip()
    return None


def _strip_prompt_prefix_literal(decoded: str, prompt: str) -> str:
    """
    Remove any literal prompt prefix from decoded text using a normalized-substring match.
    """
    if not decoded:
        return ""

    if not prompt:
        return decoded.strip()

    d_norm = _normalize_whitespace(decoded)
    p_norm = _normalize_whitespace(prompt)

    if d_norm.find(p_norm) != -1:
        remainder = _remove_prefix_by_normalized_search(decoded, p_norm)
        if remainder is not None:
            return remainder

    role_stripped = _strip_role_prefix(decoded)
    if role_stripped is not None:
        return role_stripped.lstrip()

    return decoded.lstrip("\n\r ").lstrip(":;- ").strip()


def resolve_model_device(
    device: Optional[str] = None,
) -> str:
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.lower() not in ("cuda", "cpu"):
        raise ValueError("device must be 'cuda' or 'cpu'")
    if device.lower() == "cuda" and not torch.cuda.is_available():
        warnings.warn(
            "Although 'cuda' was requested, no CUDA device is available. Using CPU instead.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "cpu"
    return device.lower()


def resolve_model_size(
    model_size: str = "small",
) -> str:
    allowed_sizes = ("small", "base", "large")
    if model_size not in allowed_sizes:
        raise ValueError(f"model_size must be one of {allowed_sizes}")
    model_size = "large-v3" if model_size == "large" else model_size
    return model_size


def find_videos(
    path: str = None,
    pattern=["mp4", "mov", "avi", "mkv", "webm"],
    recursive: bool = True,
    limit=5,
    random_seed: int = None,
) -> dict:
    """Find video files on the file system."""
    if path is None:
        path = os.environ.get("AMMICO_DATA_HOME", ".")
    if isinstance(pattern, str):
        pattern = [pattern]
    results = []
    for p in pattern:
        results.extend(_match_pattern(path, p, recursive=recursive))
    if len(results) == 0:
        raise FileNotFoundError(f"No files found in {path} with pattern '{pattern}'")
    if random_seed is not None:
        random.seed(random_seed)
        random.shuffle(results)
    videos = _limit_results(results, limit)
    return initialize_dict(videos)


def find_files(
    path: str = None,
    pattern=["png", "jpg", "jpeg", "gif", "webp", "avif", "tiff"],
    recursive: bool = True,
    limit=20,
    random_seed: int = None,
) -> dict:
    """Find image files on the file system.

    Args:
        path (str, optional): The base directory where we are looking for the images. Defaults
            to None, which uses the ammico data directory if set or the current
            working directory otherwise.
        pattern (str|list, optional): The naming pattern that the filename should match.
                Use either '.ext' or just 'ext'
                Defaults to ["png", "jpg", "jpeg", "gif", "webp", "avif","tiff"]. Can be used to allow other patterns or to only include
                specific prefixes or suffixes.
        recursive (bool, optional): Whether to recurse into subdirectories. Default is set to True.
        limit (int/list, optional): The maximum number of images to be found.
            Provide a list or tuple of length 2 to batch the images.
            Defaults to 20. To return all images, set to None or -1.
        random_seed (int, optional): The random seed to use for shuffling the images.
            If None is provided the data will not be shuffeled. Defaults to None.
    Returns:
        dict: A nested dictionary with file ids and all filenames including the path.
    """

    if path is None:
        path = os.environ.get("AMMICO_DATA_HOME", ".")

    if isinstance(pattern, str):
        pattern = [pattern]
    results = []
    for p in pattern:
        results.extend(_match_pattern(path, p, recursive=recursive))

    if len(results) == 0:
        raise FileNotFoundError(f"No files found in {path} with pattern '{pattern}'")

    if random_seed is not None:
        random.seed(random_seed)
        random.shuffle(results)

    images = _limit_results(results, limit)

    return initialize_dict(images)


def initialize_dict(filelist: list) -> dict:
    """Initialize the nested dictionary for all the found images.

    Args:
        filelist (list): The list of files to be analyzed, including their paths.
    Returns:
        dict: The nested dictionary with all image ids and their paths."""
    mydict = {}
    for img_path in filelist:
        id_ = os.path.splitext(os.path.basename(img_path))[0]
        mydict[id_] = {"filename": img_path}
    return mydict


def _check_for_missing_keys(mydict: dict) -> dict:
    """Check the nested dictionary for any missing keys in the subdicts.

    Args:
        mydict(dict): The nested dictionary with keys to check.
    Returns:
        dict: The dictionary with keys appended."""
    # check that we actually got a nested dict
    # also get all keys for all items
    # currently we go through the whole dictionary twice
    # however, compared to the rest of the code this is negligible
    keylist = []
    for key in mydict.keys():
        if not isinstance(mydict[key], dict):
            raise ValueError(
                "Please provide a nested dictionary - you provided {}".format(key)
            )
        keylist.append(list(mydict[key].keys()))
    # find the longest list of keys
    max_keys = max(keylist, key=len)
    # now generate missing keys
    for key in mydict.keys():
        for mkey in max_keys:
            if mkey not in mydict[key].keys():
                mydict[key][mkey] = None
    return mydict


def append_data_to_dict(mydict: dict) -> dict:
    """Append entries from nested dictionaries to keys in a global dict."""

    # first initialize empty list for each key that is present
    outdict = {key: [] for key in list(mydict.values())[0].keys()}
    # now append the values to each key in a list
    for subdict in mydict.values():
        for key in subdict.keys():
            outdict[key].append(subdict[key])
    return outdict


def dump_df(mydict: dict) -> DataFrame:
    """Utility to dump the dictionary into a dataframe."""
    return DataFrame.from_dict(mydict)


def get_dataframe(mydict: dict) -> DataFrame:
    _check_for_missing_keys(mydict)
    outdict = append_data_to_dict(mydict)
    return dump_df(outdict)


def is_interactive():
    """Check if we are running in an interactive environment."""
    import __main__ as main

    return not hasattr(main, "__file__")


def get_color_table():
    path_tables = pkg / "data" / "Color_tables.csv"
    df_colors = read_csv(
        path_tables,
        delimiter=";",
        dtype=str,
        encoding="UTF-8",
        header=[0, 1],
    )
    return {
        col_key: df_colors[col_key].dropna().to_dict("list")
        for col_key in df_colors.columns.levels[0]
    }


def get_supported_whisperx_languages() -> List[str]:
    """Get the list of supported whisperx languages."""
    supported_languages = set(DEFAULT_ALIGN_MODELS_TORCH.keys()) | set(
        DEFAULT_ALIGN_MODELS_HF.keys()
    )
    return sorted(supported_languages)
