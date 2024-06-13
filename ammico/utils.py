import glob
import os
from pandas import DataFrame, read_csv
import pooch
import importlib_resources
import collections
import random


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


class AnalysisMethod:
    """Base class to be inherited by all analysis methods."""

    def __init__(self, subdict: dict) -> None:
        self.subdict = subdict
        # define keys that will be set by the analysis

    def set_keys(self):
        raise NotImplementedError()

    def analyse_image(self):
        raise NotImplementedError()


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
