import glob
import os
from pandas import DataFrame, read_csv
import pooch
import importlib_resources


pkg = importlib_resources.files("ammico")


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


def find_files(
    path: str = None, pattern: str = "*.png", recursive: bool = True, limit: int = 20
) -> list:
    """Find image files on the file system.

    Args:
        path (str, optional): The base directory where we are looking for the images. Defaults
            to None, which uses the XDG data directory if set or the current
            working directory otherwise.
        pattern (str, optional): The naming pattern that the filename should match. Defaults to
            "*.png". Can be used to allow other patterns or to only include
            specific prefixes or suffixes.
        recursive (bool, optional): Whether to recurse into subdirectories. Default is set to False.
        limit (int, optional): The maximum number of images to be found.
            Defaults to 20. To return all images, set to None.

    Returns:
        list: A list with all filenames including the path.
    """
    if path is None:
        path = os.environ.get("XDG_DATA_HOME", ".")
    result = list(glob.glob(f"{path}/{pattern}", recursive=recursive))
    if limit is not None:
        result = result[:limit]

    if len(result) == 0:
        raise FileNotFoundError(f"No files found in {path} with pattern '{pattern}'")

    return result


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
