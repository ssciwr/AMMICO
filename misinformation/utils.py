import glob
import os
from pandas import DataFrame
import pooch
from lavis.models import load_model_and_preprocess


class DownloadResource:
    """A remote resource that needs on demand downloading

    We use this as a wrapper to the pooch library. The wrapper registers
    each data file and allows prefetching through the CLI entry point
    misinformation_prefetch_models.
    """

    # We store a list of defined resouces in a class variable, allowing
    # us prefetching from a CLI e.g. to bundle into a Docker image
    resources = []

    def __init__(self, **kwargs):
        DownloadResource.resources.append(self)
        self.kwargs = kwargs

    def get(self):
        return pooch.retrieve(**self.kwargs)


def misinformation_prefetch_models():
    """Prefetch all the download resources"""
    for res in DownloadResource.resources:
        res.get()


class AnalysisMethod:
    """Base class to be inherited by all analysis methods."""

    def __init__(self, subdict) -> None:
        self.subdict = subdict
        # define keys that will be set by the analysis

    def set_keys(self):
        raise NotImplementedError()

    def analyse_image(self):
        raise NotImplementedError()


def find_files(path=None, pattern="*.png", recursive=True, limit=20):
    """Find image files on the file system

    :param path:
        The base directory where we are looking for the images. Defaults
        to None, which uses the XDG data directory if set or the current
        working directory otherwise.
    :param pattern:
        The naming pattern that the filename should match. Defaults to
        "*.png". Can be used to allow other patterns or to only include
        specific prefixes or suffixes.
    :param recursive:
        Whether to recurse into subdirectories.
    :param limit:
        The maximum number of images to be found. Defaults to 20.
        To return all images, set to None.
    """
    if path is None:
        path = os.environ.get("XDG_DATA_HOME", ".")

    result = list(glob.glob(f"{path}/{pattern}", recursive=recursive))

    if limit is not None:
        result = result[:limit]

    return result


def initialize_dict(filelist: list) -> dict:
    mydict = {}
    for img_path in filelist:
        id_ = os.path.splitext(os.path.basename(img_path))[0]
        mydict[id_] = {"filename": img_path}
    return mydict


def append_data_to_dict(mydict: dict) -> dict:
    """Append entries from list of dictionaries to keys in global dict."""

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


if __name__ == "__main__":
    files = find_files(
        path="/home/inga/projects/misinformation-project/misinformation/data/test_no_text/"
    )
    mydict = initialize_dict(files)
    outdict = {}
    outdict = append_data_to_dict(mydict)
    df = dump_df(outdict)
    print(df.head(10))
