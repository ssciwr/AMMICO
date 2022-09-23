import glob
import os
from pandas import DataFrame
import pooch
import pandas as pd


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
        id = img_path.split(".")[0].split("/")[-1]
        mydict[id] = {"filename": img_path}
    return mydict


def append_data_to_dict(mydict: dict) -> dict:
    """Append entries from list of dictionaries to keys in global dict."""

    # first initialize empty list for each key that is present
    outdict = {key: [] for key in list(mydict.values())[0].keys()}
    # now append the values to each key in a list
    for subdict in mydict.values():
        for key in subdict.keys():
            outdict[key].append(subdict[key])
    # mydict = {key: [mydict[key] for mydict in dictlist] for key in dictlist[0]}
    return outdict


def dump_df(mydict: dict) -> DataFrame:
    """Utility to dump the dictionary into a dataframe."""
    return DataFrame.from_dict(mydict)


class LabelManager:
    def __init__(self):
        self.labels_code = None
        self.labels = None
        self.f_labels = None
        self.f_labels_code = None
        self.load()

    def load(self):
        self.labels_code = pd.read_excel(
            "./test/data/EUROPE_APRMAY20_data_variable_labels_coding.xlsx",
            sheet_name="variable_labels_codings",
        )
        self.labels = pd.read_csv(
            "./test/data/Europe_APRMAY20data190722.csv", sep=",", decimal="."
        )

    def filter_from_order(self, orders: list):
        cols = []
        for order in orders:
            col = self.labels_code.iloc[order - 1, 1]
            cols.append(col.lower())

        self.f_labels_code = self.labels_code.loc[
            self.labels_code["order"].isin(orders)
        ]
        self.f_labels = self.labels[cols]

    def gen_dict(self):
        labels_dict = {}
        if self.f_labels is None:
            print("No filtered labels found")
            return labels_dict

        cols = self.f_labels.columns.tolist()
        for index, row in self.f_labels.iterrows():
            row_dict = {}
            for col in cols:
                row_dict[col] = row[col]
            labels_dict[row["pic_id"]] = row_dict

        return labels_dict


if __name__ == "__main__":
    files = find_files(
        path="/home/inga/projects/misinformation-project/misinformation/data/test_no_text/"
    )
    mydict = initialize_dict(files)
    outdict = {}
    outdict = append_data_to_dict(mydict)
    df = dump_df(outdict)
    print(df.head(10))

    # example of LabelManager for loading csv data to dict
    lm = LabelManager()
    lm.filter_from_order([1, 2, 3, 22, 23, 24])
    labels = lm.gen_dict()
    print(labels)
