import glob
import os


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
