import glob
import ipywidgets
import os

from IPython.display import display
from deepface import DeepFace


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


def facial_expression_analysis(img_path):
    return DeepFace.analyze(
        img_path=img_path, actions=["age", "gender", "race", "emotion"], prog_bar=False
    )


class JSONContainer:
    """Expose a Python dictionary as a JSON document in JupyterLab
    rich display rendering.
    """

    def __init__(self, data={}):
        self._data = data

    def _repr_json_(self):
        return self._data


def explore_face_recognition(image_paths):
    # Set up the facial recognition output widget
    output = ipywidgets.Output(layout=ipywidgets.Layout(width="30%"))

    # Set up the image selection and display widget
    images = [ipywidgets.Image.from_file(p) for p in image_paths]
    image_widget = ipywidgets.Tab(
        children=images,
        titles=[f"#{i}" for i in range(len(image_paths))],
        layout=ipywidgets.Layout(width="70%"),
    )

    # Register the facial recognition logic
    def _recognition(_):
        data = {}
        data["filename"] = image_paths[image_widget.selected_index]

        try:
            data["deepface_results"] = facial_expression_analysis(data["filename"])
            data["deepface_find_face"] = True
        except ValueError:
            data["deepface_find_face"] = False

        output.clear_output()
        with output:
            display(JSONContainer(data))

    # Register the handler and trigger it immediately
    image_widget.observe(_recognition, names=("selected_index",), type="change")

    with ipywidgets.Output():
        _recognition(None)

    # Show the combined widget
    return ipywidgets.HBox([image_widget, output])
