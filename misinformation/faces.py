import ipywidgets

from IPython.display import display
from deepface import DeepFace
from retinaface import RetinaFace


def facial_expression_analysis(img_path):
    result = {"filename": img_path}

    # Find (multiple) faces in the image and cut them
    faces = RetinaFace.extract_faces(img_path)

    # If no faces are found, we return an empty dictionary
    if len(faces) == 0:
        return result

    # Find the biggest face image in the detected ones
    maxface = max(faces, key=lambda f: f.shape[0] * f.shape[1])

    # Run the full DeepFace analysis
    result["deepface_results"] = DeepFace.analyze(
        img_path=maxface,
        actions=["age", "gender", "race", "emotion"],
        prog_bar=False,
        detector_backend="skip",
    )

    # We remove the region, as the data is not correct - after all we are
    # running the analysis on a subimage.
    del result["deepface_results"]["region"]

    return result


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

    # Precompute all the results for a user experience without delay
    with ipywidgets.Output():
        results = [facial_expression_analysis(i) for i in image_paths]

    # Register the tab switch logic
    def tabswitch(_):
        output.clear_output()
        with output:
            display(JSONContainer(results[image_widget.selected_index]))

    # Register the handler and trigger it immediately
    image_widget.observe(tabswitch, names=("selected_index",), type="change")
    tabswitch(None)

    # Show the combined widget
    return ipywidgets.HBox([image_widget, output])
