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
    # Create an image selector widget
    image_select = ipywidgets.Select(
        options=image_paths, layout=ipywidgets.Layout(width="20%"), rows=20
    )

    # Set up the facial recognition output widget
    output = ipywidgets.Output(layout=ipywidgets.Layout(width="30%"))

    # Set up the image selection and display widget
    image_widget = ipywidgets.Box(
        children=[],
        layout=ipywidgets.Layout(width="50%"),
    )

    # Register the tab switch logic
    def switch(_):
        # Clear existing output
        image_widget.children = ()
        output.clear_output()

        # Create the new content
        image_widget.children = (ipywidgets.Image.from_file(image_select.value),)

        # This output widget absorbes print statements that are messing with
        # the widget output and cannot be disabled through the API.
        with ipywidgets.Output():
            analysis = facial_expression_analysis(image_select.value)
        with output:
            display(JSONContainer(analysis))

    # Register the handler and trigger it immediately
    image_select.observe(switch, names=("value",), type="change")
    switch(None)

    # Show the combined widget
    return ipywidgets.HBox([image_select, image_widget, output])
