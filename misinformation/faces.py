import cv2
import ipywidgets
import numpy as np
import pooch

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
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

    # Determine whether the face wears a mask
    result["wears_mask"] = wears_mask(maxface)

    # Adapt the features we are looking for depending on whether a mask is
    # worn. White masks screw race detection, emotion detection is useless.
    actions = ["age", "gender"]
    if not result["wears_mask"]:
        actions = actions + ["race", "emotion"]

    # Run the full DeepFace analysis
    result["deepface_results"] = DeepFace.analyze(
        img_path=maxface,
        actions=actions,
        prog_bar=False,
        detector_backend="skip",
    )

    # We remove the region, as the data is not correct - after all we are
    # running the analysis on a subimage.
    del result["deepface_results"]["region"]

    return result


def wears_mask(face):
    global mask_detection_model

    # Preprocess the face to match the assumptions of the face mask
    # detection model
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)

    # Lazily load the model
    model_path = pooch.retrieve(
        url="https://github.com/chandrikadeb7/Face-Mask-Detection/raw/v1.0.0/mask_detector.model",
        known_hash="sha256:d0b30e2c7f8f187c143d655dee8697fcfbe8678889565670cd7314fb064eadc8",
    )
    mask_detection_model = load_model(model_path)

    # Run the model (ignoring output)
    with NocatchOutput():
        mask, withoutMask = mask_detection_model.predict(face)[0]

    # Convert from np.bool_ to bool to later be able to serialize the result
    return bool(mask > withoutMask)


class JSONContainer:
    """Expose a Python dictionary as a JSON document in JupyterLab
    rich display rendering.
    """

    def __init__(self, data={}):
        self._data = data

    def _repr_json_(self):
        return self._data


class NocatchOutput(ipywidgets.Output):
    """An output container that suppresses output, but not exceptions

    Taken from https://github.com/jupyter-widgets/ipywidgets/issues/3208#issuecomment-1070836153
    """

    def __exit__(self, *args, **kwargs):
        super().__exit__(*args, **kwargs)


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
        with NocatchOutput():
            analysis = facial_expression_analysis(image_select.value)
        with output:
            display(JSONContainer(analysis))

    # Register the handler and trigger it immediately
    image_select.observe(switch, names=("value",), type="change")
    switch(None)

    # Show the combined widget
    return ipywidgets.HBox([image_select, image_widget, output])
