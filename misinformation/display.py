import ipywidgets
from IPython.display import display

import misinformation.faces as faces
import misinformation.text as text


class JSONContainer:
    """Expose a Python dictionary as a JSON document in JupyterLab
    rich display rendering.
    """

    def __init__(self, data={}):
        self._data = data

    def _repr_json_(self):
        return self._data


def explore_analysis(image_paths, identify="faces"):
    # dictionary mapping the type of analysis to be explored
    identify_dict = {
        "faces": faces.facial_expression_analysis,
        "text-on-image": text.detect_text,
    }
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
        with faces.NocatchOutput():
            analysis = identify_dict[identify](image_select.value)
        with output:
            display(JSONContainer(analysis))

    # Register the handler and trigger it immediately
    image_select.observe(switch, names=("value",), type="change")
    switch(None)

    # Show the combined widget
    return ipywidgets.HBox([image_select, image_widget, output])
