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


def explore_analysis(mydict, identify="faces"):
    # dictionary mapping the type of analysis to be explored
    identify_dict = {
        "faces": faces.facial_expression_analysis,
        "text-on-image": text.detect_text,
    }
    # create a list containing the image ids for the widget
    # image_paths = [mydict[key]["filename"] for key in mydict.keys()]
    image_ids = [key for key in mydict.keys()]
    # Create an image selector widget
    image_select = ipywidgets.Select(
        options=image_ids, layout=ipywidgets.Layout(width="20%"), rows=20
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
        image_widget.children = (
            ipywidgets.Image.from_file(mydict[image_select.value]["filename"]),
        )

        # This output widget absorbes print statements that are messing with
        # the widget output and cannot be disabled through the API.
        with faces.NocatchOutput():
            mydict[image_select.value] = identify_dict[identify](
                mydict[image_select.value]
            )
        with output:
            display(JSONContainer(mydict[image_select.value]))

    # Register the handler and trigger it immediately
    image_select.observe(switch, names=("value",), type="change")
    switch(None)

    # Show the combined widget
    return ipywidgets.HBox([image_select, image_widget, output])
