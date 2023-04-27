import ipywidgets
from IPython.display import display

import misinformation.faces as faces
import misinformation.text as text
import misinformation.objects as objects

import misinformation.summary as summary

import dash_renderjson
import dash
from dash import html, Input, Output, dcc, State
import jupyter_dash
from PIL import Image


class JSONContainer:
    """Expose a Python dictionary as a JSON document in JupyterLab
    rich display rendering.
    """

    def __init__(self, data=None):
        if data is None:
            data = {}
        self._data = data

    def _repr_json_(self):
        return self._data


def explore_analysis_dash(mydict, identify="faces"):
    app = jupyter_dash.JupyterDash(__name__)

    theme = {
        "scheme": "monokai",
        "author": "wimer hazenberg (http://www.monokai.nl)",
        "base00": "#272822",
        "base01": "#383830",
        "base02": "#49483e",
        "base03": "#75715e",
        "base04": "#a59f85",
        "base05": "#f8f8f2",
        "base06": "#f5f4f1",
        "base07": "#f9f8f5",
        "base08": "#f92672",
        "base09": "#fd971f",
        "base0A": "#f4bf75",
        "base0B": "#a6e22e",
        "base0C": "#a1efe4",
        "base0D": "#66d9ef",
        "base0E": "#ae81ff",
        "base0F": "#cc6633",
    }

    def _top_file_explorer(mydict):
        left_layout = html.Div(
            [
                dcc.Dropdown(
                    options={value["filename"]: key for key, value in mydict.items()},
                    id="left_select_id",
                )
            ]
        )
        return left_layout

    def _middle_picture_frame():
        middle_layout = html.Div(
            [
                html.Img(
                    id="img_middle_picture_id",
                    style={
                        "width": "80%",
                    },
                )
            ]
        )
        return middle_layout

    def _right_output_json():
        right_layout = html.Div(
            [
                dcc.Loading(
                    id="loading-2",
                    children=[
                        html.Div(
                            [
                                dash_renderjson.DashRenderjson(
                                    id="right_json_viewer",
                                    data={"a": "1"},
                                    max_depth=-1,
                                    theme=theme,
                                    invert_theme=True,
                                )
                            ]
                        )
                    ],
                    type="circle",
                )
            ]
        )
        return right_layout

    @app.callback(
        Output("img_middle_picture_id", "src"),
        Input("left_select_id", "value"),
        prevent_initial_call=True,
    )
    def update_picture(img_path):
        if img_path is not None:
            image = Image.open(img_path)
            return image
        else:
            return None

    @app.callback(
        Output("right_json_viewer", "data"),
        Input("img_middle_picture_id", "src"),
        State("Div_top", "children"),
        State("left_select_id", "options"),
        State("left_select_id", "value"),
        prevent_initial_call=True,
    )
    def _right_output_analysis(image, div_top, all_options, current_value):
        identify_dict = {
            "faces": faces.EmotionDetector,
            "text-on-image": text.TextDetector,
            "objects": objects.ObjectDetector,
            "summary": summary.SummaryDetector,
        }
        image_id = all_options[current_value]
        identify_function = identify_dict[div_top[1]]

        mydict[image_id] = identify_function(mydict[image_id]).analyse_image()
        return mydict[image_id]

    app_layout = html.Div(
        [
            # top
            html.Div(
                ["Identify: ", identify, _top_file_explorer(mydict)],
                id="Div_top",
                style={
                    "width": "20%",
                    # "display": "inline-block",
                },
            ),
            # middle
            html.Div(
                ["middle", _middle_picture_frame()],
                id="Div_middle",
                style={
                    "width": "40%",
                    "display": "inline-block",
                    "verticalAlign": "top",
                },
            ),
            # right
            html.Div(
                ["right", _right_output_json()],
                id="Div_right",
                style={
                    "width": "30%",
                    "display": "inline-block",
                    "verticalAlign": "top",
                },
            ),
        ],
        style={"width": "80%", "display": "inline-block"},
    )
    app.layout = app_layout
    # app.layout = html.Div([dash_renderjson.DashRenderjson(id="input", data=data, max_depth=-1, theme=theme, invert_theme=True)])

    return app


def explore_analysis(mydict, identify="faces"):
    # dictionary mapping the type of analysis to be explored
    identify_dict = {
        "faces": faces.EmotionDetector,
        "text-on-image": text.TextDetector,
        "objects": objects.ObjectDetector,
        "summary": summary.SummaryDetector,
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
            ).analyse_image()
        with output:
            display(JSONContainer(mydict[image_select.value]))

    # Register the handler and trigger it immediately
    image_select.observe(switch, names=("value",), type="change")
    switch(None)

    # Show the combined widget
    return ipywidgets.HBox([image_select, image_widget, output])
