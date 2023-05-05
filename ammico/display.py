from IPython.display import display

import ammico.faces as faces
import ammico.text as text
import ammico.objects as objects
from ammico.utils import is_interactive

import ammico.summary as summary

import dash_renderjson
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


class AnalysisExplorer:
    def __init__(self, mydict, identify="faces") -> None:
        self.app = jupyter_dash.JupyterDash(__name__)
        self.mydict = mydict
        self.identify = identify
        self.theme = {
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

        # setup the layout
        app_layout = html.Div(
            [
                # top
                html.Div(
                    ["Identify: ", identify, self._top_file_explorer(mydict)],
                    id="Div_top",
                    style={
                        "width": "30%",
                        # "display": "inline-block",
                    },
                ),
                # middle
                html.Div(
                    [self._middle_picture_frame()],
                    id="Div_middle",
                    style={
                        "width": "60%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                    },
                ),
                # right
                html.Div(
                    [self._right_output_json()],
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
        self.app.layout = app_layout
        # add callbacks to app
        self.app.callback(
            Output("img_middle_picture_id", "src"),
            Input("left_select_id", "value"),
            prevent_initial_call=True,
        )(self.update_picture)

        self.app.callback(
            Output("right_json_viewer", "data"),
            Input("img_middle_picture_id", "src"),
            State("left_select_id", "options"),
            State("left_select_id", "value"),
            prevent_initial_call=True,
        )(self._right_output_analysis)

    # I split the different sections into subfunctions for better clarity
    def _top_file_explorer(self, mydict):
        # initilizes the dropdown that selects which file is to be analyzed.
        left_layout = html.Div(
            [
                dcc.Dropdown(
                    options={value["filename"]: key for key, value in mydict.items()},
                    id="left_select_id",
                )
            ]
        )
        return left_layout

    def _middle_picture_frame(self):
        # This just holds the image
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

    def _right_output_json(self):
        # provides the json viewer for the analysis output.
        right_layout = html.Div(
            [
                dcc.Loading(
                    id="loading-2",
                    children=[
                        html.Div(
                            [
                                dash_renderjson.DashRenderjson(
                                    id="right_json_viewer",
                                    data={},
                                    max_depth=-1,
                                    theme=self.theme,
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

    def run_server(self, port=8050):
        if not is_interactive():
            raise EnvironmentError(
                "Dash server should only be called in interactive an interactive environment like jupyter notebooks."
            )

        self.app.run_server(debug=True, mode="inline", port=port)

    # Dash callbacks
    def update_picture(self, img_path):
        if img_path is not None:
            image = Image.open(img_path)
            return image
        else:
            return None

    def _right_output_analysis(self, image, all_options, current_value):
        # calls the analysis function and returns the output
        identify_dict = {
            "faces": faces.EmotionDetector,
            "text-on-image": text.TextDetector,
            "objects": objects.ObjectDetector,
            "summary": summary.SummaryDetector,
        }
        # get image ID from dropdown value, which is the filepath.
        image_id = all_options[current_value]

        identify_function = identify_dict[self.identify]

        self.mydict[image_id] = identify_function(self.mydict[image_id]).analyse_image()
        return self.mydict[image_id]
