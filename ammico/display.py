import ammico.faces as faces
import ammico.text as text
import ammico.objects as objects
from ammico.utils import is_interactive
import ammico.summary as summary
import dash_renderjson
from dash import html, Input, Output, dcc, State
import jupyter_dash
from PIL import Image


class AnalysisExplorer:
    def __init__(self, mydict: dict, identify: str = "faces") -> None:
        """Initialize the AnalysisExplorer class to create an interactive
        visualization of the analysis results.

        Args:
            mydict (dict): A nested dictionary containing image data for all images.
            identify (str, optional): The type of analysis to perform (default: "faces").
                Options are "faces" (face and emotion detection), "text-on-image" (image
                extraction and analysis), "objects" (object detection), "summary" (image caption
                generation).
        """
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

        #  Setup the layout
        app_layout = html.Div(
            [
                # Top
                html.Div(
                    ["Identify: ", identify, self._top_file_explorer(mydict)],
                    id="Div_top",
                    style={
                        "width": "30%",
                    },
                ),
                # Middle
                html.Div(
                    [self._middle_picture_frame()],
                    id="Div_middle",
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                    },
                ),
                # Right
                html.Div(
                    [self._right_output_json()],
                    id="Div_right",
                    style={
                        "width": "45%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                    },
                ),
            ],
            style={"width": "95%", "display": "inline-block"},
        )
        self.app.layout = app_layout

        # Add callbacks to the app
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
    def _top_file_explorer(self, mydict: dict) -> html.Div:
        """Initialize the file explorer dropdown for selecting the file to be analyzed.

        Args:
            mydict (dict): A dictionary containing image data.

        Returns:
            html.Div: The layout for the file explorer dropdown.
        """
        left_layout = html.Div(
            [
                dcc.Dropdown(
                    options={value["filename"]: key for key, value in mydict.items()},
                    id="left_select_id",
                )
            ]
        )
        return left_layout

    def _middle_picture_frame(self) -> html.Div:
        """Initialize the picture frame to display the image.

        Returns:
            html.Div: The layout for the picture frame.
        """
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

    def _right_output_json(self) -> html.Div:
        """Initialize the JSON viewer for displaying the analysis output.

        Returns:
            html.Div: The layout for the JSON viewer.
        """
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

    def run_server(self, port: int = 8050) -> None:
        """Run the Dash server to start the analysis explorer.

        This method should only be called in an interactive environment like Jupyter notebooks.
        Raises an EnvironmentError if not called in an interactive environment.

        Args:
            port (int, optional): The port number to run the server on (default: 8050).
        """
        if not is_interactive():
            raise EnvironmentError(
                "Dash server should only be called in an interactive environment like Jupyter notebooks."
            )

        self.app.run_server(debug=True, mode="inline", port=port)

    # Dash callbacks
    def update_picture(self, img_path: str):
        """Callback function to update the displayed image.

        Args:
            img_path (str): The path of the selected image.

        Returns:
            Union[PIL.PngImagePlugin, None]: The image object to be displayed
                or None if the image path is

        """
        if img_path is not None:
            image = Image.open(img_path)
            return image
        else:
            return None

    def _right_output_analysis(self, all_options: dict, current_value: str) -> dict:
        """Callback function to perform analysis on the selected image and return the output.

        Args:
            all_options (dict): The available options in the file explorer dropdown.
            current_value (str): The current selected value in the file explorer dropdown.

        Returns:
            dict: The analysis output for the selected image.
        """
        identify_dict = {
            "faces": faces.EmotionDetector,
            "text-on-image": text.TextDetector,
            "objects": objects.ObjectDetector,
            "summary": summary.SummaryDetector,
        }

        # Get image ID from dropdown value, which is the filepath
        image_id = all_options[current_value]

        identify_function = identify_dict[self.identify]

        self.mydict[image_id] = identify_function(self.mydict[image_id]).analyse_image()
        return self.mydict[image_id]
