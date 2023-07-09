import ammico.faces as faces
import ammico.text as text
import ammico.objects as objects
import ammico.colors as colors
from ammico.utils import is_interactive
import ammico.summary as summary
import dash_renderjson
from dash import html, Input, Output, dcc, State
import jupyter_dash
from PIL import Image


COLOR_SCHEMES = [
    "CIE 1976",
    "CIE 1994",
    "CIE 2000",
    "CMC",
    "ITP",
    "CAM02-LCD",
    "CAM02-SCD",
    "CAM02-UCS",
    "CAM16-LCD",
    "CAM16-SCD",
    "CAM16-UCS",
    "DIN99",
]
SUMMARY_ANALYSIS_TYPE = ["summary_and_questions", "summary", "questions"]
SUMMARY_MODEL = ["base", "large"]


class AnalysisExplorer:
    def __init__(self, mydict: dict) -> None:
        """Initialize the AnalysisExplorer class to create an interactive
        visualization of the analysis results.

        Args:
            mydict (dict): A nested dictionary containing image data for all images.

        """
        self.app = jupyter_dash.JupyterDash(__name__)
        self.mydict = mydict
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

        # Setup the layout
        app_layout = html.Div(
            [
                # Top
                html.Div(
                    [self._top_file_explorer(mydict)],
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
            Input("button_run", "n_clicks"),
            State("left_select_id", "options"),
            State("left_select_id", "value"),
            State("Dropdown_select_Detector", "value"),
            State("setting_Text_analyse_text", "value"),
            State("setting_Text_model_names", "value"),
            State("setting_Text_revision_numbers", "value"),
            State("setting_Emotion_emotion_threshold", "value"),
            State("setting_Emotion_race_threshold", "value"),
            State("setting_Color_delta_e_method", "value"),
            State("setting_Summary_analysis_type", "value"),
            State("setting_Summary_model", "value"),
            State("setting_Summary_list_of_questions", "value"),
            prevent_initial_call=True,
        )(self._right_output_analysis)

        self.app.callback(
            Output("settings_TextDetector", "style"),
            Output("settings_EmotionDetector", "style"),
            Output("settings_ColorDetector", "style"),
            Output("settings_Summary_Detector", "style"),
            Input("Dropdown_select_Detector", "value"),
        )(self._update_detector_setting)

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

    def _create_setting_layout(self):
        settings_layout = html.Div(
            [
                html.Div(
                    id="settings_TextDetector",
                    style={"display": "none"},
                    children=[
                        dcc.Checklist(
                            ["Analyse text"],
                            ["Analyse text"],
                            id="setting_Text_analyse_text",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    "Select models for text_summary, text_sentiment, text_NER or leave blank for default:",
                                    style={
                                        "height": "30px",
                                        "margin-top": "5px",
                                    },
                                ),
                                dcc.Input(
                                    type="text",
                                    id="setting_Text_model_names",
                                    style={"height": "auto", "margin-bottom": "auto"},
                                ),
                            ],
                            style={
                                "width": "33%",
                                "display": "inline-block",
                                "margin-top": "10px",
                            },
                        ),
                        html.Div(
                            [
                                html.Div(
                                    "Select model revision number for text_summary, text_sentiment, text_NER or leave blank for default:",
                                    style={
                                        "height": "30px",
                                        "margin-top": "5px",
                                    },
                                ),
                                dcc.Input(
                                    type="text",
                                    id="setting_Text_revision_numbers",
                                    style={"height": "auto", "margin-bottom": "auto"},
                                ),
                            ],
                            style={
                                "width": "33%",
                                "display": "inline-block",
                                "margin-top": "10px",
                            },
                        ),
                    ],
                ),
                html.Div(
                    id="settings_EmotionDetector",
                    style={"display": "none"},
                    children=[
                        html.Div(
                            [
                                html.Div(
                                    "Emotion threshold",
                                    style={"height": "30px", "margin-top": "5px"},
                                ),
                                dcc.Input(
                                    value=50,
                                    type="number",
                                    max=100,
                                    min=0,
                                    id="setting_Emotion_emotion_threshold",
                                    style={"height": "auto", "margin-bottom": "auto"},
                                ),
                            ],
                            style={"width": "49%", "display": "inline-block"},
                        ),
                        html.Div(
                            [
                                html.Div(
                                    "Race threshold",
                                    style={
                                        "height": "30px",
                                        "margin-top": "5px",
                                    },
                                ),
                                dcc.Input(
                                    type="number",
                                    value=50,
                                    max=100,
                                    min=0,
                                    id="setting_Emotion_race_threshold",
                                    style={"height": "auto", "margin-bottom": "auto"},
                                ),
                            ],
                            style={
                                "width": "49%",
                                "display": "inline-block",
                                "margin-top": "10px",
                            },
                        ),
                    ],
                ),
                html.Div(
                    id="settings_ColorDetector",
                    style={"display": "none"},
                    children=[
                        html.Div(
                            [
                                dcc.Dropdown(
                                    options=COLOR_SCHEMES,
                                    value="CIE 1976",
                                    id="setting_Color_delta_e_method",
                                )
                            ],
                            style={
                                "width": "49%",
                                "display": "inline-block",
                                "margin-top": "10px",
                            },
                        )
                    ],
                ),
                html.Div(
                    id="settings_Summary_Detector",
                    style={"display": "none"},
                    children=[
                        html.Div(
                            [
                                dcc.Dropdown(
                                    options=SUMMARY_ANALYSIS_TYPE,
                                    value="summary_and_questions",
                                    id="setting_Summary_analysis_type",
                                )
                            ],
                            style={
                                "width": "33%",
                                "display": "inline-block",
                            },
                        ),
                        html.Div(
                            [
                                dcc.Dropdown(
                                    options=SUMMARY_MODEL,
                                    value="base",
                                    id="setting_Summary_model",
                                )
                            ],
                            style={
                                "width": "33%",
                                "display": "inline-block",
                                "margin-top": "10px",
                            },
                        ),
                        html.Div(
                            [
                                html.Div(
                                    "Please enter a question",
                                    style={
                                        "height": "50px",
                                        "margin-top": "5px",
                                    },
                                ),
                                dcc.Input(
                                    type="text",
                                    id="setting_Summary_list_of_questions",
                                    style={"height": "auto", "margin-bottom": "auto"},
                                ),
                            ],
                            style={
                                "width": "33%",
                                "display": "inline-block",
                                "margin-top": "10px",
                            },
                        ),
                    ],
                ),
            ],
        )
        return settings_layout

    def _right_output_json(self) -> html.Div:
        """Initialize the DetectorDropdown, argument Div and JSON viewer for displaying the analysis output.

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
                                dcc.Dropdown(
                                    options=[
                                        "TextDetector",
                                        "ObjectDetector",
                                        "EmotionDetector",
                                        "SummaryDetector",
                                        "ColorDetector",
                                    ],
                                    value="TextDetector",
                                    id="Dropdown_select_Detector",
                                ),
                                html.Div(
                                    children=[self._create_setting_layout()],
                                    id="div_detector_args",
                                ),
                                html.Button("Run Detector", id="button_run"),
                                dash_renderjson.DashRenderjson(
                                    id="right_json_viewer",
                                    data={},
                                    max_depth=-1,
                                    theme=self.theme,
                                    invert_theme=True,
                                ),
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

    def _update_detector_setting(self, setting_input):
        # return settings_TextDetector -> style, settings_EmotionDetector -> style
        display_none = {"display": "none"}
        display_flex = {
            "display": "flex",
            "flexWrap": "wrap",
            "width": 400,
            "margin-top": "20px",
        }

        if setting_input == "TextDetector":
            return display_flex, display_none, display_none, display_none

        if setting_input == "EmotionDetector":
            return display_none, display_flex, display_none, display_none

        if setting_input == "ColorDetector":
            return display_none, display_none, display_flex, display_none

        if setting_input == "SummaryDetector":
            return display_none, display_none, display_none, display_flex

        else:
            return display_none, display_none, display_none, display_none

    def _right_output_analysis(
        self,
        n_clicks,
        all_img_options: dict,
        current_img_value: str,
        detector_value: str,
        settings_text_analyse_text: bool,
        settings_text_model_names: str,
        settings_text_revision_numbers: str,
        setting_emotion_emotion_threshold: int,
        setting_emotion_race_threshold: int,
        setting_color_delta_e_method: str,
        setting_summary_analysis_type: str,
        setting_summary_model: str,
        setting_summary_list_of_questions: str,
    ) -> dict:
        """Callback function to perform analysis on the selected image and return the output.

        Args:
            all_options (dict): The available options in the file explorer dropdown.
            current_value (str): The current selected value in the file explorer dropdown.

        Returns:
            dict: The analysis output for the selected image.
        """
        identify_dict = {
            "EmotionDetector": faces.EmotionDetector,
            "TextDetector": text.TextDetector,
            "ObjectDetector": objects.ObjectDetector,
            "SummaryDetector": summary.SummaryDetector,
            "ColorDetector": colors.ColorDetector,
        }

        # Get image ID from dropdown value, which is the filepath
        if current_img_value is None:
            return {}
        image_id = all_img_options[current_img_value]
        # copy image so prvious runs don't leave their default values in the dict
        image_copy = self.mydict[image_id].copy()

        identify_function = identify_dict[detector_value]
        if detector_value == "TextDetector":
            detector_class = identify_function(
                image_copy,
                analyse_text=settings_text_analyse_text,
                model_names=[settings_text_model_names]
                if (settings_text_model_names is not None)
                else None,
                revision_numbers=[settings_text_revision_numbers]
                if (settings_text_revision_numbers is not None)
                else None,
            )
        elif detector_value == "EmotionDetector":
            detector_class = identify_function(
                image_copy,
                race_threshold=setting_emotion_race_threshold,
                emotion_threshold=setting_emotion_emotion_threshold,
            )
        elif detector_value == "ColorDetector":
            detector_class = identify_function(
                image_copy,
                delta_e_method=setting_color_delta_e_method,
            )
        elif detector_value == "SummaryDetector":
            detector_class = identify_function(
                image_copy,
                analysis_type=setting_summary_analysis_type,
                summary_model_type=setting_summary_model,
                list_of_questions=[setting_summary_list_of_questions]
                if (setting_summary_list_of_questions is not None)
                else None,
            )
        else:
            detector_class = identify_function(image_copy)
        return detector_class.analyse_image()
