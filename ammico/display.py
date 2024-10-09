import ammico.faces as faces
import ammico.text as text
import ammico.colors as colors
from ammico.utils import is_interactive
import ammico.summary as summary
import pandas as pd
from dash import html, Input, Output, dcc, State, Dash
from PIL import Image
import dash_bootstrap_components as dbc


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
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
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
                # Top row, only file explorer
                dbc.Row(
                    [dbc.Col(self._top_file_explorer(mydict))],
                    id="Div_top",
                    style={
                        "width": "30%",
                    },
                ),
                # second row, middle picture and right output
                dbc.Row(
                    [
                        # first column: picture
                        dbc.Col(self._middle_picture_frame()),
                        dbc.Col(self._right_output_json()),
                    ]
                ),
            ],
            # style={"width": "95%", "display": "inline-block"},
        )
        self.app.layout = app_layout

        # Add callbacks to the app
        self.app.callback(
            Output("img_middle_picture_id", "src"),
            Input("left_select_id", "value"),
            prevent_initial_call=True,
        )(self.update_picture)

        self.app.callback(
            Output("right_json_viewer", "children"),
            Input("button_run", "n_clicks"),
            State("left_select_id", "options"),
            State("left_select_id", "value"),
            State("Dropdown_select_Detector", "value"),
            State("setting_Text_analyse_text", "value"),
            State("setting_Text_model_names", "value"),
            State("setting_Text_revision_numbers", "value"),
            State("setting_privacy_env_var", "value"),
            State("setting_Emotion_emotion_threshold", "value"),
            State("setting_Emotion_race_threshold", "value"),
            State("setting_Emotion_gender_threshold", "value"),
            State("setting_Emotion_env_var", "value"),
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
                # text summary start
                html.Div(
                    id="settings_TextDetector",
                    style={"display": "none"},
                    children=[
                        dbc.Row(
                            dcc.Checklist(
                                ["Analyse text"],
                                ["Analyse text"],
                                id="setting_Text_analyse_text",
                                style={"margin-bottom": "10px"},
                            ),
                        ),
                        # row 1
                        dbc.Row(
                            dbc.Col(
                                [
                                    html.P(
                                        "Privacy disclosure acceptance environment variable"
                                    ),
                                    dcc.Input(
                                        type="text",
                                        value="PRIVACY_AMMICO",
                                        id="setting_privacy_env_var",
                                        style={"width": "100%"},
                                    ),
                                ],
                                align="start",
                            ),
                        ),
                        # text row 2
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.P(
                                            "Select models for text_summary, text_sentiment, text_NER or leave blank for default:",
                                            # style={"width": "45%"},
                                        ),
                                    ]
                                ),  #
                                dbc.Col(
                                    [
                                        html.P(
                                            "Select model revision number for text_summary, text_sentiment, text_NER or leave blank for default:"
                                        ),
                                    ]
                                ),
                            ]
                        ),  # row 2
                        #  input row 3
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Input(
                                        type="text",
                                        id="setting_Text_model_names",
                                        style={"width": "100%"},
                                    ),
                                ),
                                dbc.Col(
                                    dcc.Input(
                                        type="text",
                                        id="setting_Text_revision_numbers",
                                        style={"width": "100%"},
                                    ),
                                ),
                            ]
                        ),  # row 3
                    ],
                ),  # text summary end
                # start emotion detector
                html.Div(
                    id="settings_EmotionDetector",
                    style={"display": "none"},
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.P("Emotion threshold"),
                                        dcc.Input(
                                            value=50,
                                            type="number",
                                            max=100,
                                            min=0,
                                            id="setting_Emotion_emotion_threshold",
                                            style={"width": "100%"},
                                        ),
                                    ],
                                    align="start",
                                ),
                                dbc.Col(
                                    [
                                        html.P("Race threshold"),
                                        dcc.Input(
                                            type="number",
                                            value=50,
                                            max=100,
                                            min=0,
                                            id="setting_Emotion_race_threshold",
                                            style={"width": "100%"},
                                        ),
                                    ],
                                    align="start",
                                ),
                                dbc.Col(
                                    [
                                        html.P("Gender threshold"),
                                        dcc.Input(
                                            type="number",
                                            value=50,
                                            max=100,
                                            min=0,
                                            id="setting_Emotion_gender_threshold",
                                            style={"width": "100%"},
                                        ),
                                    ],
                                    align="start",
                                ),
                                dbc.Col(
                                    [
                                        html.P(
                                            "Disclosure acceptance environment variable"
                                        ),
                                        dcc.Input(
                                            type="text",
                                            value="DISCLOSURE_AMMICO",
                                            id="setting_Emotion_env_var",
                                            style={"width": "100%"},
                                        ),
                                    ],
                                    align="start",
                                ),
                            ],
                            style={"width": "100%"},
                        ),
                    ],
                ),  # end emotion detector
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
                        dbc.Col(
                            [
                                dbc.Row([html.P("Analysis type:")]),
                                dbc.Row([html.P("Model type:")]),
                                dbc.Row([html.P("Analysis question:")]),
                            ],
                        ),
                        dbc.Col(
                            [
                                dbc.Row(
                                    dcc.Dropdown(
                                        options=SUMMARY_ANALYSIS_TYPE,
                                        value="summary_and_questions",
                                        id="setting_Summary_analysis_type",
                                    )
                                ),
                                dbc.Row(
                                    dcc.Dropdown(
                                        options=SUMMARY_MODEL,
                                        value="base",
                                        id="setting_Summary_model",
                                    )
                                ),
                                dbc.Row(
                                    dcc.Input(
                                        type="text",
                                        id="setting_Summary_list_of_questions",
                                        style={
                                            "height": "auto",
                                            "margin-left": "11px",
                                        },
                                    ),
                                ),
                            ]
                        ),
                    ],
                ),
            ],
            style={"width": "100%", "display": "inline-block"},
        )
        return settings_layout

    def _right_output_json(self) -> html.Div:
        """Initialize the DetectorDropdown, argument Div and JSON viewer for displaying the analysis output.

        Returns:
            html.Div: The layout for the JSON viewer.
        """
        right_layout = html.Div(
            [
                dbc.Col(
                    [
                        dbc.Row(
                            dcc.Dropdown(
                                options=[
                                    "TextDetector",
                                    "EmotionDetector",
                                    "SummaryDetector",
                                    "ColorDetector",
                                ],
                                value="TextDetector",
                                id="Dropdown_select_Detector",
                                style={"width": "60%"},
                            ),
                            justify="start",
                        ),
                        dbc.Row(
                            children=[self._create_setting_layout()],
                            id="div_detector_args",
                            justify="start",
                        ),
                        dbc.Row(
                            html.Button(
                                "Run Detector",
                                id="button_run",
                                style={
                                    "margin-top": "15px",
                                    "margin-bottom": "15px",
                                    "margin-left": "11px",
                                    "width": "30%",
                                },
                            ),
                            justify="start",
                        ),
                        dbc.Row(
                            dcc.Loading(
                                id="loading-2",
                                children=[
                                    # This is where the json is shown.
                                    html.Div(id="right_json_viewer"),
                                ],
                                type="circle",
                            ),
                            justify="start",
                        ),
                    ],
                    align="start",
                )
            ]
        )
        return right_layout

    def run_server(self, port: int = 8050) -> None:
        """Run the Dash server to start the analysis explorer.


        Args:
            port (int, optional): The port number to run the server on (default: 8050).
        """

        self.app.run_server(debug=True, port=port)

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
        settings_text_analyse_text: list,
        settings_text_model_names: str,
        settings_text_revision_numbers: str,
        setting_privacy_env_var: str,
        setting_emotion_emotion_threshold: int,
        setting_emotion_race_threshold: int,
        setting_emotion_gender_threshold: int,
        setting_emotion_env_var: str,
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
            "SummaryDetector": summary.SummaryDetector,
            "ColorDetector": colors.ColorDetector,
        }

        # Get image ID from dropdown value, which is the filepath
        if current_img_value is None:
            return {}
        image_id = all_img_options[current_img_value]
        # copy image so prvious runs don't leave their default values in the dict
        image_copy = self.mydict[image_id].copy()

        # detector value is the string name of the chosen detector
        identify_function = identify_dict[detector_value]

        if detector_value == "TextDetector":
            analyse_text = (
                True if settings_text_analyse_text == ["Analyse text"] else False
            )
            detector_class = identify_function(
                image_copy,
                analyse_text=analyse_text,
                model_names=(
                    [settings_text_model_names]
                    if (settings_text_model_names is not None)
                    else None
                ),
                revision_numbers=(
                    [settings_text_revision_numbers]
                    if (settings_text_revision_numbers is not None)
                    else None
                ),
                accept_privacy=(
                    setting_privacy_env_var
                    if setting_privacy_env_var
                    else "PRIVACY_AMMICO"
                ),
            )
        elif detector_value == "EmotionDetector":
            detector_class = identify_function(
                image_copy,
                emotion_threshold=setting_emotion_emotion_threshold,
                race_threshold=setting_emotion_race_threshold,
                gender_threshold=setting_emotion_gender_threshold,
                accept_disclosure=(
                    setting_emotion_env_var
                    if setting_emotion_env_var
                    else "DISCLOSURE_AMMICO"
                ),
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
                model_type=setting_summary_model,
                list_of_questions=(
                    [setting_summary_list_of_questions]
                    if (setting_summary_list_of_questions is not None)
                    else None
                ),
            )
        else:
            detector_class = identify_function(image_copy)
        analysis_dict = detector_class.analyse_image()

        # Initialize an empty dictionary
        new_analysis_dict = {}

        # Iterate over the items in the original dictionary
        for k, v in analysis_dict.items():
            # Check if the value is a list
            if isinstance(v, list):
                # If it is, convert each item in the list to a string and join them with a comma
                new_value = ", ".join([str(f) for f in v])
            else:
                # If it's not a list, keep the value as it is
                new_value = v

            # Add the new key-value pair to the new dictionary
            new_analysis_dict[k] = new_value

        df = pd.DataFrame([new_analysis_dict]).set_index("filename").T
        df.index.rename("filename", inplace=True)
        return dbc.Table.from_dataframe(
            df, striped=True, bordered=True, hover=True, index=True
        )
