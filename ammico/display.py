import ammico.text as text
import ammico.colors as colors
import ammico.image_summary as image_summary
from ammico.model import MultimodalSummaryModel
import pandas as pd
from dash import html, Input, Output, dcc, State, Dash
from PIL import Image
import dash_bootstrap_components as dbc
import warnings
from typing import Dict, Any, List, Optional


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
            State("Dropdown_analysis_type", "value"),
            State("textarea_questions", "value"),
            State("setting_privacy_env_var", "value"),
            State("setting_Color_delta_e_method", "value"),
            prevent_initial_call=True,
        )(self._right_output_analysis)

        self.app.callback(
            Output("settings_TextDetector", "style"),
            Output("settings_ColorDetector", "style"),
            Output("settings_VQA", "style"),
            Input("Dropdown_select_Detector", "value"),
        )(self._update_detector_setting)

        self.app.callback(
            Output("textarea_questions", "style"),
            Input("Dropdown_analysis_type", "value"),
        )(self._show_questions_textarea_on_demand)

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
                    ],
                ),  # text summary end
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
                # start VQA settings
                html.Div(
                    id="settings_VQA",
                    style={"display": "none"},
                    children=[
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            dbc.Col(
                                                dcc.Dropdown(
                                                    id="Dropdown_analysis_type",
                                                    options=[
                                                        {"label": v, "value": v}
                                                        for v in SUMMARY_ANALYSIS_TYPE
                                                    ],
                                                    value="summary_and_questions",
                                                    clearable=False,
                                                    style={
                                                        "width": "100%",
                                                        "minWidth": "240px",
                                                        "maxWidth": "520px",
                                                    },
                                                ),
                                            ),
                                            justify="start",
                                        ),
                                        html.Div(style={"height": "8px"}),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dcc.Textarea(
                                                        id="textarea_questions",
                                                        value="Are there people in the image?\nWhat is this picture about?",
                                                        placeholder="One question per line...",
                                                        style={
                                                            "width": "100%",
                                                            "minHeight": "160px",
                                                            "height": "220px",
                                                            "resize": "vertical",
                                                            "overflow": "auto",
                                                        },
                                                        rows=8,
                                                    ),
                                                    width=12,
                                                ),
                                            ],
                                            justify="start",
                                        ),
                                    ]
                                )
                            ],
                            style={
                                "width": "100%",
                                "marginTop": "10px",
                                "zIndex": 2000,
                            },
                        )
                    ],
                ),
            ],
            style={"width": "100%", "display": "inline-block", "overflow": "visible"},
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
                                    "ColorDetector",
                                    "VQA",
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

        self.app.run(debug=True, port=port)

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
        # return settings_TextDetector -> style,
        display_none = {"display": "none"}
        display_flex = {
            "display": "flex",
            "flexWrap": "wrap",
            "width": 400,
            "margin-top": "20px",
        }

        if setting_input == "TextDetector":
            return display_flex, display_none, display_none, display_none
        if setting_input == "ColorDetector":
            return display_none, display_none, display_flex, display_none
        if setting_input == "VQA":
            return display_none, display_none, display_none, display_flex
        else:
            return display_none, display_none, display_none, display_none

    def _parse_questions(self, text: Optional[str]) -> Optional[List[str]]:
        if not text:
            return None
        qs = [q.strip() for q in text.splitlines() if q.strip()]
        return qs if qs else None

    def _right_output_analysis(
        self,
        n_clicks,
        all_img_options: dict,
        current_img_value: str,
        detector_value: str,
        analysis_type_value: str,
        textarea_questions_value: str,
        setting_privacy_env_var: str,
        setting_color_delta_e_method: str,
    ) -> dict:
        """Callback function to perform analysis on the selected image and return the output.

        Args:
            all_options (dict): The available options in the file explorer dropdown.
            current_value (str): The current selected value in the file explorer dropdown.

        Returns:
            dict: The analysis output for the selected image.
        """
        identify_dict = {
            "TextDetector": text.TextDetector,
            "ColorDetector": colors.ColorDetector,
            "VQA": image_summary.ImageSummaryDetector,
        }

        # Get image ID from dropdown value, which is the filepath
        if current_img_value is None:
            return {}
        image_id = all_img_options[current_img_value]
        image_copy = self.mydict.get(image_id, {}).copy()

        analysis_dict: Dict[str, Any] = {}
        if detector_value == "VQA":
            try:
                qwen_model = MultimodalSummaryModel(
                    model_id="Qwen/Qwen2.5-VL-3B-Instruct"
                )  # TODO: allow user to specify model
                vqa_cls = identify_dict.get("VQA")
                vqa_detector = vqa_cls(qwen_model, subdict={})
                questions_list = self._parse_questions(textarea_questions_value)
                analysis_result = vqa_detector.analyse_image(
                    image_copy,
                    analysis_type=analysis_type_value,
                    list_of_questions=questions_list,
                    is_concise_summary=True,
                    is_concise_answer=True,
                )
                analysis_dict = analysis_result or {}
            except Exception as e:
                warnings.warn(f"VQA/Image tasks failed: {e}")
                analysis_dict = {"image_tasks_error": str(e)}
        else:
            # detector value is the string name of the chosen detector
            identify_function = identify_dict[detector_value]

            if detector_value == "TextDetector":
                detector_class = identify_function(
                    image_copy,
                    accept_privacy=(
                        setting_privacy_env_var
                        if setting_privacy_env_var
                        else "PRIVACY_AMMICO"
                    ),
                )
            elif detector_value == "ColorDetector":
                detector_class = identify_function(
                    image_copy,
                    delta_e_method=setting_color_delta_e_method,
                )
            else:
                detector_class = identify_function(image_copy)

            analysis_dict = detector_class.analyse_image()

        new_analysis_dict: Dict[str, Any] = {}

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

    def _show_questions_textarea_on_demand(self, analysis_type_value: str) -> dict:
        if analysis_type_value in ("questions", "summary_and_questions"):
            return {"display": "block", "width": "100%"}
        else:
            return {"display": "none"}
