from google.cloud import vision
from google.auth.exceptions import DefaultCredentialsError
from googletrans import Translator
import spacy
import io
import os
import re
from ammico.utils import AnalysisMethod
import grpc
import pandas as pd

PRIVACY_STATEMENT = """The Text Detector uses Google Cloud Vision
    and Google Translate. Detailed information about how information
    is being processed is provided here:
    https://ssciwr.github.io/AMMICO/build/html/faq_link.html.
    Google’s privacy policy can be read here: https://policies.google.com/privacy.
    By continuing to use this Detector, you agree to send the data you want analyzed
    to the Google servers for extraction and translation."""


def privacy_disclosure(accept_privacy: str = "PRIVACY_AMMICO"):
    """
    Asks the user to accept the privacy statement.

    Args:
        accept_privacy (str): The name of the disclosure variable (default: "PRIVACY_AMMICO").
    """
    if not os.environ.get(accept_privacy):
        accepted = _ask_for_privacy_acceptance(accept_privacy)
    elif os.environ.get(accept_privacy) == "False":
        accepted = False
    elif os.environ.get(accept_privacy) == "True":
        accepted = True
    else:
        print(
            "Could not determine privacy disclosure - skipping \
              text detection and translation."
        )
        accepted = False
    return accepted


def _ask_for_privacy_acceptance(accept_privacy: str = "PRIVACY_AMMICO"):
    """
    Asks the user to accept the disclosure.
    """
    print(PRIVACY_STATEMENT)
    answer = input("Do you accept the privacy disclosure? (yes/no): ")
    answer = answer.lower().strip()
    if answer == "yes":
        print("You have accepted the privacy disclosure.")
        print("""Text detection and translation will be performed.""")
        os.environ[accept_privacy] = "True"
        accepted = True
    elif answer == "no":
        print("You have not accepted the privacy disclosure.")
        print("No text detection and translation will be performed.")
        os.environ[accept_privacy] = "False"
        accepted = False
    else:
        print("Please answer with yes or no.")
        accepted = _ask_for_privacy_acceptance()
    return accepted


class TextDetector(AnalysisMethod):
    def __init__(
        self,
        subdict: dict,
        skip_extraction: bool = False,
        accept_privacy: str = "PRIVACY_AMMICO",
    ) -> None:
        """Init text detection class.

        Args:
            subdict (dict): Dictionary containing file name/path, and possibly previous
                analysis results from other modules.
            skip_extraction (bool, optional): Decide if text will be extracted from images or
                is already provided via a csv. Defaults to False.
            accept_privacy (str, optional): Environment variable to accept the privacy
                statement for the Google Cloud processing of the data. Defaults to
                "PRIVACY_AMMICO".
        """
        super().__init__(subdict)
        # disable this for now
        # maybe it would be better to initialize the keys differently
        # the reason is that they are inconsistent depending on the selected
        # options, and also this may not be really necessary and rather restrictive
        # self.subdict.update(self.set_keys())
        self.accepted = privacy_disclosure(accept_privacy)
        if not self.accepted:
            raise ValueError(
                "Privacy disclosure not accepted - skipping text detection."
            )
        self.translator = Translator(raise_exception=True)
        self.skip_extraction = skip_extraction
        if not isinstance(skip_extraction, bool):
            raise ValueError("skip_extraction needs to be set to true or false")
        if self.skip_extraction:
            print("Skipping text extraction from image.")
            print("Reading text directly from provided dictionary.")
        self._initialize_spacy()

    def set_keys(self) -> dict:
        """Set the default keys for text analysis.

        Returns:
            dict: The dictionary with default text keys.
        """
        params = {"text": None, "text_language": None, "text_english": None}
        return params

    def _initialize_spacy(self):
        """Initialize the Spacy library for text analysis."""
        try:
            self.nlp = spacy.load("en_core_web_md")
        except Exception:
            spacy.cli.download("en_core_web_md")
            self.nlp = spacy.load("en_core_web_md")

    def _check_add_space_after_full_stop(self):
        """Add a space after a full stop. Required by googletrans."""
        # we have found text, now we check for full stops
        index_stop = [
            i.start()
            for i in re.finditer("\.", self.subdict["text"])  # noqa
        ]
        if not index_stop:  # no full stops found
            return
        # check if this includes the last string item
        end_of_list = False
        if len(self.subdict["text"]) <= (index_stop[-1] + 1):
            # the last found full stop is at the end of the string
            # but we can include all others
            if len(index_stop) == 1:
                end_of_list = True
            else:
                index_stop.pop()
        if end_of_list:  # only one full stop at end of string
            return
        # if this is not the end of the list, check if there is a space after the full stop
        no_space = [i for i in index_stop if self.subdict["text"][i + 1] != " "]
        if not no_space:  # all full stops have a space after them
            return
        # else, amend the text
        add_one = 1
        for i in no_space:
            self.subdict["text"] = (
                self.subdict["text"][: i + add_one]
                + " "
                + self.subdict["text"][i + add_one :]
            )
            add_one += 1

    def _truncate_text(self, max_length: int = 5000) -> str:
        """Truncate the text if it is too long for googletrans."""
        if self.subdict["text"] and len(self.subdict["text"]) > max_length:
            print("Text is too long - truncating to {} characters.".format(max_length))
            self.subdict["text_truncated"] = self.subdict["text"][:max_length]

    def analyse_image(self) -> dict:
        """Perform text extraction and analysis of the text.

        Returns:
            dict: The updated dictionary with text analysis results.
        """
        if not self.skip_extraction:
            self.get_text_from_image()
        # check that text was found
        if not self.subdict["text"]:
            print("No text found - skipping analysis.")
        else:
            # make sure all full stops are followed by whitespace
            # otherwise googletrans breaks
            self._check_add_space_after_full_stop()
            self._truncate_text()
            self.translate_text()
            self.remove_linebreaks()
            if self.subdict["text_english"]:
                self._run_spacy()
        return self.subdict

    def get_text_from_image(self):
        """Detect text on the image using Google Cloud Vision API."""
        if not self.accepted:
            raise ValueError(
                "Privacy disclosure not accepted - skipping text detection."
            )
        path = self.subdict["filename"]
        try:
            client = vision.ImageAnnotatorClient()
        except DefaultCredentialsError:
            raise DefaultCredentialsError(
                "Please provide credentials for google cloud vision API, see https://cloud.google.com/docs/authentication/application-default-credentials."
            )
        with io.open(path, "rb") as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        # check for usual connection errors and retry if necessary
        try:
            response = client.text_detection(image=image)
        except grpc.RpcError as exc:
            print("Cloud vision API connection failed")
            print("Skipping this image ..{}".format(path))
            print("Connection failed with code {}: {}".format(exc.code(), exc))
        # here check if text was found on image
        if response:
            texts = response.text_annotations[0].description
            self.subdict["text"] = texts
        else:
            print("No text found on image.")
            self.subdict["text"] = None
        if response.error.message:
            print("Google Cloud Vision Error")
            raise ValueError(
                "{}\nFor more info on error messages, check: "
                "https://cloud.google.com/apis/design/errors".format(
                    response.error.message
                )
            )

    def translate_text(self):
        """Translate the detected text to English using the Translator object."""
        if not self.accepted:
            raise ValueError(
                "Privacy disclosure not accepted - skipping text translation."
            )
        text_to_translate = (
            self.subdict["text_truncated"]
            if "text_truncated" in self.subdict
            else self.subdict["text"]
        )
        try:
            translated = self.translator.translate(text_to_translate)
        except Exception:
            print("Could not translate the text with error {}.".format(Exception))
            translated = None
            print("Skipping translation for this text.")
        self.subdict["text_language"] = translated.src if translated else None
        self.subdict["text_english"] = translated.text if translated else None

    def remove_linebreaks(self):
        """Remove linebreaks from original and translated text."""
        if self.subdict["text"] and self.subdict["text_english"]:
            self.subdict["text"] = self.subdict["text"].replace("\n", " ")
            self.subdict["text_english"] = self.subdict["text_english"].replace(
                "\n", " "
            )

    def _run_spacy(self):
        """Generate Spacy doc object for further text analysis."""
        self.doc = self.nlp(self.subdict["text_english"])


class TextAnalyzer:
    """Used to get text from a csv and then run the TextDetector on it."""

    def __init__(
        self, csv_path: str, column_key: str = None, csv_encoding: str = "utf-8"
    ) -> None:
        """Init the TextTranslator class.

        Args:
            csv_path (str): Path to the CSV file containing the text entries.
            column_key (str): Key for the column containing the text entries.
                Defaults to None.
            csv_encoding (str): Encoding of the CSV file. Defaults to "utf-8".
        """
        self.csv_path = csv_path
        self.column_key = column_key
        self.csv_encoding = csv_encoding
        self._check_valid_csv_path()
        self._check_file_exists()
        if not self.column_key:
            print("No column key provided - using 'text' as default.")
            self.column_key = "text"
        if not self.csv_encoding:
            print("No encoding provided - using 'utf-8' as default.")
            self.csv_encoding = "utf-8"
        if not isinstance(self.column_key, str):
            raise ValueError("The provided column key is not a string.")
        if not isinstance(self.csv_encoding, str):
            raise ValueError("The provided encoding is not a string.")

    def _check_valid_csv_path(self):
        if not isinstance(self.csv_path, str):
            raise ValueError("The provided path to the CSV file is not a string.")
        if not self.csv_path.endswith(".csv"):
            raise ValueError("The provided file is not a CSV file.")

    def _check_file_exists(self):
        try:
            with open(self.csv_path, "r") as file:  # noqa
                pass
        except FileNotFoundError:
            raise FileNotFoundError("The provided CSV file does not exist.")

    def read_csv(self) -> dict:
        """Read the CSV file and return the dictionary with the text entries.

        Returns:
            dict: The dictionary with the text entries.
        """
        df = pd.read_csv(self.csv_path, encoding=self.csv_encoding)

        if self.column_key not in df:
            raise ValueError(
                "The provided column key is not in the CSV file. Please check."
            )
        self.mylist = df[self.column_key].to_list()
        self.mydict = {}
        for i, text in enumerate(self.mylist):
            self.mydict[self.csv_path + "row-" + str(i)] = {
                "filename": self.csv_path,
                "text": text,
            }
