from google.cloud import vision
from googletrans import Translator
import spacy
import io
from misinformation import utils

# make widgets work again
# clean text has weird spaces and separation of "do n't"
# increase coverage for text


class TextDetector(utils.AnalysisMethod):
    def __init__(self, subdict: dict) -> None:
        super().__init__(subdict)
        self.subdict.update(self.set_keys())
        self.translator = Translator()
        # spacy load should be separaate method with error if model not found / dynamic download
        self.nlp = spacy.load("en_core_web_md")

    def set_keys(self) -> dict:
        params = {
            "text": None,
            "text_language": None,
            "text_english": None,
            "text_cleaned": None,
        }
        return params

    def analyse_image(self):
        self.get_text_from_image()
        self.translate_text()
        self._init_spacy()
        self.clean_text()
        return self.subdict

    def get_text_from_image(self):
        """Detects text on the image."""
        path = self.subdict["filename"]
        client = vision.ImageAnnotatorClient()
        with io.open(path, "rb") as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        response = client.text_detection(image=image)
        texts = response.text_annotations[0].description
        # here check if text was found
        self.subdict = {"text": texts}
        if response.error.message:
            raise ValueError(
                "{}\nFor more info on error messages, check: "
                "https://cloud.google.com/apis/design/errors".format(
                    response.error.message
                )
            )

    def translate_text(self):
        translated = self.translator.translate(self.subdict["text"])
        self.subdict["text_language"] = translated.src
        self.subdict["text_english"] = translated.text

    def _init_spacy(self):
        """Generate spacy doc object."""
        self.doc = self.nlp(self.subdict["text_english"])

    def clean_text(self):
        """Clean the text from unrecognized words and any numbers."""
        templist = []
        for token in self.doc:
            templist.append(
                token.text
            ) if token.pos_ != "NUM" and token.has_vector else None
        self.subdict["text_clean"] = " ".join(templist).rstrip().lstrip()
