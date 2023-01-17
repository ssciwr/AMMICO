from google.cloud import vision
from googletrans import Translator
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from textblob import TextBlob
from textblob import download_corpora
import io
from misinformation import utils

# make widgets work again
# clean text has weird spaces and separation of "do n't"
# increase coverage for text


class TextDetector(utils.AnalysisMethod):
    def __init__(
        self, subdict: dict, analyse_text: bool = False, analyse_topic: bool = False
    ) -> None:
        super().__init__(subdict)
        self.subdict.update(self.set_keys())
        self.translator = Translator()
        self.analyse_text = analyse_text
        self.analyse_topic = analyse_topic
        if self.analyse_text:
            self._initialize_spacy()
            self._initialize_textblob()

    def set_keys(self) -> dict:
        params = {"text": None, "text_language": None, "text_english": None}
        return params

    def _initialize_spacy(self):
        try:
            self.nlp = spacy.load("en_core_web_md")
        except Exception:
            spacy.cli.download("en_core_web_md")
            self.nlp = spacy.load("en_core_web_md")
        self.nlp.add_pipe("spacytextblob")

    def _initialize_textblob(self):
        try:
            TextBlob("Here")
        except Exception:
            download_corpora.main()

    def analyse_image(self):
        self.get_text_from_image()
        self.translate_text()
        if self.analyse_text:
            self._run_spacy()
            self.clean_text()
            self.correct_spelling()
            self.sentiment_analysis()
        if self.analyse_topic:
            self.analyse_topic()
        return self.subdict

    def get_text_from_image(self):
        """Detects text on the image."""
        path = self.subdict["filename"]
        client = vision.ImageAnnotatorClient()
        with io.open(path, "rb") as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        response = client.text_detection(image=image)
        # here check if text was found
        if response:
            texts = response.text_annotations[0].description
            self.subdict["text"] = texts
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

    def _run_spacy(self):
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

    def correct_spelling(self):
        self.textblob = TextBlob(self.subdict["text_english"])
        self.subdict["text_english_correct"] = str(self.textblob.correct())

    def sentiment_analysis(self):
        # self.subdict["sentiment"] = self.doc._.blob.sentiment_assessments.assessments
        # polarity is between [-1.0, 1.0]
        self.subdict["polarity"] = self.doc._.blob.polarity
        # subjectivity is a float within the range [0.0, 1.0]
        # where 0.0 is very objective and 1.0 is very subjective
        self.subdict["subjectivity"] = self.doc._.blob.subjectivity

    def analyse_topic(self):
        pass
