from google.cloud import vision
from googletrans import Translator
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from textblob import TextBlob
from textblob import download_corpora
import io
from misinformation import utils
import grpc
import pandas as pd
from bertopic import BERTopic
from transformers import pipeline

# make widgets work again
# clean text has weird spaces and separation of "do n't"
# increase coverage for text


class TextDetector(utils.AnalysisMethod):
    def __init__(self, subdict: dict, analyse_text: bool = False) -> None:
        super().__init__(subdict)
        self.subdict.update(self.set_keys())
        self.translator = Translator()
        self.analyse_text = analyse_text
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
        self.remove_linebreaks()
        if self.analyse_text:
            self._run_spacy()
            self.clean_text()
            self.correct_spelling()
            self.sentiment_analysis()
        return self.subdict

    def get_text_from_image(self):
        """Detects text on the image."""
        path = self.subdict["filename"]
        client = vision.ImageAnnotatorClient()
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
        if response.error.message:
            print("Google Cloud Vision Error")
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

    def remove_linebreaks(self):
        """Remove linebreaks from original and translated text."""
        if self.subdict["text"]:
            self.subdict["text"] = self.subdict["text"].replace("\n", " ")
            self.subdict["text_english"] = self.subdict["text_english"].replace(
                "\n", " "
            )

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

    def text_summary(self):
        # use the transformers pipeline to summarize the text
        pipe = pipeline("summarization")
        self.subdict.update(pipe(self.subdict["text_english"])[0])

    def text_sentiment_transformers(self):
        # use the transformers pipeline for text classification
        pipe = pipeline("text-classification")
        result = pipe(self.subdict["text_english"])
        self.subdict["sentiment"] = result[0]["label"]
        self.subdict["sentiment_score"] = result[0]["score"]

    def text_ner(self):
        # use the transformers pipeline for named entity recognition
        pipe = pipeline("token-classification")
        result = pipe(self.subdict["text_english"])
        self.subdict["entity"] = []
        self.subdict["entity_type"] = []
        for entity in result:
            self.subdict["entity"].append(entity["word"])
            self.subdict["entity_type"].append(entity["entity"])


class PostprocessText:
    def __init__(
        self, mydict: dict = None, use_csv: bool = False, csv_path: str = None
    ) -> None:
        self.use_csv = use_csv
        if mydict:
            print("Reading data from dict.")
            self.mydict = mydict
            self.list_text_english = self.get_text_dict()
        elif self.use_csv:
            print("Reading data from df.")
            self.df = pd.read_csv(csv_path, encoding="utf8")
            self.list_text_english = self.get_text_df()
        else:
            raise ValueError(
                "Please provide either dictionary with textual data or \
                              a csv file by setting `use_csv` to True and providing a \
                             `csv_path`."
            )

    def analyse_topic(self, return_topics: int = 3):
        """Topic analysis using BERTopic."""
        # load spacy pipeline
        nlp = spacy.load(
            "en_core_web_md",
            exclude=["tagger", "parser", "ner", "attribute_ruler", "lemmatizer"],
        )
        try:
            # unfortunately catching exceptions does not work here - need to figure out why
            self.topic_model = BERTopic(embedding_model=nlp)
        except TypeError:
            print("BERTopic excited with an error - maybe your dataset is too small?")
        self.topics, self.probs = self.topic_model.fit_transform(self.list_text_english)
        # return the topic list
        topic_df = self.topic_model.get_topic_info()
        # return the most frequent return_topics
        most_frequent_topics = []
        if len(topic_df) < return_topics:
            print("You requested more topics than are identified in your dataset -")
            print(
                "Returning only {} topics as these are all that have been found.".format(
                    len(topic_df)
                )
            )
        for i in range(min(return_topics, len(topic_df))):
            most_frequent_topics.append(self.topic_model.get_topic(i))
        return self.topic_model, topic_df, most_frequent_topics

    def get_text_dict(self):
        # use dict to put text_english in list
        list_text_english = []
        for key in self.mydict.keys():
            if "text_english" not in self.mydict[key]:
                raise ValueError(
                    "Please check your provided dictionary - \
                no english text data found."
                )
            list_text_english.append(self.mydict[key]["text_english"])
        return list_text_english

    def get_text_df(self):
        # use csv file to obtain dataframe and put text_english in list
        # check that "text_english" is there
        if "text_english" not in self.df:
            raise ValueError(
                "Please check your provided dataframe - \
                                no english text data found."
            )
        return self.df["text_english"].tolist()
