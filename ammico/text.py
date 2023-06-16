from google.cloud import vision
from google.auth.exceptions import DefaultCredentialsError
from googletrans import Translator
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from textblob import TextBlob
from textblob import download_corpora
import io
from ammico import utils
import grpc
import pandas as pd
from bertopic import BERTopic
from transformers import pipeline
import os

# clean text has weird spaces and separation of "do n't"
# increase coverage for text


class TextDetector(utils.AnalysisMethod):
    def __init__(self, subdict: dict, analyse_text: bool = False) -> None:
        """Init text detection class.

        Args:
            subdict (dict): Dictionary containing file name/path, and possibly previous
            analysis results from other modules.
            analyse_text (bool, optional): Decide if extracted text will be further subject
            to analysis. Defaults to False.
        """
        super().__init__(subdict)
        self.subdict.update(self.set_keys())
        self.translator = Translator()
        self.analyse_text = analyse_text
        if self.analyse_text:
            self._initialize_spacy()
            self._initialize_textblob()

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
        self.nlp.add_pipe("spacytextblob")

    def _initialize_textblob(self):
        """Initialize the TextBlob library for text analysis."""
        try:
            TextBlob("Here")
        except Exception:
            download_corpora.main()

    def analyse_image(self) -> dict:
        """Perform text extraction and analysis of the text.

        Returns:
            dict: The updated dictionary with text analysis results.
        """
        self.get_text_from_image()
        self.translate_text()
        self.remove_linebreaks()
        if self.analyse_text:
            self._run_spacy()
            self.clean_text()
            self.correct_spelling()
            self.sentiment_analysis()
            self.text_summary()
            self.text_sentiment_transformers()
            self.text_ner()
        return self.subdict

    def get_text_from_image(self):
        """Detect text on the image using Google Cloud Vision API."""
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
        """Generate Spacy doc object for further text analysis."""
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
        """Correct the spelling of the English text using TextBlob."""
        self.textblob = TextBlob(self.subdict["text_english"])
        self.subdict["text_english_correct"] = str(self.textblob.correct())

    def sentiment_analysis(self):
        """Perform sentiment analysis on the text using SpacyTextBlob."""
        # polarity is between [-1.0, 1.0]
        self.subdict["polarity"] = self.doc._.blob.polarity
        # subjectivity is a float within the range [0.0, 1.0]
        # where 0.0 is very objective and 1.0 is very subjective
        self.subdict["subjectivity"] = self.doc._.blob.subjectivity

    def text_summary(self):
        """Generate a summary of the text using the Transformers pipeline."""
        # use the transformers pipeline to summarize the text
        # use the current default model - 03/2023
        model_name = "sshleifer/distilbart-cnn-12-6"
        model_revision = "a4f8f3e"
        max_number_of_characters = 3000
        pipe = pipeline(
            "summarization",
            model=model_name,
            revision=model_revision,
            min_length=5,
            max_length=20,
        )
        try:
            summary = pipe(self.subdict["text_english"][0:max_number_of_characters])
            self.subdict["text_summary"] = summary[0]["summary_text"]
        except IndexError:
            print(
                "Cannot provide summary for this object - please check that the text has been translated correctly."
            )
            print("Image: {}".format(self.subdict["filename"]))
            self.subdict["text_summary"] = None

    def text_sentiment_transformers(self):
        """Perform text classification for sentiment using the Transformers pipeline."""
        # use the transformers pipeline for text classification
        # use the current default model - 03/2023
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        model_revision = "af0f99b"
        pipe = pipeline(
            "text-classification",
            model=model_name,
            revision=model_revision,
            truncation=True,
        )
        result = pipe(self.subdict["text_english"])
        self.subdict["sentiment"] = result[0]["label"]
        self.subdict["sentiment_score"] = result[0]["score"]

    def text_ner(self):
        """Perform named entity recognition on the text using the Transformers pipeline."""
        # use the transformers pipeline for named entity recognition
        # use the current default model - 03/2023
        model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
        model_revision = "f2482bf"
        pipe = pipeline(
            "token-classification",
            model=model_name,
            revision=model_revision,
            aggregation_strategy="simple",
        )
        result = pipe(self.subdict["text_english"])
        self.subdict["entity"] = []
        self.subdict["entity_type"] = []
        for entity in result:
            self.subdict["entity"].append(entity["word"])
            self.subdict["entity_type"].append(entity["entity_group"])


class PostprocessText:
    def __init__(
        self,
        mydict: dict = None,
        use_csv: bool = False,
        csv_path: str = None,
        analyze_text: str = "text_english",
    ) -> None:
        """
        Initializes the PostprocessText class that handles the topic analysis.

        Args:
            mydict (dict, optional): Dictionary with textual data. Defaults to None.
            use_csv (bool, optional): Flag indicating whether to use a CSV file. Defaults to False.
            csv_path (str, optional): Path to the CSV file. Required if `use_csv` is True. Defaults to None.
            analyze_text (str, optional): Key for the text field to analyze. Defaults to "text_english".
        """
        self.use_csv = use_csv
        if mydict:
            print("Reading data from dict.")
            self.mydict = mydict
            self.list_text_english = self.get_text_dict(analyze_text)
        elif self.use_csv:
            print("Reading data from df.")
            self.df = pd.read_csv(csv_path, encoding="utf8")
            self.list_text_english = self.get_text_df(analyze_text)
        else:
            raise ValueError(
                "Please provide either dictionary with textual data or \
                              a csv file by setting `use_csv` to True and providing a \
                             `csv_path`."
            )

    def analyse_topic(self, return_topics: int = 3) -> tuple:
        """
        Performs topic analysis using BERTopic.

        Args:
            return_topics (int, optional): Number of topics to return. Defaults to 3.

        Returns:
            tuple: A tuple containing the topic model, topic dataframe, and most frequent topics.
        """
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

    def get_text_dict(self, analyze_text: str) -> list:
        """
        Extracts text from the provided dictionary.

        Args:
            analyze_text (str): Key for the text field to analyze.

        Returns:
            list: A list of text extracted from the dictionary.
        """
        # use dict to put text_english or text_summary in list
        list_text_english = []
        for key in self.mydict.keys():
            if analyze_text not in self.mydict[key]:
                raise ValueError(
                    "Please check your provided dictionary - \
                no {} text data found.".format(
                        analyze_text
                    )
                )
            list_text_english.append(self.mydict[key][analyze_text])
        return list_text_english

    def get_text_df(self, analyze_text: str) -> list:
        """
        Extracts text from the provided dataframe.

        Args:
            analyze_text (str): Column name for the text field to analyze.

        Returns:
            list: A list of text extracted from the dataframe.
        """
        # use csv file to obtain dataframe and put text_english or text_summary in list
        # check that "text_english" or "text_summary" is there
        if analyze_text not in self.df:
            raise ValueError(
                "Please check your provided dataframe - \
                                no {} text data found.".format(
                    analyze_text
                )
            )
        return self.df[analyze_text].tolist()
