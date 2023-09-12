from google.cloud import vision
from google.auth.exceptions import DefaultCredentialsError
from googletrans import Translator
import spacy
import io
from ammico.utils import AnalysisMethod
import grpc
import pandas as pd
from bertopic import BERTopic
from transformers import pipeline


class TextDetector(AnalysisMethod):
    def __init__(
        self,
        subdict: dict,
        analyse_text: bool = False,
        model_names: list = None,
        revision_numbers: list = None,
    ) -> None:
        """Init text detection class.

        Args:
            subdict (dict): Dictionary containing file name/path, and possibly previous
                analysis results from other modules.
            analyse_text (bool, optional): Decide if extracted text will be further subject
                to analysis. Defaults to False.
            model_names (list, optional): Provide model names for summary, sentiment and ner
                analysis. Defaults to None, in which case the default model from transformers
                are used (as of 03/2023): "sshleifer/distilbart-cnn-12-6" (summary),
                "distilbert-base-uncased-finetuned-sst-2-english" (sentiment),
                "dbmdz/bert-large-cased-finetuned-conll03-english".
                To select other models, provide a list with three entries, the first for
                summary, second for sentiment, third for NER, with the desired model names.
                Set one of these to None to still use the default model.
            revision_numbers (list, optional): Model revision (commit) numbers on the
                Hugging Face hub. Provide this to make sure you are using the same model.
                Defaults to None, except if the default models are used; then it defaults to
                "a4f8f3e" (summary, distilbart), "af0f99b" (sentiment, distilbert),
                "f2482bf" (NER, bert).
        """
        super().__init__(subdict)
        self.subdict.update(self.set_keys())
        self.translator = Translator()
        if not isinstance(analyse_text, bool):
            raise ValueError("analyse_text needs to be set to true or false")
        self.analyse_text = analyse_text
        if self.analyse_text:
            self._initialize_spacy()
        if model_names:
            self._check_valid_models(model_names)
        if revision_numbers:
            self._check_revision_numbers(revision_numbers)
        # initialize revision numbers and models
        self._init_revision_numbers(model_names, revision_numbers)
        self._init_model(model_names)

    def _check_valid_models(self, model_names):
        # check that model_names and revision_numbers are valid lists or None
        # check that model names are a list
        if not isinstance(model_names, list):
            raise ValueError("Model names need to be provided as a list!")
        # check that enough models are provided, one for each method
        if len(model_names) != 3:
            raise ValueError(
                "Not enough or too many model names provided - three are required, one each for summary, sentiment, ner"
            )

    def _check_revision_numbers(self, revision_numbers):
        # check that revision numbers are list
        if not isinstance(revision_numbers, list):
            raise ValueError("Revision numbers need to be provided as a list!")
        # check that three revision numbers are provided, one for each method
        if len(revision_numbers) != 3:
            raise ValueError(
                "Not enough or too many revision numbers provided - three are required, one each for summary, sentiment, ner"
            )

    def _init_revision_numbers(self, model_names, revision_numbers):
        """Helper method to set the revision (version) number for each model."""
        revision_numbers_default = ["a4f8f3e", "af0f99b", "f2482bf"]
        if model_names:
            # if model_names is provided, set revision numbers for each of the methods
            # either as the provided revision number or None or as the default revision number,
            # if one of the methods uses the default model
            self._init_revision_numbers_per_model(
                model_names, revision_numbers, revision_numbers_default
            )
        else:
            # model_names was not provided, revision numbers are the default revision numbers or None
            self.revision_summary = revision_numbers_default[0]
            self.revision_sentiment = revision_numbers_default[1]
            self.revision_ner = revision_numbers_default[2]

    def _init_revision_numbers_per_model(
        self, model_names, revision_numbers, revision_numbers_default
    ):
        task_list = []
        if not revision_numbers:
            # no revision numbers for non-default models provided
            revision_numbers = [None, None, None]
        for model, revision, revision_default in zip(
            model_names, revision_numbers, revision_numbers_default
        ):
            # a model was specified for this task, set specified revision number or None
            # or: model for this task was set to None, so we take default version number for default model
            task_list.append(revision if model else revision_default)
        self.revision_summary = task_list[0]
        self.revision_sentiment = task_list[1]
        self.revision_ner = task_list[2]

    def _init_model(self, model_names):
        """Helper method to set the model name for each analysis method."""
        # assign models for each of the text analysis methods
        # and check that they are valid
        model_names_default = [
            "sshleifer/distilbart-cnn-12-6",
            "distilbert-base-uncased-finetuned-sst-2-english",
            "dbmdz/bert-large-cased-finetuned-conll03-english",
        ]
        # no model names provided, set the default
        if not model_names:
            model_names = model_names_default
        # now assign model names for each of the methods
        # either to the provided model name or the default if one of the
        # task's models is set to None
        self.model_summary = (
            model_names[0] if model_names[0] else model_names_default[0]
        )
        self.model_sentiment = (
            model_names[1] if model_names[1] else model_names_default[1]
        )
        self.model_ner = model_names[2] if model_names[2] else model_names_default[2]

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

    def text_summary(self):
        """Generate a summary of the text using the Transformers pipeline."""
        # use the transformers pipeline to summarize the text
        # use the current default model - 03/2023
        max_number_of_characters = 3000
        pipe = pipeline(
            "summarization",
            model=self.model_summary,
            revision=self.revision_summary,
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
        pipe = pipeline(
            "text-classification",
            model=self.model_sentiment,
            revision=self.revision_sentiment,
            truncation=True,
        )
        result = pipe(self.subdict["text_english"])
        self.subdict["sentiment"] = result[0]["label"]
        self.subdict["sentiment_score"] = round(result[0]["score"], 2)

    def text_ner(self):
        """Perform named entity recognition on the text using the Transformers pipeline."""
        # use the transformers pipeline for named entity recognition
        # use the current default model - 03/2023
        pipe = pipeline(
            "token-classification",
            model=self.model_ner,
            revision=self.revision_ner,
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
        # initialize spacy
        self._initialize_spacy()

    def _initialize_spacy(self):
        try:
            self.nlp = spacy.load(
                "en_core_web_md",
                exclude=["tagger", "parser", "ner", "attribute_ruler", "lemmatizer"],
            )
        except Exception:
            spacy.cli.download("en_core_web_md")
            self.nlp = spacy.load(
                "en_core_web_md",
                exclude=["tagger", "parser", "ner", "attribute_ruler", "lemmatizer"],
            )

    def analyse_topic(self, return_topics: int = 3) -> tuple:
        """
        Performs topic analysis using BERTopic.

        Args:
            return_topics (int, optional): Number of topics to return. Defaults to 3.

        Returns:
            tuple: A tuple containing the topic model, topic dataframe, and most frequent topics.
        """
        try:
            # unfortunately catching exceptions does not work here - need to figure out why
            self.topic_model = BERTopic(embedding_model=self.nlp)
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
