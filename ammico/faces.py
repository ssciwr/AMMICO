import cv2
import numpy as np
import os
import shutil
import pathlib
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from deepface import DeepFace
from retinaface import RetinaFace
from ammico.utils import DownloadResource, AnalysisMethod


DEEPFACE_PATH = ".deepface"


def deepface_symlink_processor(name):
    def _processor(fname, action, pooch):
        if not os.path.exists(name):
            # symlink does not work on windows
            # use copy if running on windows
            if os.name != "nt":
                os.symlink(fname, name)
            else:
                shutil.copy(fname, name)
        return fname

    return _processor


face_mask_model = DownloadResource(
    url="https://github.com/chandrikadeb7/Face-Mask-Detection/raw/v1.0.0/mask_detector.model",
    known_hash="sha256:d0b30e2c7f8f187c143d655dee8697fcfbe8678889565670cd7314fb064eadc8",
)

deepface_age_model = DownloadResource(
    url="https://github.com/serengil/deepface_models/releases/download/v1.0/age_model_weights.h5",
    known_hash="sha256:0aeff75734bfe794113756d2bfd0ac823d51e9422c8961125b570871d3c2b114",
    processor=deepface_symlink_processor(
        pathlib.Path.home().joinpath(DEEPFACE_PATH, "weights", "age_model_weights.h5")
    ),
)

deepface_face_expression_model = DownloadResource(
    url="https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5",
    known_hash="sha256:e8e8851d3fa05c001b1c27fd8841dfe08d7f82bb786a53ad8776725b7a1e824c",
    processor=deepface_symlink_processor(
        pathlib.Path.home().joinpath(
            ".deepface", "weights", "facial_expression_model_weights.h5"
        )
    ),
)

deepface_gender_model = DownloadResource(
    url="https://github.com/serengil/deepface_models/releases/download/v1.0/gender_model_weights.h5",
    known_hash="sha256:45513ce5678549112d25ab85b1926fb65986507d49c674a3d04b2ba70dba2eb5",
    processor=deepface_symlink_processor(
        pathlib.Path.home().joinpath(
            DEEPFACE_PATH, "weights", "gender_model_weights.h5"
        )
    ),
)

deepface_race_model = DownloadResource(
    url="https://github.com/serengil/deepface_models/releases/download/v1.0/race_model_single_batch.h5",
    known_hash="sha256:eb22b28b1f6dfce65b64040af4e86003a5edccb169a1a338470dde270b6f5e54",
    processor=deepface_symlink_processor(
        pathlib.Path.home().joinpath(
            DEEPFACE_PATH, "weights", "race_model_single_batch.h5"
        )
    ),
)

retinaface_model = DownloadResource(
    url="https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.h5",
    known_hash="sha256:ecb2393a89da3dd3d6796ad86660e298f62a0c8ae7578d92eb6af14e0bb93adf",
    processor=deepface_symlink_processor(
        pathlib.Path.home().joinpath(DEEPFACE_PATH, "weights", "retinaface.h5")
    ),
)

ETHICAL_STATEMENT = """DeepFace and RetinaFace provide wrappers to trained models in face
recognition and emotion detection. Age, gender and race/ethnicity models were trained on
the backbone of VGG-Face with transfer learning.

ETHICAL DISCLOSURE STATEMENT:
The Emotion Detector uses DeepFace and RetinaFace to probabilistically assess the gender,
age and race of the detected faces. Such assessments may not reflect how the individuals
identify. Additionally, the classification is carried out in simplistic categories and
contains only the most basic classes (for example, "male" and "female" for gender, and seven
non-overlapping categories for ethnicity). To access these probabilistic assessments, you
must therefore agree with the following statement: "I understand the ethical and privacy
implications such assessments have for the interpretation of the results and that this
analysis may result in personal and possibly sensitive data, and I wish to proceed."
Please type your answer in the adjacent box: "YES" for "I agree with the statement" or "NO"
for "I disagree with the statement."
"""


def ethical_disclosure(accept_disclosure: str = "DISCLOSURE_AMMICO"):
    """
    Asks the user to accept the ethical disclosure.

    Args:
        accept_disclosure (str): The name of the disclosure variable (default: "DISCLOSURE_AMMICO").
    """
    if not os.environ.get(accept_disclosure):
        accepted = _ask_for_disclosure_acceptance(accept_disclosure)
    elif os.environ.get(accept_disclosure) == "False":
        accepted = False
    elif os.environ.get(accept_disclosure) == "True":
        accepted = True
    else:
        print(
            "Could not determine disclosure - skipping \
              race/ethnicity, gender and age detection."
        )
        accepted = False
    return accepted


def _ask_for_disclosure_acceptance(accept_disclosure: str = "DISCLOSURE_AMMICO"):
    """
    Asks the user to accept the disclosure.
    """
    print(ETHICAL_STATEMENT)
    answer = input("Do you accept the disclosure? (yes/no): ")
    answer = answer.lower().strip()
    if answer == "yes":
        print("You have accepted the disclosure.")
        print(
            """Age, gender, race/ethnicity detection will be performed based on the provided
            confidence thresholds."""
        )
        os.environ[accept_disclosure] = "True"
        accepted = True
    elif answer == "no":
        print("You have not accepted the disclosure.")
        print("No age, gender, race/ethnicity detection will be performed.")
        os.environ[accept_disclosure] = "False"
        accepted = False
    else:
        print("Please answer with yes or no.")
        accepted = _ask_for_disclosure_acceptance()
    return accepted


class EmotionDetector(AnalysisMethod):
    def __init__(
        self,
        subdict: dict,
        emotion_threshold: float = 50.0,
        race_threshold: float = 50.0,
        gender_threshold: float = 50.0,
        accept_disclosure: str = "DISCLOSURE_AMMICO",
    ) -> None:
        """
        Initializes the EmotionDetector object.

        Args:
            subdict (dict): The dictionary to store the analysis results.
            emotion_threshold (float): The threshold for detecting emotions (default: 50.0).
            race_threshold (float): The threshold for detecting race (default: 50.0).
            gender_threshold (float): The threshold for detecting gender (default: 50.0).
            accept_disclosure (str): The name of the disclosure variable, that is
                set upon accepting the disclosure (default: "DISCLOSURE_AMMICO").
        """
        super().__init__(subdict)
        self.subdict.update(self.set_keys())
        # check if thresholds are valid
        if emotion_threshold < 0 or emotion_threshold > 100:
            raise ValueError("Emotion threshold must be between 0 and 100.")
        if race_threshold < 0 or race_threshold > 100:
            raise ValueError("Race threshold must be between 0 and 100.")
        if gender_threshold < 0 or gender_threshold > 100:
            raise ValueError("Gender threshold must be between 0 and 100.")
        self.emotion_threshold = emotion_threshold
        self.race_threshold = race_threshold
        self.gender_threshold = gender_threshold
        self.emotion_categories = {
            "angry": "Negative",
            "disgust": "Negative",
            "fear": "Negative",
            "sad": "Negative",
            "happy": "Positive",
            "surprise": "Neutral",
            "neutral": "Neutral",
        }
        self.accepted = ethical_disclosure(accept_disclosure)

    def set_keys(self) -> dict:
        """
        Sets the initial parameters for the analysis.

        Returns:
            dict: The dictionary with initial parameter values.
        """
        params = {
            "face": "No",
            "multiple_faces": "No",
            "no_faces": 0,
            "wears_mask": ["No"],
        }
        return params

    def analyse_image(self) -> dict:
        """
        Performs facial expression analysis on the image.

        Returns:
            dict: The updated subdict dictionary with analysis results.
        """
        return self.facial_expression_analysis()

    def _define_actions(self, fresult: dict) -> list:
        # Adapt the features we are looking for depending on whether a mask is worn.
        # White masks screw race detection, emotion detection is useless.
        # also, depending on the disclosure, we might not want to run the analysis
        # for gender, age, ethnicity/race
        conditional_actions = {
            "all": ["age", "gender", "race", "emotion"],
            "all_with_mask": ["age"],
            "restricted_access": ["emotion"],
            "restricted_access_with_mask": [],
        }
        if fresult["wears_mask"] and self.accepted:
            self.actions = conditional_actions["all_with_mask"]
        elif fresult["wears_mask"] and not self.accepted:
            self.actions = conditional_actions["restricted_access_with_mask"]
        elif not fresult["wears_mask"] and self.accepted:
            self.actions = conditional_actions["all"]
        elif not fresult["wears_mask"] and not self.accepted:
            self.actions = conditional_actions["restricted_access"]
        else:
            raise ValueError(
                "Invalid mask detection {} and disclosure \
                             acceptance {} result.".format(
                    fresult["wears_mask"], self.accepted
                )
            )

    def _ensure_deepface_models(self):
        # Ensure that all data has been fetched by pooch
        if "emotion" in self.actions:
            deepface_face_expression_model.get()
        if "race" in self.actions:
            deepface_race_model.get()
        if "age" in self.actions:
            deepface_age_model.get()
        if "gender" in self.actions:
            deepface_gender_model.get()

    def analyze_single_face(self, face: np.ndarray) -> dict:
        """
        Analyzes the features of a single face on the image.

        Args:
            face (np.ndarray): The face image array.

        Returns:
            dict: The analysis results for the face.
        """
        fresult = {}
        # Determine whether the face wears a mask
        fresult["wears_mask"] = self.wears_mask(face)
        self._define_actions(fresult)
        self._ensure_deepface_models()
        # Run the full DeepFace analysis
        # this returns a list of dictionaries
        # one dictionary per face that is detected in the image
        # since we are only passing a subregion of the image
        # that contains one face, the list will only contain one dict
        print("actions are:", self.actions)
        if self.actions != []:
            fresult["result"] = DeepFace.analyze(
                img_path=face,
                actions=self.actions,
                silent=True,
            )
        return fresult

    def facial_expression_analysis(self) -> dict:
        """
        Performs facial expression analysis on the image.

        Returns:
            dict: The updated subdict dictionary with analysis results.
        """
        # Find (multiple) faces in the image and cut them
        retinaface_model.get()

        faces = RetinaFace.extract_faces(self.subdict["filename"])
        # If no faces are found, we return empty keys
        if len(faces) == 0:
            return self.subdict
        # Sort the faces by sight to prioritize prominent faces
        faces = list(reversed(sorted(faces, key=lambda f: f.shape[0] * f.shape[1])))
        self.subdict["face"] = "Yes"
        self.subdict["multiple_faces"] = "Yes" if len(faces) > 1 else "No"
        # number of faces only counted up to 15, after that set to 99
        self.subdict["no_faces"] = len(faces) if len(faces) <= 15 else 99
        # note number of faces being identified
        # We limit ourselves to identify emotion on max three faces per image
        result = {"number_faces": len(faces) if len(faces) <= 3 else 3}
        for i, face in enumerate(faces[:3]):
            result[f"person{i+1}"] = self.analyze_single_face(face)
        self.clean_subdict(result)
        return self.subdict

    def clean_subdict(self, result: dict) -> dict:
        """
        Cleans the subdict dictionary by converting results into appropriate formats.

        Args:
            result (dict): The analysis results.
        Returns:
            dict: The updated subdict dictionary.
        """
        # Each person subdict converted into list for keys
        self.subdict["wears_mask"] = []
        if "emotion" in self.actions:
            self.subdict["emotion (category)"] = []
        for key in self.actions:
            self.subdict[key] = []
        # now iterate over the number of faces
        # and check thresholds
        # the results for each person are returned as a nested dict
        # race and emotion are given as dict with confidence values
        # gender and age are given as one value with no confidence
        # being passed
        for i in range(result["number_faces"]):
            person = "person{}".format(i + 1)
            wears_mask = result[person]["wears_mask"]
            self.subdict["wears_mask"].append("Yes" if wears_mask else "No")
            # actually the actions dict should take care of
            # the person wearing a mask or not
            for key in self.actions:
                resultdict = result[person]["result"][0]
                if key == "emotion":
                    classified_emotion = resultdict["dominant_emotion"]
                    confidence_value = resultdict[key][classified_emotion]
                    outcome = (
                        classified_emotion
                        if confidence_value > self.emotion_threshold and not wears_mask
                        else None
                    )
                    print("emotion confidence", confidence_value, outcome)
                    # also set the emotion category
                    if outcome:
                        self.subdict["emotion (category)"].append(
                            self.emotion_categories[outcome]
                        )
                    else:
                        self.subdict["emotion (category)"].append(None)
                elif key == "race":
                    classified_race = resultdict["dominant_race"]
                    confidence_value = resultdict[key][classified_race]
                    outcome = (
                        classified_race
                        if confidence_value > self.race_threshold and not wears_mask
                        else None
                    )
                elif key == "age":
                    outcome = resultdict[key]
                elif key == "gender":
                    classified_gender = resultdict["dominant_gender"]
                    confidence_value = resultdict[key][classified_gender]
                    outcome = (
                        classified_gender
                        if confidence_value > self.gender_threshold and not wears_mask
                        else None
                    )
                self.subdict[key].append(outcome)
        return self.subdict

    def wears_mask(self, face: np.ndarray) -> bool:
        """
        Determines whether a face wears a mask.

        Args:
            face (np.ndarray): The face image array.

        Returns:
            bool: True if the face wears a mask, False otherwise.
        """
        global mask_detection_model
        # Preprocess the face to match the assumptions of the face mask detection model
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)
        # Lazily load the model
        mask_detection_model = load_model(face_mask_model.get())
        # Run the model
        mask, without_mask = mask_detection_model.predict(face)[0]
        # Convert from np.bool_ to bool to later be able to serialize the result
        return bool(mask > without_mask)
