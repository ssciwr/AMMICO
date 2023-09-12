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


class EmotionDetector(AnalysisMethod):
    def __init__(
        self,
        subdict: dict,
        emotion_threshold: float = 50.0,
        race_threshold: float = 50.0,
    ) -> None:
        """
        Initializes the EmotionDetector object.

        Args:
            subdict (dict): The dictionary to store the analysis results.
            emotion_threshold (float): The threshold for detecting emotions (default: 50.0).
            race_threshold (float): The threshold for detecting race (default: 50.0).
        """
        super().__init__(subdict)
        self.subdict.update(self.set_keys())
        self.emotion_threshold = emotion_threshold
        self.race_threshold = race_threshold
        self.emotion_categories = {
            "angry": "Negative",
            "disgust": "Negative",
            "fear": "Negative",
            "sad": "Negative",
            "happy": "Positive",
            "surprise": "Neutral",
            "neutral": "Neutral",
        }

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
            "age": [None],
            "gender": [None],
            "race": [None],
            "emotion": [None],
            "emotion (category)": [None],
        }
        return params

    def analyse_image(self) -> dict:
        """
        Performs facial expression analysis on the image.

        Returns:
            dict: The updated subdict dictionary with analysis results.
        """
        return self.facial_expression_analysis()

    def analyze_single_face(self, face: np.ndarray) -> dict:
        """
        Analyzes the features of a single face.

        Args:
            face (np.ndarray): The face image array.

        Returns:
            dict: The analysis results for the face.
        """
        fresult = {}
        # Determine whether the face wears a mask
        fresult["wears_mask"] = self.wears_mask(face)
        # Adapt the features we are looking for depending on whether a mask is worn.
        # White masks screw race detection, emotion detection is useless.
        actions = ["age", "gender"]
        if not fresult["wears_mask"]:
            actions = actions + ["race", "emotion"]
        # Ensure that all data has been fetched by pooch
        deepface_age_model.get()
        deepface_face_expression_model.get()
        deepface_gender_model.get()
        deepface_race_model.get()
        # Run the full DeepFace analysis
        fresult.update(
            DeepFace.analyze(
                img_path=face,
                actions=actions,
                prog_bar=False,
                detector_backend="skip",
            )
        )
        # We remove the region, as the data is not correct - after all we are
        # running the analysis on a subimage.
        del fresult["region"]
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
        self.subdict["no_faces"] = len(faces) if len(faces) <= 15 else 99
        # note number of faces being identified
        result = {"number_faces": len(faces) if len(faces) <= 3 else 3}
        # We limit ourselves to three faces
        for i, face in enumerate(faces[:3]):
            result[f"person{ i+1 }"] = self.analyze_single_face(face)
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
        self.subdict["age"] = []
        self.subdict["gender"] = []
        self.subdict["race"] = []
        self.subdict["emotion"] = []
        self.subdict["emotion (category)"] = []
        for i in range(result["number_faces"]):
            person = "person{}".format(i + 1)
            self.subdict["wears_mask"].append(
                "Yes" if result[person]["wears_mask"] else "No"
            )
            self.subdict["age"].append(result[person]["age"])
            # Gender is now reported as a list of dictionaries.
            # Each dict represents one face.
            # Each dict contains probability for Woman and Man.
            # We take only the higher probability result for each dict.
            self.subdict["gender"].append(result[person]["gender"])
            # Race and emotion are only detected if a person does not wear a mask
            if result[person]["wears_mask"]:
                self.subdict["race"].append(None)
                self.subdict["emotion"].append(None)
                self.subdict["emotion (category)"].append(None)
            elif not result[person]["wears_mask"]:
                # Check whether the race threshold was exceeded
                if (
                    result[person]["race"][result[person]["dominant_race"]]
                    > self.race_threshold
                ):
                    self.subdict["race"].append(result[person]["dominant_race"])
                else:
                    self.subdict["race"].append(None)

                # Check whether the emotion threshold was exceeded
                if (
                    result[person]["emotion"][result[person]["dominant_emotion"]]
                    > self.emotion_threshold
                ):
                    self.subdict["emotion"].append(result[person]["dominant_emotion"])
                    self.subdict["emotion (category)"].append(
                        self.emotion_categories[result[person]["dominant_emotion"]]
                    )
                else:
                    self.subdict["emotion"].append(None)
                    self.subdict["emotion (category)"].append(None)
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
