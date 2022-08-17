import cv2
import numpy as np
import os
import pathlib
import ipywidgets

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from deepface import DeepFace
from retinaface import RetinaFace

from misinformation.utils import DownloadResource
import misinformation.utils as utils


def deepface_symlink_processor(name):
    def _processor(fname, action, pooch):
        if not os.path.exists(name):
            os.symlink(fname, name)
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
        pathlib.Path.home().joinpath(".deepface", "weights", "age_model_weights.h5")
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
        pathlib.Path.home().joinpath(".deepface", "weights", "gender_model_weights.h5")
    ),
)

deepface_race_model = DownloadResource(
    url="https://github.com/serengil/deepface_models/releases/download/v1.0/race_model_single_batch.h5",
    known_hash="sha256:eb22b28b1f6dfce65b64040af4e86003a5edccb169a1a338470dde270b6f5e54",
    processor=deepface_symlink_processor(
        pathlib.Path.home().joinpath(
            ".deepface", "weights", "race_model_single_batch.h5"
        )
    ),
)

retinaface_model = DownloadResource(
    url="https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.h5",
    known_hash="sha256:ecb2393a89da3dd3d6796ad86660e298f62a0c8ae7578d92eb6af14e0bb93adf",
    processor=deepface_symlink_processor(
        pathlib.Path.home().joinpath(".deepface", "weights", "retinaface.h5")
    ),
)


class EmotionDetector(utils.AnalysisMethod):
    def __init__(self, subdict: dict) -> None:
        super().__init__(subdict)
        self.subdict.update(self.set_keys())
        self.emotion_threshold = 25.0
        self.race_threshold = 80.0
        self.negative_emotion = ["angry", "disgust", "fear", "sad"]
        self.positive_emotion = ["happy"]
        self.neutral_emotion = ["surprise", "neutral"]

    def set_keys(self) -> dict:
        params = {
            "face": "No",
            "wears_mask": "No",
            "age": None,
            "gender": None,
            "race": None,
            "emotion": None,
            "emotion (category)": None,
        }
        return params

    def analyse_image(self):
        return self.facial_expression_analysis()

    def analyze_single_face(self, face: np.ndarray) -> dict:
        fresult = {}

        # Determine whether the face wears a mask
        fresult["wears_mask"] = self.wears_mask(face)

        # Adapt the features we are looking for depending on whether a mask is
        # worn. White masks screw race detection, emotion detection is useless.
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

        # Find (multiple) faces in the image and cut them
        retinaface_model.get()
        faces = RetinaFace.extract_faces(self.subdict["filename"])

        # If no faces are found, we return empty keys
        if len(faces) == 0:
            return self.subdict

        # Sort the faces by sight to prioritize prominent faces
        faces = list(reversed(sorted(faces, key=lambda f: f.shape[0] * f.shape[1])))

        self.subdict["face"] = "yes"
        # note number of faces being identified
        result = {"number_faces": len(faces) if len(faces) <= 3 else 3}
        # We limit ourselves to three faces
        for i, face in enumerate(faces[:3]):
            result[f"person{ i+1 }"] = self.analyze_single_face(face)

        self.clean_subdict(result)

        return self.subdict

    def clean_subdict(self, result: dict) -> dict:
        # each person subdict converted into list for keys
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
            self.subdict["gender"].append(result[person]["gender"])
            # race, emotion only detected if person does not wear mask
            if result[person]["wears_mask"]:
                self.subdict["race"].append(None)
                self.subdict["emotion"].append(None)
                self.subdict["emotion (category)"].append(None)
            elif not result[person]["wears_mask"]:
                # also assign categories based on threshold
                cumulative = [
                    sum(
                        result[person]["emotion"][key]
                        for key in result[person]["emotion"].keys()
                        if key in self.negative_emotion
                    )
                ]
                cumulative.append(
                    sum(
                        result[person]["emotion"][key]
                        for key in result[person]["emotion"].keys()
                        if key in self.positive_emotion
                    )
                )
                cumulative.append(
                    sum(
                        result[person]["emotion"][key]
                        for key in result[person]["emotion"].keys()
                        if key in self.neutral_emotion
                    )
                )
                expression = ["Negative", "Positive", "Neutral"]
                # now zip the two lists and sort according to highest contribution
                category = sorted(zip(cumulative, expression), reverse=True)[0][1]
                self.subdict["race"].append(result[person]["dominant_race"])
                dominant = result[person]["dominant_emotion"]
                self.subdict["emotion"].append(
                    (dominant, result[person]["emotion"][dominant])
                )
                self.subdict["emotion (category)"].append(category)
        return self.subdict

    def wears_mask(self, face: np.ndarray) -> bool:
        global mask_detection_model

        # Preprocess the face to match the assumptions of the face mask
        # detection model
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        # Lazily load the model
        mask_detection_model = load_model(face_mask_model.get())

        # Run the model (ignoring output)
        with NocatchOutput():
            mask, withoutMask = mask_detection_model.predict(face)[0]

        # Convert from np.bool_ to bool to later be able to serialize the result
        return bool(mask > withoutMask)


class NocatchOutput(ipywidgets.Output):
    """An output container that suppresses output, but not exceptions

    Taken from https://github.com/jupyter-widgets/ipywidgets/issues/3208#issuecomment-1070836153
    """

    def __exit__(self, *args, **kwargs):
        super().__exit__(*args, **kwargs)


if __name__ == "__main__":
    files = utils.find_files(
        path="/home/inga/projects/misinformation-project/misinformation/data/test_no_text/"
    )
    # files = [
    # "/home/inga/projects/misinformation-project/misinformation/data/test_no_text/102141_1_eng.png"
    # ]
    mydict = utils.initialize_dict(files)
    image_ids = [key for key in mydict.keys()]
    for i in image_ids:
        mydict[i] = EmotionDetector(mydict[i]).analyse_image()
    print(mydict)
