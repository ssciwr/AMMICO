from google.cloud import vision
import io
from misinformation import utils


class TextDetector(utils.AnalysisMethod):
    def __init__(self, subdict: dict) -> None:
        super().__init__(subdict)
        self.subdict.update(self.set_keys())

    def set_keys(self) -> dict:
        params = {
            "text": None,
            "text_language": None,
            "text_english": None,
            "text_cleaned": None,
        }
        return params

    def analyse_image(self):
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
            raise Exception(
                "{}\nFor more info on error messages, check: "
                "https://cloud.google.com/apis/design/errors".format(
                    response.error.message
                )
            )
        return self.subdict
