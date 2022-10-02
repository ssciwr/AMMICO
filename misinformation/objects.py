from misinformation.utils import AnalysisMethod
from misinformation.objects_cvlib import ObjectCVLib
from misinformation.objects_cvlib import init_default_objects

# from misinformation.objects_imageai import ObjectImageAI


class ObjectDetectorClient(AnalysisMethod):
    def __init__(self):
        # The detector is default to CVLib
        self.detector = ObjectCVLib()

    def set_client_to_imageai(self):
        # disable imageai temporarily
        # self.detector = ObjectImageAI()
        None

    def set_client_to_cvlib(self):
        self.detector = ObjectCVLib()

    def analyse_image(self, subdict):
        """Localize objects in the local image.

        Args:
        subdict: The dictionary for an image expression instance.
        """
        subdict = self.detector.analyse_image(subdict)

        return subdict


class ObjectDetector(AnalysisMethod):
    od_client = ObjectDetectorClient()

    def __init__(self, subdict: dict):
        super().__init__(subdict)
        self.subdict.update(self.set_keys())

    def set_keys(self):
        return init_default_objects()

    def analyse_image(self):
        self.subdict = ObjectDetector.od_client.analyse_image(self.subdict)
        return self.subdict

    @staticmethod
    def set_client_to_cvlib():
        ObjectDetector.od_client.set_client_to_cvlib()

    @staticmethod
    def set_client_to_imageai():
        ObjectDetector.od_client.set_client_to_imageai()
