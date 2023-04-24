from ammico.utils import AnalysisMethod
from ammico.objects_cvlib import ObjectCVLib
from ammico.objects_cvlib import init_default_objects


class ObjectDetectorClient(AnalysisMethod):
    def __init__(self):
        # The detector is default to CVLib
        # Here other libraries can be added
        self.detector = ObjectCVLib()

    def set_client_to_cvlib(self):
        self.detector = ObjectCVLib()

    def analyse_image(self, subdict=None):
        """Localize objects in the local image.

        Args:
        subdict: The dictionary for an image expression instance.
        """

        return self.detector.analyse_image(subdict)


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
