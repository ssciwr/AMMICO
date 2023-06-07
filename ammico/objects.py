from ammico.utils import AnalysisMethod
from ammico.objects_cvlib import ObjectCVLib
from ammico.objects_cvlib import init_default_objects


class ObjectDetectorClient(AnalysisMethod):
    def __init__(self):
        # The detector is set to CVLib by default
        self.detector = ObjectCVLib()

    def set_client_to_cvlib(self):
        """Set the object detection client to use CVLib."""
        self.detector = ObjectCVLib()

    def analyse_image(self, subdict=None):
        """Localize objects in the given image.

        Args:
            subdict (dict): The dictionary for an image expression instance.

        Returns:
            dict: The updated dictionary with object detection results.
        """
        return self.detector.analyse_image(subdict)


class ObjectDetector(AnalysisMethod):
    od_client = ObjectDetectorClient()

    def __init__(self, subdict: dict):
        super().__init__(subdict)
        self.subdict.update(self.set_keys())

    def set_keys(self):
        """Set the default object keys for analysis.

        Returns:
            dict: The dictionary with default object keys.
        """
        return init_default_objects()

    def analyse_image(self):
        """Perform object detection on the image.

        Returns:
            dict: The updated dictionary with object detection results.
        """
        self.subdict = ObjectDetector.od_client.analyse_image(self.subdict)
        return self.subdict

    @staticmethod
    def set_client_to_cvlib():
        """Set the object detection client to use CVLib."""
        ObjectDetector.od_client.set_client_to_cvlib()
