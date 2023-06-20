import cv2
import cvlib as cv


def objects_from_cvlib(objects_list: list) -> dict:
    objects = init_default_objects()
    for key in objects:
        if key in objects_list:
            objects[key] = "yes"
    return objects


def init_default_objects():
    objects = {
        "person": "no",
        "bicycle": "no",
        "car": "no",
        "motorcycle": "no",
        "airplane": "no",
        "bus": "no",
        "train": "no",
        "truck": "no",
        "boat": "no",
        "traffic light": "no",
        "cell phone": "no",
    }
    return objects


class ObjectsMethod:
    """Base class to be inherited by all objects methods."""

    def __init__(self):
        # initialize in child class
        pass

    def analyse_image(self, subdict):
        raise NotImplementedError()


class ObjectCVLib(ObjectsMethod):
    def __init__(self, client_type=1):
        # as long as imageai is not activated this remains empty
        pass

    def detect_objects_cvlib(self, image_path):
        """Localize objects in the local image.

        Args:
            image_path: The path to the local file.
        """
        img = cv2.imread(image_path)

        _, label, _ = cv.detect_common_objects(img)
        objects = objects_from_cvlib(label)
        return objects

    def analyse_image_from_file(self, image_path):
        """Localize objects in the local image.

        Args:
            image_path: The path to the local file.
        """
        objects = self.detect_objects_cvlib(image_path)
        return objects

    def analyse_image(self, subdict):
        """Localize objects in the local image.

        Args:
            subdict: The dictionary for an image expression instance.
        """
        objects = self.analyse_image_from_file(subdict["filename"])
        for key in objects:
            subdict[key] = objects[key]

        return subdict
