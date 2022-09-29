from misinformation.utils import AnalysisMethod
from misinformation.utils import DownloadResource
from google.cloud import vision
import cv2
import cvlib as cv
import os
import pathlib


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


def objects_from_cvlib(objects_list: list) -> dict:
    objects = init_default_objects()
    for key in objects:
        if key in objects_list:
            objects[key] = "yes"
    return objects


def objects_from_imageai(detections: list) -> dict:
    return print("Imageai is currently disabled")
    objects = init_default_objects()
    for obj in detections:
        obj_name = obj["name"]
        objects[obj_name] = "yes"
    return objects


def objects_symlink_processor(name):
    def _processor(fname, action, pooch):
        if not os.path.exists(os.path.dirname(name)):
            os.makedirs(os.path.dirname(name))

        if not os.path.exists(name):
            os.symlink(fname, name)
        return fname

    return _processor


pre_model_path = pathlib.Path.home().joinpath(
    ".misinformation", "objects", "resnet50_coco_best_v2.1.0.h5"
)


# retina_objects_model = DownloadResource(
#     url="https://github.com/OlafenwaMoses/ImageAI/releases/download/essentials-v5/resnet50_coco_best_v2.1.0.h5/",
#     known_hash="sha256:6518ad56a0cca4d1bd8cbba268dd4e299c7633efe7d15902d5acbb0ba180027c",
#     processor=objects_symlink_processor(pre_model_path),
# )


class ObjectDetectorClient(AnalysisMethod):
    # Using cvlib as client
    CLIENT_CVLIB = 1
    # Using imageai as client
    CLIENT_IMAGEAI = 2

    # client_type: 1 using cvlib, 2 using imageai, 3 using google vision (disabled)
    def __init__(self, client_type=1):
        # init google vision client
        self.gv_client = vision.ImageAnnotatorClient()

        # init imageai client
        # retina_objects_model.get()
        # if not os.path.exists(pre_model_path):
        #     print("Download retina objects model failed.")
        #     return
        # self.imgai_client = ObjectDetection()
        # self.imgai_client.setModelTypeAsRetinaNet()
        # self.imgai_client.setModelPath(pre_model_path)
        # self.imgai_client.loadModel()
        self.custom = self.imgai_client.CustomObjects(
            person=True,
            bicycle=True,
            car=True,
            motorcycle=True,
            airplane=True,
            bus=True,
            train=True,
            truck=True,
            boat=True,
            traffic_light=True,
            cell_phone=True,
        )

        support_type = [
            ObjectDetectorClient.CLIENT_CVLIB,
            ObjectDetectorClient.CLIENT_IMAGEAI,
        ]
        assert client_type in support_type

        self.client_type = client_type

    def set_client_type(self, client_type):
        support_type = [
            ObjectDetectorClient.CLIENT_CVLIB,
            ObjectDetectorClient.CLIENT_IMAGEAI,
        ]
        assert client_type in support_type
        self.client_type = client_type

    # need a payment account for google cloud vision, or get a free trail 90 days
    def detect_objects_google_vision(self, image_path):
        """Localize objects in the local image.

        Args:
        image_path: The path to the local file.
        """

        self.gv_client = vision.ImageAnnotatorClient()

        with open(image_path, "rb") as image_file:
            content = image_file.read()
        image = vision.Image(content=content)

        objects = self.gv_client.object_localization(
            image=image
        ).localized_object_annotations

        print("Number of objects found: {}".format(len(objects)))
        for object_ in objects:
            print("\n{} (confidence: {})".format(object_.name, object_.score))
            print("Normalized bounding polygon vertices: ")
            for vertex in object_.bounding_poly.normalized_vertices:
                print(" - ({}, {})".format(vertex.x, vertex.y))

        return objects

    def detect_objects_cvlib(self, image_path):
        """Localize objects in the local image.

        Args:
        image_path: The path to the local file.
        """
        img = cv2.imread(image_path)
        bbox, label, conf = cv.detect_common_objects(img)
        # output_image = draw_bbox(im, bbox, label, conf)
        objects = objects_from_cvlib(label)
        return objects

    def detect_objects_imageai(self, image_path, custom=True, min_prob=30):
        """Localize objects in the local image.

        Args:
        image_path: The path to the local file.
        custom: If only detect user defined specific objects.
        min_prob: Minimum probability that we trust as objects.
        """
        return print("Imageai is currently disabled due to old dependencies.")
        img = cv2.imread(image_path)
        if custom:
            box_img, detections = self.imgai_client.detectCustomObjectsFromImage(
                custom_objects=self.custom,
                input_type="array",
                input_image=img,
                output_type="array",
                minimum_percentage_probability=min_prob,
            )
        else:
            box_img, detections = self.imgai_client.detectObjectsFromImage(
                input_type="array",
                input_image=img,
                output_type="array",
                minimum_percentage_probability=min_prob,
            )
        objects = objects_from_imageai(detections)
        return objects

    def analyse_image_from_file(self, image_path):
        """Localize objects in the local image.

        Args:
        image_path: The path to the local file.
        """
        if self.client_type == 1:
            objects = self.detect_objects_cvlib(image_path)
        elif self.client_type == 2:
            objects = self.detect_objects_imageai(image_path)
        else:
            objects = init_default_objects()
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
    def set_client_type(client_type):
        ObjectDetector.od_client.set_client_type(client_type)
