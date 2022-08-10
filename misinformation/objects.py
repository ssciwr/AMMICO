from misinformation.utils import AnalysisMethod
from google.cloud import vision
import cv2
import cvlib as cv
from imageai.Detection import ObjectDetection
import os


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


def objects_from_imageai(detections: dict) -> dict:
    objects = init_default_objects()
    for obj in detections:
        obj_name = obj["name"]
        objects[obj_name] = "yes"
    return objects


class ObjectDetector(AnalysisMethod):
    # Using cvlib as client
    CLIENT_CVLIB = 1
    # Using imageai as client
    CLIENT_IMAGEAI = 2

    # client_type: 1 using cvlib, 2 using imageai, 3 using google vision (disabled)
    def __init__(self, client_type=1):
        # init google vision client
        self.gv_client = vision.ImageAnnotatorClient()

        # init imageai client
        execution_path = os.getcwd()
        self.imgai_client = ObjectDetection()
        self.imgai_client.setModelTypeAsRetinaNet()
        # default model path is ./misinformation/model/resnet50_coco_best_v2.0.1.h5
        misinformation_path = os.path.join(execution_path, "misinformation")
        model_path = os.path.join(misinformation_path, "model")
        model_path = os.path.join(model_path, "resnet50_coco_best_v2.0.1.h5")
        self.imgai_client.setModelPath(model_path)
        self.imgai_client.loadModel()
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

        support_type = [ObjectDetector.CLIENT_CVLIB, ObjectDetector.CLIENT_IMAGEAI]
        assert client_type in support_type

        self.client_type = client_type

    def set_client_type(self, client_type):
        support_type = [ObjectDetector.CLIENT_CVLIB, ObjectDetector.CLIENT_IMAGEAI]
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

    def analyse_image(self, image_path):
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
