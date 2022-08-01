from google.cloud import vision
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
from imageai.Detection import ObjectDetection
import skimage.io
import os


class ObjectDetector:
    def __init__(self):
        # init google vision client
        self.gv_client = vision.ImageAnnotatorClient()

        # init imageai client
        execution_path = os.getcwd()
        self.imgai_client = ObjectDetection()
        self.imgai_client.setModelTypeAsRetinaNet()
        self.imgai_client.setModelPath(
            os.path.join(execution_path, "resnet50_coco_best_v2.0.1.h5")
        )
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

    # need a payment account for google cloud vision, or get a free trail 90 days
    def detect_objects_google_vision(self, image_path):
        """Localize objects in the local image.

        Args:
        path: The path to the local file.
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
        im = cv2.imread(image_path)
        bbox, label, conf = cv.detect_common_objects(im)
        # output_image = draw_bbox(im, bbox, label, conf)
        return label

    def detect_objects_imageai(self, image_path, custom=True, min_prob=30):
        execution_path = os.getcwd()
        if custom:
            detections = self.imgai_client.detectCustomObjectsFromImage(
                custom_objects=custom,
                input_image=os.path.join(execution_path, image_path),
                output_type="array",
                minimum_percentage_probability=min_prob,
            )
        else:
            detections = self.imgai_client.detectObjectsFromImage(
                input_image=os.path.join(execution_path, image_path),
                output_type="array",
                minimum_percentage_probability=min_prob,
            )

        return detections
