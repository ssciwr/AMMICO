from misinformation.utils import DownloadResource
from misinformation.objects_cvlib import ObjectsMethod
from misinformation.objects_cvlib import init_default_objects
from imageai.Detection import ObjectDetection

import cv2
import os
import pathlib


def objects_from_imageai(detections: list) -> dict:
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


retina_objects_model = DownloadResource(
    url="https://github.com/OlafenwaMoses/ImageAI/releases/download/essentials-v5/resnet50_coco_best_v2.1.0.h5/",
    known_hash="sha256:6518ad56a0cca4d1bd8cbba268dd4e299c7633efe7d15902d5acbb0ba180027c",
    processor=objects_symlink_processor(pre_model_path),
)


class ObjectImageAI(ObjectsMethod):
    def __init__(self):
        # init imageai client
        retina_objects_model.get()
        if not os.path.exists(pre_model_path):
            print("Download retina objects model failed.")
            return
        self.imgai_client = ObjectDetection()
        self.imgai_client.setModelTypeAsRetinaNet()
        self.imgai_client.setModelPath(pre_model_path)
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

    def analyse_image_from_file(self, image_path):
        """Localize objects in the local image.

        Args:
        image_path: The path to the local file.
        """
        objects = self.detect_objects_imageai(image_path)
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
