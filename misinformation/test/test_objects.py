import os
import pytest
import misinformation
import misinformation.objects as ob


@pytest.fixture()
def default_objects():
    return ob.init_default_objects()


def test_objects_from_cvlib(default_objects):
    objects_list = ["cell phone", "motorcycle", "traffic light"]
    objects = ob.objects_from_cvlib(objects_list)
    out_objects = default_objects
    for obj in objects_list:
        out_objects[obj] = "yes"

    assert str(objects) == str(out_objects)


def test_objects_from_imageai(default_objects):
    detections = [
        {
            "name": "cell phone",
            "percentage_probability": 50.0,
            "box_points": [0, 0, 0, 0],
        },
        {
            "name": "motorcycle",
            "percentage_probability": 50.0,
            "box_points": [0, 0, 0, 0],
        },
        {
            "name": "traffic light",
            "percentage_probability": 50.0,
            "box_points": [0, 0, 0, 0],
        },
    ]
    objects = ob.objects_from_imageai(detections)
    out_objects = default_objects
    for obj in detections:
        out_objects[obj["name"]] = "yes"

    assert str(objects) == str(out_objects)


def test_analyse_image_cvlib():
    mydict = {"image_objects": {"filename": "./misinformation/test/data/IMG_2809.png"}}
    misinformation.explore_analysis(mydict, identify="objects")

    with open("./misinformation/test/data/example_objects_cvlib.txt", "r") as file:
        out_dict = file.read()
    assert str(mydict) == out_dict


def test_analyse_image_imageai(default_objects):
    mydict = {"image_objects": {"filename": "./misinformation/test/data/IMG_2809.png"}}
    ob.ObjectDetector.set_client_type(2)
    misinformation.explore_analysis(mydict, identify="objects")

    with open("./misinformation/test/data/example_objects_imageai.txt", "r") as file:
        out_dict = file.read()
    assert str(mydict) == out_dict
