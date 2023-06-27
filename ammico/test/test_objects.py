import json
import pytest
import ammico.objects as ob
import ammico.objects_cvlib as ob_cvlib
import sys

OBJECT_1 = "cell phone"
OBJECT_2 = "motorcycle"
OBJECT_3 = "traffic light"
TEST_IMAGE_1 = "IMG_2809.png"
JSON_1 = "example_objects_cvlib.json"


@pytest.fixture()
def default_objects():
    return ob.init_default_objects()


def test_objects_from_cvlib(default_objects):
    objects_list = [OBJECT_1, OBJECT_2, OBJECT_3]
    objects = ob_cvlib.objects_from_cvlib(objects_list)
    out_objects = default_objects
    for obj in objects_list:
        out_objects[obj] = "yes"

    assert str(objects) == str(out_objects)


@pytest.mark.skipif(sys.platform == "darwin", reason="segmentation fault on mac")
def test_analyse_image_cvlib(get_path):
    mydict = {"filename": get_path + TEST_IMAGE_1}
    ob_cvlib.ObjectCVLib().analyse_image(mydict)

    with open(get_path + JSON_1, "r") as file:
        out_dict = json.load(file)
    out_dict["filename"] = get_path + out_dict["filename"]
    for key in mydict.keys():
        assert mydict[key] == out_dict[key]


def test_init_default_objects():
    default_obj_list = [
        "person",
        "bicycle",
        "car",
        OBJECT_2,
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        OBJECT_3,
        OBJECT_1,
    ]
    init_objects = ob_cvlib.init_default_objects()
    for obj in default_obj_list:
        assert init_objects[obj] == "no"


@pytest.mark.skipif(sys.platform == "darwin", reason="segmentation fault on mac")
def test_analyse_image_from_file_cvlib(get_path):
    file_path = get_path + TEST_IMAGE_1
    objs = ob_cvlib.ObjectCVLib().analyse_image_from_file(file_path)

    with open(get_path + JSON_1, "r") as file:
        out_dict = json.load(file)
    out_dict["filename"] = get_path + out_dict["filename"]
    for key in objs.keys():
        assert objs[key] == out_dict[key]


# @pytest.mark.skipif(sys.platform == "darwin", reason="segmentation fault on mac")
def test_detect_objects_cvlib(get_path):
    file_path = get_path + TEST_IMAGE_1
    objs = ob_cvlib.ObjectCVLib().detect_objects_cvlib(file_path)

    with open(get_path + JSON_1, "r") as file:
        out_dict = json.load(file)
    for key in objs.keys():
        assert objs[key] == out_dict[key]


def test_set_keys(default_objects, get_path):
    mydict = {"filename": get_path + TEST_IMAGE_1}
    key_objs = ob.ObjectDetector(mydict).set_keys()
    assert str(default_objects) == str(key_objs)


# @pytest.mark.skipif(sys.platform == "darwin", reason="segmentation fault on mac")
def test_analyse_image(get_path):
    mydict = {"filename": get_path + TEST_IMAGE_1}
    ob.ObjectDetector.set_client_to_cvlib()
    ob.ObjectDetector(mydict).analyse_image()
    with open(get_path + JSON_1, "r") as file:
        out_dict = json.load(file)
    out_dict["filename"] = get_path + out_dict["filename"]

    assert str(mydict) == str(out_dict)
