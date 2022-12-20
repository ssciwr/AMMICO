import json
import pytest
import misinformation.objects as ob
import misinformation.objects_cvlib as ob_cvlib

OBJECT_1 = "cell phone"
OBJECT_2 = "motorcycle"
OBJECT_3 = "traffic light"
TEST_IMAGE_1 = "./test/data/IMG_2809.png"
JSON_1 = "./test/data/example_objects_cvlib.json"
JSON_2 = "./test/data/example_objects_imageai.json"


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


def test_analyse_image_cvlib():
    mydict = {"filename": TEST_IMAGE_1}
    ob_cvlib.ObjectCVLib().analyse_image(mydict)

    with open(JSON_1, "r") as file:
        out_dict = json.load(file)
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


def test_analyse_image_from_file_cvlib():
    file_path = TEST_IMAGE_1
    objs = ob_cvlib.ObjectCVLib().analyse_image_from_file(file_path)

    with open(JSON_1, "r") as file:
        out_dict = json.load(file)
    for key in objs.keys():
        assert objs[key] == out_dict[key]


def test_detect_objects_cvlib():
    file_path = TEST_IMAGE_1
    objs = ob_cvlib.ObjectCVLib().detect_objects_cvlib(file_path)

    with open(JSON_1, "r") as file:
        out_dict = json.load(file)
    for key in objs.keys():
        assert objs[key] == out_dict[key]


@pytest.mark.imageai
def test_objects_from_imageai(default_objects):
    objects_list = [OBJECT_1, OBJECT_2, OBJECT_3]
    objs_input = [
        {"name": OBJECT_1},
        {"name": OBJECT_2},
        {"name": OBJECT_3},
    ]
    objects = ob_iai.objects_from_imageai(objs_input)  # noqa: F821
    out_objects = default_objects
    for obj in objects_list:
        out_objects[obj] = "yes"

    assert str(objects) == str(out_objects)


@pytest.mark.imageai
def test_analyse_image_from_file_imageai():
    file_path = TEST_IMAGE_1
    objs = ob_iai.ObjectImageAI().analyse_image_from_file(file_path)  # noqa: F821

    with open(JSON_2, "r") as file:
        out_dict = json.load(file)
    for key in objs.keys():
        assert objs[key] == out_dict[key]


@pytest.mark.imageai
def test_detect_objects_imageai():
    file_path = TEST_IMAGE_1
    objs = ob_iai.ObjectImageAI().detect_objects_imageai(file_path)  # noqa: F821

    with open(JSON_2, "r") as file:
        out_dict = json.load(file)
    for key in objs.keys():
        assert objs[key] == out_dict[key]


@pytest.mark.imageai
def test_analyse_image_imageai():
    mydict = {"filename": TEST_IMAGE_1}
    ob_iai.ObjectImageAI().analyse_image(mydict)  # noqa: F821
    with open(JSON_2, "r") as file:
        out_dict = json.load(file)
    for key in mydict.keys():
        assert mydict[key] == out_dict[key]


def test_set_keys(default_objects):
    mydict = {"filename": TEST_IMAGE_1}
    key_objs = ob.ObjectDetector(mydict).set_keys()
    assert str(default_objects) == str(key_objs)


def test_analyse_image():
    mydict = {"filename": TEST_IMAGE_1}
    ob.ObjectDetector.set_client_to_cvlib()
    ob.ObjectDetector(mydict).analyse_image()
    with open(JSON_1, "r") as file:
        out_dict = json.load(file)

    assert str(mydict) == str(out_dict)
