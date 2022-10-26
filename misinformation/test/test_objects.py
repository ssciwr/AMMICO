import json
import pytest
import misinformation.objects as ob
import misinformation.objects_cvlib as ob_cvlib
# import misinformation.objects_imageai as ob_iai


@pytest.fixture()
def default_objects():
    return ob.init_default_objects()


def test_objects_from_cvlib(default_objects):
    objects_list = ["cell phone", "motorcycle", "traffic light"]
    objects = ob_cvlib.objects_from_cvlib(objects_list)
    out_objects = default_objects
    for obj in objects_list:
        out_objects[obj] = "yes"

    assert str(objects) == str(out_objects)


def test_analyse_image_cvlib():
    mydict = {"filename": "./test/data/IMG_2809.png"}
    ob_cvlib.ObjectCVLib().analyse_image(mydict)

    with open("./test/data/example_objects_cvlib.json", "r") as file:
        out_dict = json.load(file)
    for key in mydict.keys():
        assert mydict[key] == out_dict[key]


def test_init_default_objects():
    default_obj_list = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "cell phone",
    ]
    init_objects = ob_cvlib.init_default_objects()
    for obj in default_obj_list:
        assert init_objects[obj] == "no"


def test_analyse_image_from_file_cvlib():
    file_path = "./test/data/IMG_2809.png"
    objs = ob_cvlib.ObjectCVLib().analyse_image_from_file(file_path)

    with open("./test/data/example_objects_cvlib.json", "r") as file:
        out_dict = json.load(file)
    for key in objs.keys():
        assert objs[key] == out_dict[key]


def test_detect_objects_cvlib():
    file_path = "./test/data/IMG_2809.png"
    objs = ob_cvlib.ObjectCVLib().detect_objects_cvlib(file_path)

    with open("./test/data/example_objects_cvlib.json", "r") as file:
        out_dict = json.load(file)
    for key in objs.keys():
        assert objs[key] == out_dict[key]


@pytest.mark.imageai
def test_objects_from_imageai(default_objects):
    objects_list = ["cell phone", "motorcycle", "traffic light"]
    objs_input = [
        {"name": "cell phone"},
        {"name": "motorcycle"},
        {"name": "traffic light"},
    ]
    objects = ob_iai.objects_from_imageai(objs_input)
    out_objects = default_objects
    for obj in objects_list:
        out_objects[obj] = "yes"

    assert str(objects) == str(out_objects)


@pytest.mark.imageai
def test_analyse_image_from_file_imageai():
    file_path = "./test/data/IMG_2809.png"
    objs = ob_iai.ObjectImageAI().analyse_image_from_file(file_path)

    with open("./test/data/example_objects_imageai.json", "r") as file:
        out_dict = json.load(file)
    for key in objs.keys():
        assert objs[key] == out_dict[key]


@pytest.mark.imageai
def test_detect_objects_imageai():
    file_path = "./test/data/IMG_2809.png"
    objs = ob_iai.ObjectImageAI().detect_objects_imageai(file_path)

    with open("./test/data/example_objects_imageai.json", "r") as file:
        out_dict = json.load(file)
    for key in objs.keys():
        assert objs[key] == out_dict[key]


@pytest.mark.imageai
def test_analyse_image_imageai():
    mydict = {"filename": "./test/data/IMG_2809.png"}
    ob_iai.ObjectImageAI().analyse_image(mydict)
    with open("./test/data/example_objects_imageai.json", "r") as file:
        out_dict = json.load(file)
    for key in mydict.keys():
        assert mydict[key] == out_dict[key]


def test_set_keys(default_objects):
    mydict = {"filename": "./test/data/IMG_2809.png"}
    key_objs = ob.ObjectDetector(mydict).set_keys()
    assert str(default_objects) == str(key_objs)


def test_analyse_image():
    mydict = {"filename": "./test/data/IMG_2809.png"}
    ob.ObjectDetector.set_client_to_cvlib()
    ob.ObjectDetector(mydict).analyse_image()
    with open("./test/data/example_objects_cvlib.json", "r") as file:
        out_dict = json.load(file)

    assert str(mydict) == str(out_dict)
