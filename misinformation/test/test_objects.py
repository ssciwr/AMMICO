import json
import pytest
import misinformation
import misinformation.objects as ob
import misinformation.objects_cvlib as ob_cvlib


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
        print(key)
        assert mydict[key] == out_dict[key]
