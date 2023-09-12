from ammico.colors import ColorDetector
import pytest
from numpy import isclose


def test_init():
    delta_e_method = "CIE 1976"
    cd = ColorDetector({})
    assert cd.delta_e_method == delta_e_method
    delta_e_method = "CIE 1994"
    cd = ColorDetector({}, delta_e_method)
    assert cd.delta_e_method == delta_e_method
    delta_e_method = "nonsense"
    with pytest.raises(ValueError):
        ColorDetector({}, delta_e_method)


def test_set_keys():
    colors = {
        "red": 0,
        "green": 0,
        "blue": 0,
        "yellow": 0,
        "cyan": 0,
        "orange": 0,
        "purple": 0,
        "pink": 0,
        "brown": 0,
        "grey": 0,
        "white": 0,
        "black": 0,
    }
    cd = ColorDetector({})

    for color_key, value in colors.items():
        assert cd.subdict[color_key] == value


def test_rgb2name(get_path):
    cd = ColorDetector({})

    assert cd.rgb2name([0, 0, 0]) == "black"
    assert cd.rgb2name([255, 255, 255]) == "white"
    assert cd.rgb2name([205, 133, 63]) == "brown"

    assert cd.rgb2name([255, 255, 255], merge_color=False) == "white"
    assert cd.rgb2name([0, 0, 0], merge_color=False) == "black"
    assert cd.rgb2name([205, 133, 63], merge_color=False) == "peru"

    with pytest.raises(ValueError):
        cd.rgb2name([1, 2])

    with pytest.raises(ValueError):
        cd.rgb2name([1, 2, 3, 4])


def test_analyze_images(get_path):
    mydict_1 = {
        "filename": get_path + "IMG_2809.png",
    }
    mydict_2 = {
        "filename": get_path + "IMG_2809.png",
    }

    test1 = ColorDetector(mydict_1, delta_e_method="CIE 2000").analyse_image()
    assert isclose(test1["red"], 0.0, atol=0.01)
    assert isclose(test1["green"], 0.63, atol=0.01)

    test2 = ColorDetector(mydict_2).analyse_image()
    assert isclose(test2["red"], 0.0, atol=0.01)
    assert isclose(test2["green"], 0.06, atol=0.01)

    mydict_1["test"] = "test"
    test3 = ColorDetector(mydict_1).analyse_image()
    assert isclose(test3["red"], 0.0, atol=0.01)
    assert isclose(test3["green"], 0.06, atol=0.01)
    assert test3["test"] == "test"
