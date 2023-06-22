from ammico.colors import ColorDetector
import pytest


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

    test1 = ColorDetector(mydict_1, delta_e_method="CIE 2000").analyse_image()
    assert test1["red"] == 0.0
    assert round(test1["green"], 2) == 0.62

    test2 = ColorDetector(mydict_1).analyse_image()
    assert test2["red"] == 0.0
    assert test2["green"] == 0.05
