import json
import pandas as pd
import ammico.utils as ut
import pytest
import os


def test_find_files(get_path):
    with pytest.raises(FileNotFoundError):
        ut.find_files(path=".", pattern="*.png")

    result_jpg = ut.find_files(path=get_path, pattern=".jpg", recursive=True, limit=10)
    assert 0 < len(result_jpg) <= 10

    result_png = ut.find_files(path=get_path, pattern=".png", recursive=True, limit=10)
    assert 0 < len(result_png) <= 10

    result_png_jpg = ut.find_files(
        path=get_path, pattern=["png", "jpg"], recursive=True, limit=10
    )
    assert 0 < len(result_png_jpg) <= 10

    result_png_jpg = ut.find_files(
        path=get_path, pattern=["png", "jpg"], recursive=True, limit=4
    )
    assert 0 < len(result_png_jpg) <= 4

    result_png_jpg = ut.find_files(
        path=get_path, pattern=["png", "jpg"], recursive=True, limit=[2, 4]
    )
    assert 0 < len(result_png_jpg) <= 2

    one_dir_up_path = os.path.join(get_path, "..")
    with pytest.raises(FileNotFoundError):
        ut.find_files(
            path=one_dir_up_path, pattern=["png", "jpg"], recursive=False, limit=[2, 4]
        )

    result_png_jpg = ut.find_files(
        path=one_dir_up_path, pattern=["png", "jpg"], recursive=True, limit=[2, 4]
    )
    assert 0 < len(result_png_jpg) <= 2

    result_png_jpg = ut.find_files(
        path=get_path, pattern=["png", "jpg"], recursive=True, limit=None
    )
    assert 0 < len(result_png_jpg)
    result_png_jpg = ut.find_files(
        path=get_path, pattern=["png", "jpg"], recursive=True, limit=-1
    )
    assert 0 < len(result_png_jpg)

    result_png_jpg_rdm1 = ut.find_files(
        path=get_path, pattern=["png", "jpg"], recursive=True, limit=10, random_seed=1
    )
    result_png_jpg_rdm2 = ut.find_files(
        path=get_path, pattern=["png", "jpg"], recursive=True, limit=10, random_seed=2
    )
    assert result_png_jpg_rdm1 != result_png_jpg_rdm2
    assert len(result_png_jpg_rdm1) == len(result_png_jpg_rdm2)

    with pytest.raises(ValueError):
        ut.find_files(path=get_path, pattern=["png", "jpg"], recursive=True, limit=-2)
    with pytest.raises(ValueError):
        ut.find_files(
            path=get_path, pattern=["png", "jpg"], recursive=True, limit=[2, 4, 5]
        )
    with pytest.raises(ValueError):
        ut.find_files(path=get_path, pattern=["png", "jpg"], recursive=True, limit=[2])
    with pytest.raises(ValueError):
        ut.find_files(
            path=get_path, pattern=["png", "jpg"], recursive=True, limit="limit"
        )


def test_initialize_dict(get_path):
    result = [
        "./test/data/image_faces.jpg",
        "./test/data/image_objects.jpg",
    ]
    mydict = ut.initialize_dict(result)
    with open(get_path + "example_utils_init_dict.json", "r") as file:
        out_dict = json.load(file)
    assert mydict == out_dict


def test_append_data_to_dict(get_path):
    with open(get_path + "example_append_data_to_dict_in.json", "r") as file:
        mydict = json.load(file)
    outdict = ut.append_data_to_dict(mydict)
    print(outdict)
    with open(get_path + "example_append_data_to_dict_out.json", "r") as file:
        example_outdict = json.load(file)

    assert outdict == example_outdict


def test_dump_df(get_path):
    with open(get_path + "example_append_data_to_dict_out.json", "r") as file:
        outdict = json.load(file)
    df = ut.dump_df(outdict)
    out_df = pd.read_csv(get_path + "example_dump_df.csv", index_col=[0])
    pd.testing.assert_frame_equal(df, out_df)


def test_is_interactive():
    assert ut.is_interactive


def test_get_color_table():
    colors = ut.get_color_table()
    assert colors["Pink"] == {
        "ColorName": [
            "Pink",
            "LightPink",
            "HotPink",
            "DeepPink",
            "PaleVioletRed",
            "MediumVioletRed",
        ],
        "HEX": ["#FFC0CB", "#FFB6C1", "#FF69B4", "#FF1493", "#DB7093", "#C71585"],
    }
