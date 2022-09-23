import os
import pytest
import misinformation
import misinformation.utils as ut


def test_find_files():
    result = ut.find_files(
        path="./misinformation/test/data/", pattern="*.png", recursive=True, limit=10
    )
    assert len(result) > 0


def test_initialize_dict():
    result = [
        "/misinformation/test/data/image_faces.jpg",
        "/misinformation/test/data/image_objects.jpg",
    ]
    mydict = ut.initialize_dict(result)
    with open("./misinformation/test/data/example_utils_init_dict.txt", "r") as file:
        out_dict = file.read()
    assert str(mydict) == str(out_dict)


def test_append_data_to_dict():
    with open(
        "./misinformation/test/data/example_append_data_to_dict_in.txt", "r"
    ) as file:
        mydict = file.read()
    mydict = eval(mydict)
    outdict = ut.append_data_to_dict(mydict)
    print(outdict)
    with open(
        "./misinformation/test/data/example_append_data_to_dict_out.txt", "r"
    ) as file:
        example_outdict = file.read()

    assert str(outdict) == example_outdict


def test_dump_df():
    with open(
        "./misinformation/test/data/example_append_data_to_dict_out.txt", "r"
    ) as file:
        outdict = file.read()
    outdict = eval(outdict)
    df = ut.dump_df(outdict)
    with open("./misinformation/test/data/example_dump_df.txt", "r") as file:
        out_df = file.read()
    assert str(df.head()) == out_df
