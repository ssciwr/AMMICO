import json
import pandas as pd
import misinformation.utils as ut


def test_find_files():
    result = ut.find_files(
        path="./test/data/", pattern="*.png", recursive=True, limit=10
    )
    assert len(result) > 0


def test_initialize_dict():
    result = [
        "./test/data/image_faces.jpg",
        "./test/data/image_objects.jpg",
    ]
    mydict = ut.initialize_dict(result)
    with open("./test/data/example_utils_init_dict.json", "r") as file:
        out_dict = json.load(file)
    assert mydict == out_dict


def test_append_data_to_dict():
    with open("./test/data/example_append_data_to_dict_in.json", "r") as file:
        mydict = json.load(file)
    outdict = ut.append_data_to_dict(mydict)
    print(outdict)
    with open("./test/data/example_append_data_to_dict_out.json", "r") as file:
        example_outdict = json.load(file)

    assert outdict == example_outdict


def test_dump_df():
    with open("./test/data/example_append_data_to_dict_out.json", "r") as file:
        outdict = json.load(file)
    df = ut.dump_df(outdict)
    out_df = pd.read_csv("./test/data/example_dump_df.csv", index_col=[0])
    pd.testing.assert_frame_equal(df, out_df)
