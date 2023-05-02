import json
import pandas as pd
import ammico.utils as ut


def test_find_files(get_path):
    result = ut.find_files(path=get_path, pattern="*.png", recursive=True, limit=10)
    assert len(result) > 0


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
