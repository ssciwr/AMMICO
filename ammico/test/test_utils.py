import json
import os

import pandas as pd
import pytest

import ammico.utils as ut


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
        path=get_path, pattern=["png", "jpg"], recursive=True, limit=2, random_seed=1
    )
    result_png_jpg_rdm2 = ut.find_files(
        path=get_path, pattern=["png", "jpg"], recursive=True, limit=2, random_seed=2
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


def test_check_for_missing_keys():
    mydict = {
        "file1": {"faces": "Yes", "text_english": "Something"},
        "file2": {"faces": "No", "text_english": "Otherthing"},
    }
    # check that dict is not changed
    mydict2 = ut._check_for_missing_keys(mydict)
    assert mydict2 == mydict
    # check that dict is updated if key is missing
    mydict = {
        "file1": {"faces": "Yes", "text_english": "Something"},
        "file2": {"faces": "No"},
    }
    mydict2 = ut._check_for_missing_keys(mydict)
    assert mydict2["file2"] == {"faces": "No", "text_english": None}
    # check that dict is updated if more than one key is missing
    mydict = {"file1": {"faces": "Yes", "text_english": "Something"}, "file2": {}}
    mydict2 = ut._check_for_missing_keys(mydict)
    assert mydict2["file2"] == {"faces": None, "text_english": None}


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


def test_get_dataframe(get_path):
    with open(get_path + "example_append_data_to_dict_in.json", "r") as file:
        mydict = json.load(file)
    out_df = pd.read_csv(get_path + "example_dump_df.csv", index_col=[0])
    df = ut.get_dataframe(mydict)
    df.to_csv("data_out.csv")
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


def test_categorize_outputs():
    collected = [
        (0.0, "Summary: A cat is sitting on a mat. \n\nVQA Answers: 1. A cat."),
        (5.0, "Summary: A dog is playing with a ball.\n\nVQA Answers: 1. A ball."),
    ]

    bullets_summary, bullets_vqa = ut._categorize_outputs(
        collected, include_questions=True
    )
    assert len(bullets_summary) == 2
    assert len(bullets_vqa) == 2
    assert "- [0.000s] A cat is sitting on a mat." in bullets_summary[0]
    assert "- [0.000s] 1. A cat." in bullets_vqa[0]
    assert "- [5.000s] A dog is playing with a ball." in bullets_summary[1]
    assert "- [5.000s] 1. A ball." in bullets_vqa[1]


@pytest.mark.parametrize(
    "content, expected",
    [
        ("Summary: A cat.\n\nVQA Answers:\n1. Yes.", ("A cat.", "1. Yes.")),
        ("**Summary:** A cat.\n\n**VQA Answers:**\n1. Yes.", ("A cat.", "1. Yes.")),
        ("**Summary**: A cat.\n**VQA Answers**:\n1. Yes.", ("A cat.", "1. Yes.")),
        ("## Summary\nA cat.\n## VQA Answers\n1. Yes.", ("A cat.", "1. Yes.")),
        ("summary: A cat.\nAnswers: 1. Yes.", ("A cat.", "1. Yes.")),
        # prose without labels is kept as summary
        (
            "A lens flare.\n\nIt is an artifact.",
            ("A lens flare. It is an artifact.", None),
        ),
        # "answers" inside a sentence is not a label
        ("Summary: The answers are unclear.", ("The answers are unclear.", None)),
        # cut off before the VQA answers
        ("Summary: A cat.\n\nVQA Answers:", ("A cat.", None)),
        ("", (None, None)),
        ("   \n", (None, None)),
    ],
)
def test_parse_summary_vqa(content, expected):
    assert ut._parse_summary_vqa(content) == expected


def test_has_summary_and_vqa():
    assert ut._has_summary_and_vqa("Summary: A cat.\nVQA Answers:\n1. Yes.")
    assert not ut._has_summary_and_vqa("A cat.")
    assert not ut._has_summary_and_vqa("")


def test_categorize_outputs_skips_unusable_replies():
    collected = [
        (0.0, "Summary: A cat.\n\nVQA Answers: 1. A cat."),
        (1.0, ""),
        (2.0, "The bright light is likely a lens flare."),
        (3.0, "**Summary:** A dog.\n\n**VQA Answers:** 1. A ball."),
    ]
    with pytest.warns(RuntimeWarning, match="1 of 4 model replies were empty"):
        bullets_summary, bullets_vqa = ut._categorize_outputs(
            collected, include_questions=True
        )
    assert bullets_summary == [
        "- [0.000s] A cat.",
        "- [2.000s] The bright light is likely a lens flare.",
        "- [3.000s] A dog.",
    ]
    assert bullets_vqa == ["- [0.000s] 1. A cat.", "- [3.000s] 1. A ball."]


def test_categorize_outputs_all_unusable():
    with pytest.warns(RuntimeWarning, match="2 of 2 model replies were empty"):
        bullets_summary, bullets_vqa = ut._categorize_outputs(
            [(0.0, ""), (1.0, " ")], include_questions=True
        )
    assert bullets_summary == []
    assert bullets_vqa == []
