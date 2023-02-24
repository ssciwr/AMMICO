import os
import pytest
from torch import device, cuda
from lavis.models import load_model_and_preprocess
import misinformation.summary as sm

TEST_IMAGE_1 = "./test/data/d755771b-225e-432f-802e-fb8dc850fff7.png"
TEST_IMAGE_2 = "./test/data/IMG_2746.png"
TEST_IMAGE_3 = "./test/data/IMG_2750.png"
TEST_IMAGE_4 = "./test/data/IMG_2805.png"
TEST_IMAGE_5 = "./test/data/IMG_2806.png"
TEST_IMAGE_6 = "./test/data/IMG_2807.png"
TEST_IMAGE_7 = "./test/data/IMG_2808.png"
TEST_IMAGE_8 = "./test/data/IMG_2809.png"
TEST_IMAGE_9 = "./test/data/IMG_3755.jpg"
TEST_IMAGE_10 = "./test/data/IMG_3756.jpg"
TEST_IMAGE_11 = "./test/data/IMG_3757.jpg"
TEST_IMAGE_12 = "./test/data/pic1.png"


def test_analyse_image():
    images = [
        TEST_IMAGE_1,
        TEST_IMAGE_2,
        TEST_IMAGE_3,
        TEST_IMAGE_4,
        TEST_IMAGE_5,
        TEST_IMAGE_6,
        TEST_IMAGE_7,
        TEST_IMAGE_8,
        TEST_IMAGE_9,
        TEST_IMAGE_10,
        TEST_IMAGE_11,
        TEST_IMAGE_12,
    ]
    mydict = {}
    for img_path in images:
        id_ = os.path.splitext(os.path.basename(img_path))[0]
        mydict[id_] = {"filename": img_path}

    summary_device = device("cuda" if cuda.is_available() else "cpu")
    summary_model, summary_vis_processors, _ = load_model_and_preprocess(
        name="blip_caption",
        model_type="base_coco",
        is_eval=True,
        device=summary_device,
    )

    for key in mydict:
        mydict[key] = sm.SummaryDetector(mydict[key]).analyse_image()
    keys = list(mydict.keys())
    assert len(mydict) == 12
    for key in keys:
        assert len(mydict[key]["3_non-deterministic summary"]) == 3

    assert mydict[keys[0]]["const_image_summary"] == str(
        "a river running through a city next to tall buildings"
    )
    assert mydict[keys[1]]["const_image_summary"] == str(
        "a crowd of people standing on top of a tennis court"
    )
    assert mydict[keys[2]]["const_image_summary"] == str(
        "a crowd of people standing on top of a field"
    )
    assert mydict[keys[3]]["const_image_summary"] == str(
        "a room with a desk and a chair"
    )
    assert mydict[keys[4]]["const_image_summary"] == str(
        "a table with plastic containers on top of it"
    )
    assert mydict[keys[5]]["const_image_summary"] == str(
        "a view of a city with mountains in the background"
    )
    assert mydict[keys[6]]["const_image_summary"] == str(
        "a view of a city street from a window"
    )
    assert mydict[keys[7]]["const_image_summary"] == str(
        "a busy city street with cars and pedestrians"
    )
    assert mydict[keys[8]]["const_image_summary"] == str(
        "a close up of an open book with writing on it"
    )
    assert mydict[keys[9]]["const_image_summary"] == str(
        "a book that is open on a table"
    )
    assert mydict[keys[10]]["const_image_summary"] == str(
        "a yellow book with green lettering on it"
    )
    assert mydict[keys[11]]["const_image_summary"] == str(
        "a person running on a beach near a rock formation"
    )

    for key in mydict:
        mydict[key] = sm.SummaryDetector(mydict[key]).analyse_image(
            summary_model, summary_vis_processors
        )
    keys = list(mydict.keys())
    assert len(mydict) == 12
    for key in keys:
        assert len(mydict[key]["3_non-deterministic summary"]) == 3

    assert mydict[keys[0]]["const_image_summary"] == str(
        "a river running through a city next to tall buildings"
    )
    assert mydict[keys[1]]["const_image_summary"] == str(
        "a crowd of people standing on top of a tennis court"
    )
    assert mydict[keys[2]]["const_image_summary"] == str(
        "a crowd of people standing on top of a field"
    )
    assert mydict[keys[3]]["const_image_summary"] == str(
        "a room with a desk and a chair"
    )
    assert mydict[keys[4]]["const_image_summary"] == str(
        "a table with plastic containers on top of it"
    )
    assert mydict[keys[5]]["const_image_summary"] == str(
        "a view of a city with mountains in the background"
    )
    assert mydict[keys[6]]["const_image_summary"] == str(
        "a view of a city street from a window"
    )
    assert mydict[keys[7]]["const_image_summary"] == str(
        "a busy city street with cars and pedestrians"
    )
    assert mydict[keys[8]]["const_image_summary"] == str(
        "a close up of an open book with writing on it"
    )
    assert mydict[keys[9]]["const_image_summary"] == str(
        "a book that is open on a table"
    )
    assert mydict[keys[10]]["const_image_summary"] == str(
        "a yellow book with green lettering on it"
    )
    assert mydict[keys[11]]["const_image_summary"] == str(
        "a person running on a beach near a rock formation"
    )

    summary_model, summary_vis_processors, _ = load_model_and_preprocess(
        name="blip_caption",
        model_type="large_coco",
        is_eval=True,
        device=summary_device,
    )

    for key in mydict:
        mydict[key] = sm.SummaryDetector(mydict[key]).analyse_image(
            summary_model, summary_vis_processors
        )
    keys = list(mydict.keys())
    assert len(mydict) == 12
    for key in keys:
        assert len(mydict[key]["3_non-deterministic summary"]) == 3

    assert mydict[keys[0]]["const_image_summary"] == str(
        "a river running through a town next to tall buildings"
    )
    assert mydict[keys[1]]["const_image_summary"] == str(
        "a crowd of people standing on top of a track"
    )
    assert mydict[keys[2]]["const_image_summary"] == str(
        "a group of people standing on top of a track"
    )
    assert mydict[keys[3]]["const_image_summary"] == str(
        "a desk and chair in a small room"
    )
    assert mydict[keys[4]]["const_image_summary"] == str(
        "a table that has some chairs on top of it"
    )
    assert mydict[keys[5]]["const_image_summary"] == str(
        "a view of a city from a window of a building"
    )
    assert mydict[keys[6]]["const_image_summary"] == str(
        "a view of a city from a window"
    )
    assert mydict[keys[7]]["const_image_summary"] == str(
        "a city street filled with lots of traffic"
    )
    assert mydict[keys[8]]["const_image_summary"] == str(
        "an open book with german text on it"
    )
    assert mydict[keys[9]]["const_image_summary"] == str(
        "a close up of a book on a table"
    )
    assert mydict[keys[10]]["const_image_summary"] == str(
        "a book with a green cover on a table"
    )
    assert mydict[keys[11]]["const_image_summary"] == str(
        "a person running on a beach near the ocean"
    )


def test_analyse_questions():
    images = [
        TEST_IMAGE_1,
        TEST_IMAGE_2,
        TEST_IMAGE_3,
        TEST_IMAGE_4,
        TEST_IMAGE_5,
        TEST_IMAGE_6,
        TEST_IMAGE_7,
        TEST_IMAGE_8,
        TEST_IMAGE_9,
        TEST_IMAGE_10,
        TEST_IMAGE_11,
        TEST_IMAGE_12,
    ]
    mydict = {}
    for img_path in images:
        id_ = os.path.splitext(os.path.basename(img_path))[0]
        mydict[id_] = {"filename": img_path}

    summary_device = device("cuda" if cuda.is_available() else "cpu")
    (
        summary_VQA_model,
        summary_VQA_vis_processors,
        summary_VQA_txt_processors,
    ) = load_model_and_preprocess(
        name="blip_vqa", model_type="vqav2", is_eval=True, device=summary_device
    )
    list_of_questions = [
        "How many persons on the picture?",
        "What happends on the picture?",
    ]
    for key in mydict:
        mydict[key] = sm.SummaryDetector(mydict[key]).analyse_questions(
            list_of_questions
        )

    keys = list(mydict.keys())
    assert len(mydict) == 12

    assert mydict[keys[0]][list_of_questions[0]] == str(2)
    assert mydict[keys[1]][list_of_questions[0]] == str(100)
    assert mydict[keys[2]][list_of_questions[0]] == str("many")
    assert mydict[keys[3]][list_of_questions[0]] == str(0)
    assert mydict[keys[4]][list_of_questions[0]] == str(0)
    assert mydict[keys[5]][list_of_questions[0]] == str("none")
    assert mydict[keys[6]][list_of_questions[0]] == str("two")
    assert mydict[keys[7]][list_of_questions[0]] == str(5)
    assert mydict[keys[8]][list_of_questions[0]] == str(0)
    assert mydict[keys[9]][list_of_questions[0]] == str(0)
    assert mydict[keys[10]][list_of_questions[0]] == str(0)
    assert mydict[keys[11]][list_of_questions[0]] == str(1)

    assert mydict[keys[0]][list_of_questions[1]] == str("flood")
    assert mydict[keys[1]][list_of_questions[1]] == str("festival")
    assert mydict[keys[2]][list_of_questions[1]] == str("people are flying kites")
    assert mydict[keys[3]][list_of_questions[1]] == str("no one's home")
    assert mydict[keys[4]][list_of_questions[1]] == str("chair is being moved")
    assert mydict[keys[5]][list_of_questions[1]] == str("traffic jam")
    assert mydict[keys[6]][list_of_questions[1]] == str("day time")
    assert mydict[keys[7]][list_of_questions[1]] == str("traffic jam")
    assert mydict[keys[8]][list_of_questions[1]] == str("nothing")
    assert mydict[keys[9]][list_of_questions[1]] == str("nothing")
    assert mydict[keys[10]][list_of_questions[1]] == str("nothing")
    assert mydict[keys[11]][list_of_questions[1]] == str("running")
