import os
from torch import device, cuda
from lavis.models import load_model_and_preprocess
import misinformation.summary as sm

images = [
    "./test/data/d755771b-225e-432f-802e-fb8dc850fff7.png",
    "./test/data/IMG_2746.png",
    "./test/data/IMG_2750.png",
    "./test/data/IMG_2805.png",
    "./test/data/IMG_2806.png",
    "./test/data/IMG_2807.png",
    "./test/data/IMG_2808.png",
    "./test/data/IMG_2809.png",
    "./test/data/IMG_3755.jpg",
    "./test/data/IMG_3756.jpg",
    "./test/data/IMG_3757.jpg",
    "./test/data/pic1.png",
]


def test_analyse_image():
    mydict = {}
    for img_path in images:
        id_ = os.path.splitext(os.path.basename(img_path))[0]
        mydict[id_] = {"filename": img_path}

    for key in mydict:
        mydict[key] = sm.SummaryDetector(mydict[key]).analyse_image()
    keys = list(mydict.keys())
    assert len(mydict) == 12
    for key in keys:
        assert len(mydict[key]["3_non-deterministic summary"]) == 3

    const_image_summary_list = [
        "a river running through a city next to tall buildings",
        "a crowd of people standing on top of a tennis court",
        "a crowd of people standing on top of a field",
        "a room with a desk and a chair",
        "a table with plastic containers on top of it",
        "a view of a city with mountains in the background",
        "a view of a city street from a window",
        "a busy city street with cars and pedestrians",
        "a close up of an open book with writing on it",
        "a book that is open on a table",
        "a yellow book with green lettering on it",
        "a person running on a beach near a rock formation",
    ]

    for i in range(len(const_image_summary_list)):
        assert mydict[keys[i]]["const_image_summary"] == const_image_summary_list[i]

    del sm.SummaryDetector.summary_model, sm.SummaryDetector.summary_vis_processors
    cuda.empty_cache()

    summary_device = device("cuda" if cuda.is_available() else "cpu")
    summary_model, summary_vis_processors, _ = load_model_and_preprocess(
        name="blip_caption",
        model_type="base_coco",
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

    const_image_summary_list2 = [
        "a river running through a city next to tall buildings",
        "a crowd of people standing on top of a tennis court",
        "a crowd of people standing on top of a field",
        "a room with a desk and a chair",
        "a table with plastic containers on top of it",
        "a view of a city with mountains in the background",
        "a view of a city street from a window",
        "a busy city street with cars and pedestrians",
        "a close up of an open book with writing on it",
        "a book that is open on a table",
        "a yellow book with green lettering on it",
        "a person running on a beach near a rock formation",
    ]

    for i in range(len(const_image_summary_list2)):
        assert mydict[keys[i]]["const_image_summary"] == const_image_summary_list2[i]

    del summary_model, summary_vis_processors
    cuda.empty_cache()

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

    const_image_summary_list3 = [
        "a river running through a town next to tall buildings",
        "a crowd of people standing on top of a track",
        "a group of people standing on top of a track",
        "a desk and chair in a small room",
        "a table that has some chairs on top of it",
        "a view of a city from a window of a building",
        "a view of a city from a window",
        "a city street with cars and people on it",
        "an open book with german text on it",
        "a close up of a book on a table",
        "a book with a green cover on a table",
        "a person running on a beach near the ocean",
    ]

    for i in range(len(const_image_summary_list2)):
        assert mydict[keys[i]]["const_image_summary"] == const_image_summary_list3[i]


def test_analyse_questions():
    mydict = {}
    for img_path in images:
        id_ = os.path.splitext(os.path.basename(img_path))[0]
        mydict[id_] = {"filename": img_path}

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

    list_of_questions_ans = [2, 100, "many", 0, 0, "none", "two", 5, 0, 0, 0, 1]

    list_of_questions_ans2 = [
        "flood",
        "festival",
        "people are flying kites",
        "no one's home",
        "chair is being moved",
        "traffic jam",
        "day time",
        "traffic jam",
        "nothing",
        "nothing",
        "nothing",
        "running",
    ]

    for i in range(len(list_of_questions_ans)):
        assert mydict[keys[i]][list_of_questions[1]] == str(list_of_questions_ans2[i])
