import json
import ammico.display as ammico_display


def test_explore_analysis_faces(get_path):
    mydict = {"IMG_2746": {"filename": get_path + "IMG_2746.png"}}
    temp = ammico_display.explore_analysis(mydict, identify="faces")  # noqa
    temp = None  # noqa
    with open(get_path + "example_faces.json", "r") as file:
        outs = json.load(file)
    mydict["IMG_2746"].pop("filename", None)
    for im_key in mydict.keys():
        sub_dict = mydict[im_key]
        for key in sub_dict.keys():
            assert sub_dict[key] == outs[key]


def test_explore_analysis_objects(get_path):
    mydict = {"IMG_2809": {"filename": get_path + "IMG_2809.png"}}
    temp = ammico_display.explore_analysis(mydict, identify="objects")  # noqa
    temp = None  # noqa
    with open(get_path + "example_analysis_objects.json", "r") as file:
        outs = json.load(file)
    mydict["IMG_2809"].pop("filename", None)
    for im_key in mydict.keys():
        sub_dict = mydict[im_key]
        for key in sub_dict.keys():
            assert sub_dict[key] == outs[key]
