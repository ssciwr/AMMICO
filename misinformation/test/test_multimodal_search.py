import os
import numpy
from torch import device, cuda
from lavis.models import load_model_and_preprocess
import misinformation.multimodal_search as ms

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


def test_read_img():
    my_dict = {}
    test_img = ms.MultimodalSearch.read_img(my_dict, TEST_IMAGE_2)
    assert list(numpy.array(test_img)[257][34]) == [70, 66, 63]


# def test_load_feature_extractor_model_blip2():
#    multimodal_device = device("cuda" if cuda.is_available() else "cpu")
#    (model, vis_processors, txt_processors,) = ms.load_feature_extractor_model_blip2(multimodal_device)
