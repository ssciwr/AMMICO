import pytest
import math
from PIL import Image
import numpy
from torch import device, cuda, no_grad
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
related_error = 1e-3
gpu_is_not_available = not cuda.is_available()


def test_read_img():
    my_dict = {}
    test_img = ms.MultimodalSearch.read_img(my_dict, TEST_IMAGE_2)
    assert list(numpy.array(test_img)[257][34]) == [70, 66, 63]


pre_proc_pic_blip2_blip_albef = [
    -1.0039474964141846,
    -1.0039474964141846,
    -0.8433647751808167,
    -0.6097899675369263,
    -0.5951915383338928,
    -0.6243883967399597,
    -0.6827820539474487,
    -0.6097899675369263,
    -0.7119789123535156,
    -1.0623412132263184,
]
pre_proc_pic_clip_vitl14 = [
    -0.7995694875717163,
    -0.7849710583686829,
    -0.7849710583686829,
    -0.7703726291656494,
    -0.7703726291656494,
    -0.7849710583686829,
    -0.7849710583686829,
    -0.7703726291656494,
    -0.7703726291656494,
    -0.7703726291656494,
]

pre_proc_pic_clip_vitl14_336 = [
    -0.7995694875717163,
    -0.7849710583686829,
    -0.7849710583686829,
    -0.7849710583686829,
    -0.7849710583686829,
    -0.7849710583686829,
    -0.7849710583686829,
    -0.9163569211959839,
    -1.149931788444519,
    -1.0039474964141846,
]

pre_proc_text_blip2_blip_albef = (
    "the bird sat on a tree located at the intersection of 23rd and 43rd streets"
)

pre_proc_text_clip_clip_vitl14_clip_vitl14_336 = (
    "The bird sat on a tree located at the intersection of 23rd and 43rd streets."
)

pre_extracted_feature_img_blip2 = [
    0.04566730558872223,
    -0.042554520070552826,
    -0.06970272958278656,
    -0.009771779179573059,
    0.01446065679192543,
    0.10173682868480682,
    0.007092420011758804,
    -0.020045937970280647,
    0.12923966348171234,
    0.006452132016420364,
]

pre_extracted_feature_img_blip = [
    -0.02480311505496502,
    0.05037587881088257,
    0.039517853409051895,
    -0.06994109600782394,
    -0.12886561453342438,
    0.047039758414030075,
    -0.11620642244815826,
    -0.003398326924070716,
    -0.07324369996786118,
    0.06994668394327164,
]

pre_extracted_feature_img_albef = [
    0.08971136063337326,
    -0.10915573686361313,
    -0.020636577159166336,
    0.048121627420186996,
    -0.05943416804075241,
    -0.129856139421463,
    -0.0034469354432076216,
    0.017888527363538742,
    -0.03284582123160362,
    -0.1037328764796257,
]

pre_extracted_feature_img_clip = [
    0.01621132344007492,
    -0.004035486374050379,
    -0.04304071143269539,
    -0.03459808602929115,
    0.016922621056437492,
    -0.025056276470422745,
    -0.04178355261683464,
    0.02165347896516323,
    -0.003224249929189682,
    0.020485712215304375,
]

pre_extracted_feature_img_parsing_clip = [
    0.01621132344007492,
    -0.004035486374050379,
    -0.04304071143269539,
    -0.03459808602929115,
    0.016922621056437492,
    -0.025056276470422745,
    -0.04178355261683464,
    0.02165347896516323,
    -0.003224249929189682,
    0.020485712215304375,
]

pre_extracted_feature_img_clip_vitl14 = [
    -0.023943455889821053,
    -0.021703708916902542,
    0.035043686628341675,
    0.019495919346809387,
    0.014351222664117813,
    -0.008634116500616074,
    0.01610446907579899,
    -0.003426523646339774,
    0.011931191198527813,
    0.0008691544644534588,
]

pre_extracted_feature_img_clip_vitl14_336 = [
    -0.009511193260550499,
    -0.012618942186236382,
    0.034754861146211624,
    0.016356879845261574,
    -0.0011549904011189938,
    -0.008054453879594803,
    0.0011990377679467201,
    -0.010806051082909107,
    0.00140204350464046,
    0.0006861367146484554,
]

pre_extracted_feature_text_blip2 = [
    -0.1384204626083374,
    -0.008662976324558258,
    0.006269007455557585,
    0.03151319921016693,
    0.060558050870895386,
    -0.03230040520429611,
    0.015861615538597107,
    -0.11856459826231003,
    -0.058296192437410355,
    0.03699290752410889,
]

pre_extracted_feature_text_blip = [
    0.0118643119931221,
    -0.01291718054562807,
    -0.0009687161073088646,
    0.01428765058517456,
    -0.05591396614909172,
    0.07386433333158493,
    -0.11475936323404312,
    0.01620068959891796,
    0.0062415082938969135,
    0.0034833091776818037,
]

pre_extracted_feature_text_albef = [
    -0.06229640915989876,
    0.11278597265481949,
    0.06628583371639252,
    0.1649140566587448,
    0.068987175822258,
    0.006291372701525688,
    0.03244050219655037,
    -0.049556829035282135,
    0.050752390176057816,
    -0.0421440489590168,
]

pre_extracted_feature_text_clip = [
    0.018169036135077477,
    0.03634127229452133,
    0.025660742074251175,
    0.009149895049631596,
    -0.035570453852415085,
    0.033126577734947205,
    -0.004808237310498953,
    -0.0031453112605959177,
    -0.02194291725754738,
    0.024019461125135422,
]

pre_extracted_feature_text_clip_vitl14 = [
    -0.0055463071912527084,
    0.006908962037414312,
    -0.019450219348073006,
    -0.018097277730703354,
    0.017567576840519905,
    -0.03828490898013115,
    -0.03781530633568764,
    -0.023951737210154533,
    0.01365653332322836,
    -0.02341713197529316,
]

pre_extracted_feature_text_clip_vitl14_336 = [
    -0.008720514364540577,
    0.005284308455884457,
    -0.021116750314831734,
    -0.018112430348992348,
    0.01685470901429653,
    -0.03517491742968559,
    -0.038612402975559235,
    -0.021867064759135246,
    0.01685977540910244,
    -0.023832324892282486,
]


@pytest.mark.parametrize(
    (
        "pre_multimodal_device",
        "pre_model",
        "pre_proc_pic",
        "pre_proc_text",
        "pre_extracted_feature_img",
        "pre_extracted_feature_text",
        "pre_extracted_feature_img2",
    ),
    [
        pytest.param(
            device("cuda"),
            "blip2",
            pre_proc_pic_blip2_blip_albef,
            pre_proc_text_blip2_blip_albef,
            pre_extracted_feature_img_blip2,
            pre_extracted_feature_text_blip2,
            pre_extracted_feature_img_blip2,
            marks=pytest.mark.skipif(
                gpu_is_not_available, reason="gpu_is_not_availible"
            ),
        ),
        (
            device("cuda"),
            "blip",
            pre_proc_pic_blip2_blip_albef,
            pre_proc_text_blip2_blip_albef,
            pre_extracted_feature_img_blip,
            pre_extracted_feature_text_blip,
            pre_extracted_feature_img_blip,
        ),
        pytest.param(
            device("cuda"),
            "blip",
            pre_proc_pic_blip2_blip_albef,
            pre_proc_text_blip2_blip_albef,
            pre_extracted_feature_img_blip,
            pre_extracted_feature_text_blip,
            pre_extracted_feature_img_blip,
            marks=pytest.mark.skipif(
                gpu_is_not_available, reason="gpu_is_not_availible"
            ),
        ),
        (
            device("cpu"),
            "albef",
            pre_proc_pic_blip2_blip_albef,
            pre_proc_text_blip2_blip_albef,
            pre_extracted_feature_img_albef,
            pre_extracted_feature_text_albef,
            pre_extracted_feature_img_albef,
        ),
        pytest.param(
            device("cuda"),
            "albef",
            pre_proc_pic_blip2_blip_albef,
            pre_proc_text_blip2_blip_albef,
            pre_extracted_feature_img_albef,
            pre_extracted_feature_text_albef,
            pre_extracted_feature_img_albef,
            marks=pytest.mark.skipif(
                gpu_is_not_available, reason="gpu_is_not_availible"
            ),
        ),
        (
            device("cpu"),
            "clip_base",
            pre_proc_pic_clip_vitl14,
            pre_proc_text_clip_clip_vitl14_clip_vitl14_336,
            pre_extracted_feature_img_parsing_clip,
            pre_extracted_feature_text_clip,
            pre_extracted_feature_img_clip,
        ),
        pytest.param(
            device("cuda"),
            "clip_base",
            pre_proc_pic_clip_vitl14,
            pre_proc_text_clip_clip_vitl14_clip_vitl14_336,
            pre_extracted_feature_img_parsing_clip,
            pre_extracted_feature_text_clip,
            pre_extracted_feature_img_clip,
            marks=pytest.mark.skipif(
                gpu_is_not_available, reason="gpu_is_not_availible"
            ),
        ),
        (
            device("cpu"),
            "clip_vitl14",
            pre_proc_pic_clip_vitl14,
            pre_proc_text_clip_clip_vitl14_clip_vitl14_336,
            pre_extracted_feature_img_clip_vitl14,
            pre_extracted_feature_text_clip_vitl14,
            pre_extracted_feature_img_clip_vitl14,
        ),
        pytest.param(
            device("cuda"),
            "clip_vitl14",
            pre_proc_pic_clip_vitl14,
            pre_proc_text_clip_clip_vitl14_clip_vitl14_336,
            pre_extracted_feature_img_clip_vitl14,
            pre_extracted_feature_text_clip_vitl14,
            pre_extracted_feature_img_clip_vitl14,
            marks=pytest.mark.skipif(
                gpu_is_not_available, reason="gpu_is_not_availible"
            ),
        ),
        (
            device("cpu"),
            "clip_vitl14_336",
            pre_proc_pic_clip_vitl14_336,
            pre_proc_text_clip_clip_vitl14_clip_vitl14_336,
            pre_extracted_feature_img_clip_vitl14_336,
            pre_extracted_feature_text_clip_vitl14_336,
            pre_extracted_feature_img_clip_vitl14_336,
        ),
        pytest.param(
            device("cuda"),
            "clip_vitl14_336",
            pre_proc_pic_clip_vitl14_336,
            pre_proc_text_clip_clip_vitl14_clip_vitl14_336,
            pre_extracted_feature_img_clip_vitl14_336,
            pre_extracted_feature_text_clip_vitl14_336,
            pre_extracted_feature_img_clip_vitl14_336,
            marks=pytest.mark.skipif(
                gpu_is_not_available, reason="gpu_is_not_availible"
            ),
        ),
    ],
)
def test_parsing_images(
    pre_multimodal_device,
    pre_model,
    pre_proc_pic,
    pre_proc_text,
    pre_extracted_feature_img,
    pre_extracted_feature_text,
    pre_extracted_feature_img2,
):
    mydict = {
        "IMG_2746": {"filename": "./test/data/IMG_2746.png"},
        "IMG_2750": {"filename": "./test/data/IMG_2750.png"},
    }
    ms.MultimodalSearch.multimodal_device = pre_multimodal_device
    (
        model,
        vis_processor,
        txt_processor,
        image_keys,
        image_names,
        features_image_stacked,
    ) = ms.MultimodalSearch.parsing_images(mydict, pre_model)

    for i, num in zip(range(10), features_image_stacked[0, 10:20].tolist()):
        assert (
            math.isclose(num, pre_extracted_feature_img[i], rel_tol=related_error)
            is True
        )

    test_pic = Image.open(TEST_IMAGE_2).convert("RGB")
    test_querry = (
        "The bird sat on a tree located at the intersection of 23rd and 43rd streets."
    )
    processed_pic = (
        vis_processor["eval"](test_pic).unsqueeze(0).to(pre_multimodal_device)
    )
    processed_text = txt_processor["eval"](test_querry)

    for i, num in zip(range(10), processed_pic[0, 0, 0, 25:35].tolist()):
        assert math.isclose(num, pre_proc_pic[i], rel_tol=related_error) is True

    assert processed_text == pre_proc_text

    search_query = [
        {"text_input": test_querry},
        {"image": TEST_IMAGE_2},
    ]
    multi_features_stacked = ms.MultimodalSearch.querys_processing(
        mydict, search_query, model, txt_processor, vis_processor, pre_model
    )

    for i, num in zip(range(10), multi_features_stacked[0, 10:20].tolist()):
        assert (
            math.isclose(num, pre_extracted_feature_text[i], rel_tol=related_error)
            is True
        )

    for i, num in zip(range(10), multi_features_stacked[1, 10:20].tolist()):
        assert (
            math.isclose(num, pre_extracted_feature_img2[i], rel_tol=related_error)
            is True
        )

    del model, vis_processor, txt_processor
    cuda.empty_cache()
