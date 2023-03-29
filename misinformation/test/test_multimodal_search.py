import pytest
import math
from PIL import Image
import numpy
from torch import device, cuda
import misinformation.multimodal_search as ms
from memory_profiler import profile

testdict = {
    "d755771b-225e-432f-802e-fb8dc850fff7": {
        "filename": "./test/data/d755771b-225e-432f-802e-fb8dc850fff7.png"
    },
    "IMG_2746": {"filename": "./test/data/IMG_2746.png"},
    "IMG_2750": {"filename": "./test/data/IMG_2750.png"},
    "IMG_2805": {"filename": "./test/data/IMG_2805.png"},
    "IMG_2806": {"filename": "./test/data/IMG_2806.png"},
    "IMG_2807": {"filename": "./test/data/IMG_2807.png"},
    "IMG_2808": {"filename": "./test/data/IMG_2808.png"},
    "IMG_2809": {"filename": "./test/data/IMG_2809.png"},
    "IMG_3755": {"filename": "./test/data/IMG_3755.jpg"},
    "IMG_3756": {"filename": "./test/data/IMG_3756.jpg"},
    "IMG_3757": {"filename": "./test/data/IMG_3757.jpg"},
    "pic1": {"filename": "./test/data/pic1.png"},
}

related_error = 1e-2
gpu_is_not_available = not cuda.is_available()


cuda.empty_cache()


@profile
@pytest.mark.long
def test_read_img():
    my_dict = {}
    test_img = ms.MultimodalSearch.read_img(my_dict, testdict["IMG_2746"]["filename"])
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

simularity_blip2 = [
    [0.05826476216316223, -0.02717375010251999],
    [0.12869958579540253, 0.006344856694340706],
    [0.11073512583971024, 0.12327021360397339],
    [0.08743024617433548, 0.058944884687662125],
    [0.04591086134314537, 0.4905201494693756],
    [0.06297147274017334, 0.47339022159576416],
    [0.18486255407333374, 0.6350338459014893],
    [0.015455856919288635, 0.018462061882019043],
    [-0.008606988936662674, 0.00741103570908308],
    [-0.0415784977376461, -0.1267213076353073],
    [-0.025470387190580368, 0.1315656304359436],
    [-0.05090826004743576, 0.059172093868255615],
]

sorted_blip2 = [
    [6, 1, 2, 3, 5, 0, 4, 7, 8, 10, 9, 11],
    [6, 4, 5, 10, 2, 11, 3, 7, 8, 1, 0, 9],
]

simularity_blip = [
    [0.15640679001808167, 0.752173662185669],
    [0.15139800310134888, 0.7804810404777527],
    [0.13010388612747192, 0.755257248878479],
    [0.13746635615825653, 0.7618774175643921],
    [0.1756758838891983, 0.8531903624534607],
    [0.17233705520629883, 0.8448910117149353],
    [0.1970970332622528, 0.8916105628013611],
    [0.11693969368934631, 0.5833531618118286],
    [0.12386563420295715, 0.5981853604316711],
    [0.08427951484918594, 0.4962371587753296],
    [0.14193706214427948, 0.7613846659660339],
    [0.12051936239004135, 0.6492202281951904],
]

sorted_blip = [
    [6, 4, 5, 0, 1, 10, 3, 2, 8, 11, 7, 9],
    [6, 4, 5, 1, 3, 10, 2, 0, 11, 8, 7, 9],
]

simularity_albef = [
    [0.12321824580430984, 0.35511350631713867],
    [0.09512615948915482, 0.27168408036231995],
    [0.09053325653076172, 0.20215675234794617],
    [0.06335515528917313, 0.15055638551712036],
    [0.09604836255311966, 0.4658776521682739],
    [0.10870333760976791, 0.5143978595733643],
    [0.11748822033405304, 0.6542638540267944],
    [0.05688793584704399, 0.22170542180538177],
    [0.05597608536481857, 0.11963296681642532],
    [0.059643782675266266, 0.14969395101070404],
    [0.06690303236246109, 0.3149859607219696],
    [0.07909377664327621, 0.11911341547966003],
]

sorted_albef = [
    [0, 6, 5, 4, 1, 2, 11, 10, 3, 9, 7, 8],
    [6, 5, 4, 0, 10, 1, 7, 2, 3, 9, 8, 11],
]

simularity_clip = [
    [0.23923014104366302, 0.5325412750244141],
    [0.20101115107536316, 0.5112978219985962],
    [0.17522737383842468, 0.49811851978302],
    [0.20062290132045746, 0.5415266156196594],
    [0.22865726053714752, 0.5762109756469727],
    [0.2310466319322586, 0.5910375714302063],
    [0.2644523084163666, 0.7851459383964539],
    [0.21474510431289673, 0.4135811924934387],
    [0.16407863795757294, 0.1474374681711197],
    [0.19819433987140656, 0.26493316888809204],
    [0.19545596837997437, 0.5007457137107849],
    [0.1647854745388031, 0.45705708861351013],
]

sorted_clip = [
    [6, 0, 5, 4, 7, 1, 3, 9, 10, 2, 11, 8],
    [6, 5, 4, 3, 0, 1, 10, 2, 11, 7, 9, 8],
]

simularity_clip_vitl14 = [
    [0.1051270067691803, 0.5184808373451233],
    [0.09705893695354462, 0.49574509263038635],
    [0.11964304000139236, 0.5424358248710632],
    [0.13881900906562805, 0.5909714698791504],
    [0.12728188931941986, 0.6758255362510681],
    [0.1277746558189392, 0.6841973662376404],
    [0.18026694655418396, 0.803142786026001],
    [0.13977059721946716, 0.45957139134407043],
    [0.11180847883224487, 0.24822194874286652],
    [0.12296056002378464, 0.35143694281578064],
    [0.11596094071865082, 0.5704031586647034],
    [0.10174489766359329, 0.44422751665115356],
]

sorted_clip_vitl14 = [
    [6, 7, 3, 5, 4, 9, 2, 10, 8, 0, 11, 1],
    [6, 5, 4, 3, 10, 2, 0, 1, 7, 11, 9, 8],
]

simularity_clip_vitl14_336 = [
    [0.09391091763973236, 0.49337542057037354],
    [0.11103834211826324, 0.4881117343902588],
    [0.12891019880771637, 0.5501476526260376],
    [0.13288410007953644, 0.5498673915863037],
    [0.12357455492019653, 0.6749162077903748],
    [0.13700757920742035, 0.7003108263015747],
    [0.1788637489080429, 0.7713702321052551],
    [0.13260436058044434, 0.4300197660923004],
    [0.11666625738143921, 0.2334875613451004],
    [0.1316065937280655, 0.3291645646095276],
    [0.12374477833509445, 0.5632147192955017],
    [0.10333051532506943, 0.43023794889450073],
]

sorted_clip_vitl14_336 = [
    [6, 5, 3, 7, 9, 2, 10, 4, 8, 1, 11, 0],
    [6, 5, 4, 10, 2, 3, 0, 1, 11, 7, 9, 8],
]


@profile
@pytest.mark.long
@pytest.mark.parametrize(
    (
        "pre_multimodal_device",
        "pre_model",
        "pre_proc_pic",
        "pre_proc_text",
        "pre_extracted_feature_img",
        "pre_extracted_feature_text",
        "pre_simularity",
        "pre_sorted",
    ),
    [
        (
            device("cpu"),
            "blip2",
            pre_proc_pic_blip2_blip_albef,
            pre_proc_text_blip2_blip_albef,
            pre_extracted_feature_img_blip2,
            pre_extracted_feature_text_blip2,
            simularity_blip2,
            sorted_blip2,
        ),
        pytest.param(
            device("cuda"),
            "blip2",
            pre_proc_pic_blip2_blip_albef,
            pre_proc_text_blip2_blip_albef,
            pre_extracted_feature_img_blip2,
            pre_extracted_feature_text_blip2,
            simularity_blip2,
            sorted_blip2,
            marks=pytest.mark.skipif(
                gpu_is_not_available, reason="gpu_is_not_availible"
            ),
        ),
        (
            device("cpu"),
            "blip",
            pre_proc_pic_blip2_blip_albef,
            pre_proc_text_blip2_blip_albef,
            pre_extracted_feature_img_blip,
            pre_extracted_feature_text_blip,
            simularity_blip,
            sorted_blip,
        ),
        pytest.param(
            device("cuda"),
            "blip",
            pre_proc_pic_blip2_blip_albef,
            pre_proc_text_blip2_blip_albef,
            pre_extracted_feature_img_blip,
            pre_extracted_feature_text_blip,
            simularity_blip,
            sorted_blip,
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
            simularity_albef,
            sorted_albef,
        ),
        pytest.param(
            device("cuda"),
            "albef",
            pre_proc_pic_blip2_blip_albef,
            pre_proc_text_blip2_blip_albef,
            pre_extracted_feature_img_albef,
            pre_extracted_feature_text_albef,
            simularity_albef,
            sorted_albef,
            marks=pytest.mark.skipif(
                gpu_is_not_available, reason="gpu_is_not_availible"
            ),
        ),
        (
            device("cpu"),
            "clip_base",
            pre_proc_pic_clip_vitl14,
            pre_proc_text_clip_clip_vitl14_clip_vitl14_336,
            pre_extracted_feature_img_clip,
            pre_extracted_feature_text_clip,
            simularity_clip,
            sorted_clip,
        ),
        pytest.param(
            device("cuda"),
            "clip_base",
            pre_proc_pic_clip_vitl14,
            pre_proc_text_clip_clip_vitl14_clip_vitl14_336,
            pre_extracted_feature_img_clip,
            pre_extracted_feature_text_clip,
            simularity_clip,
            sorted_clip,
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
            simularity_clip_vitl14,
            sorted_clip_vitl14,
        ),
        pytest.param(
            device("cuda"),
            "clip_vitl14",
            pre_proc_pic_clip_vitl14,
            pre_proc_text_clip_clip_vitl14_clip_vitl14_336,
            pre_extracted_feature_img_clip_vitl14,
            pre_extracted_feature_text_clip_vitl14,
            simularity_clip_vitl14,
            sorted_clip_vitl14,
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
            simularity_clip_vitl14_336,
            sorted_clip_vitl14_336,
        ),
        pytest.param(
            device("cuda"),
            "clip_vitl14_336",
            pre_proc_pic_clip_vitl14_336,
            pre_proc_text_clip_clip_vitl14_clip_vitl14_336,
            pre_extracted_feature_img_clip_vitl14_336,
            pre_extracted_feature_text_clip_vitl14_336,
            simularity_clip_vitl14_336,
            sorted_clip_vitl14_336,
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
    pre_simularity,
    pre_sorted,
):

    ms.MultimodalSearch.multimodal_device = pre_multimodal_device
    (
        model,
        vis_processor,
        txt_processor,
        image_keys,
        image_names,
        features_image_stacked,
    ) = ms.MultimodalSearch.parsing_images(testdict, pre_model)

    for i, num in zip(range(10), features_image_stacked[0, 10:20].tolist()):
        assert (
            math.isclose(num, pre_extracted_feature_img[i], rel_tol=related_error)
            is True
        )

    test_pic = Image.open(testdict["IMG_2746"]["filename"]).convert("RGB")
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
        {"image": testdict["IMG_2746"]["filename"]},
    ]
    multi_features_stacked = ms.MultimodalSearch.querys_processing(
        testdict, search_query, model, txt_processor, vis_processor, pre_model
    )

    for i, num in zip(range(10), multi_features_stacked[0, 10:20].tolist()):
        assert (
            math.isclose(num, pre_extracted_feature_text[i], rel_tol=related_error)
            is True
        )

    for i, num in zip(range(10), multi_features_stacked[1, 10:20].tolist()):
        assert (
            math.isclose(num, pre_extracted_feature_img[i], rel_tol=related_error)
            is True
        )

    search_query2 = [
        {"text_input": "A bus"},
        {"image": "../misinformation/test/data/IMG_3758.png"},
    ]

    similarity, sorted_list = ms.MultimodalSearch.multimodal_search(
        testdict,
        model,
        vis_processor,
        txt_processor,
        pre_model,
        image_keys,
        features_image_stacked,
        search_query2,
    )

    for i, num in zip(range(12), similarity.tolist()):
        for j, num2 in zip(range(len(num)), num):
            assert (
                math.isclose(num2, pre_simularity[i][j], rel_tol=100 * related_error)
                is True
            )

    for i, num in zip(range(2), sorted_list):
        for j, num2 in zip(range(2), num):
            assert num2 == pre_sorted[i][j]

    del model, vis_processor, txt_processor
    cuda.empty_cache()
