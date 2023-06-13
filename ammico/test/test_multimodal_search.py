import pytest
import math
from PIL import Image
import numpy
from torch import device, cuda
import ammico.multimodal_search as ms

related_error = 1e-2
gpu_is_not_available = not cuda.is_available()

cuda.empty_cache()


def test_read_img(get_testdict):
    my_dict = {}
    test_img = ms.MultimodalSearch.read_img(
        my_dict, get_testdict["IMG_2746"]["filename"]
    )
    assert list(numpy.array(test_img)[257][34]) == [70, 66, 63]


pre_proc_pic_blip2_blip_albef = [
    -1.0039474964141846,
    -1.0039474964141846,
]
pre_proc_pic_clip_vitl14 = [
    -0.7995694875717163,
    -0.7849710583686829,
]

pre_proc_pic_clip_vitl14_336 = [
    -0.7995694875717163,
    -0.7849710583686829,
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
]

pre_extracted_feature_img_blip = [
    -0.02480311505496502,
    0.05037587881088257,
]

pre_extracted_feature_img_albef = [
    0.08971136063337326,
    -0.10915573686361313,
]

pre_extracted_feature_img_clip = [
    0.01621132344007492,
    -0.004035486374050379,
]

pre_extracted_feature_img_parsing_clip = [
    0.01621132344007492,
    -0.004035486374050379,
]

pre_extracted_feature_img_clip_vitl14 = [
    -0.023943455889821053,
    -0.021703708916902542,
]

pre_extracted_feature_img_clip_vitl14_336 = [
    -0.009511193260550499,
    -0.012618942186236382,
]

pre_extracted_feature_text_blip2 = [
    -0.1384204626083374,
    -0.008662976324558258,
]

pre_extracted_feature_text_blip = [
    0.0118643119931221,
    -0.01291718054562807,
]

pre_extracted_feature_text_albef = [
    -0.06229640915989876,
    0.11278597265481949,
]

pre_extracted_feature_text_clip = [
    0.018169036135077477,
    0.03634127229452133,
]

pre_extracted_feature_text_clip_vitl14 = [
    -0.0055463071912527084,
    0.006908962037414312,
]

pre_extracted_feature_text_clip_vitl14_336 = [
    -0.008720514364540577,
    0.005284308455884457,
]

simularity_blip2 = [
    [0.05826476216316223, -0.02717375010251999],
    [0.06297147274017334, 0.47339022159576416],
]

sorted_blip2 = [
    [1, 0],
    [1, 0],
]

simularity_blip = [
    [0.15640679001808167, 0.752173662185669],
    [0.17233705520629883, 0.8448910117149353],
]

sorted_blip = [
    [1, 0],
    [1, 0],
]

simularity_albef = [
    [0.12321824580430984, 0.35511350631713867],
    [0.10870333760976791, 0.5143978595733643],
]

sorted_albef = [
    [0, 1],
    [1, 0],
]

simularity_clip = [
    [0.23923014104366302, 0.5325412750244141],
    [0.2310466319322586, 0.5910375714302063],
]

sorted_clip = [
    [1, 0],
    [1, 0],
]

simularity_clip_vitl14 = [
    [0.1051270067691803, 0.5184808373451233],
    [0.1277746558189392, 0.6841973662376404],
]

sorted_clip_vitl14 = [
    [1, 0],
    [1, 0],
]

simularity_clip_vitl14_336 = [
    [0.09391091763973236, 0.49337542057037354],
    [0.13700757920742035, 0.7003108263015747],
]

sorted_clip_vitl14_336 = [
    [1, 0],
    [1, 0],
]

dict_itm_scores_for_blib = {
    "blip_base": [
        0.07107225805521011,
        0.004100032616406679,
    ],
    "blip_large": [
        0.07890705019235611,
        0.00271016638725996,
    ],
    "blip2_coco": [
        0.0833505243062973,
        0.004216152708977461,
    ],
}

dict_image_gradcam_with_itm_for_blip = {
    "blip_base": [123.36285799741745, 132.31662154197693, 53.38280035299249],
    "blip_large": [119.99512910842896, 128.7044593691826, 55.552959859540515],
}


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
    get_path,
    get_testdict,
    tmp_path,
):
    ms.MultimodalSearch.multimodal_device = pre_multimodal_device
    my_obj = ms.MultimodalSearch(get_testdict)
    (
        model,
        vis_processor,
        txt_processor,
        image_keys,
        _,
        features_image_stacked,
    ) = my_obj.parsing_images(pre_model, path_to_save_tensors=tmp_path)

    for i, num in zip(range(10), features_image_stacked[0, 10:12].tolist()):
        assert (
            math.isclose(num, pre_extracted_feature_img[i], rel_tol=related_error)
            is True
        )

    test_pic = Image.open(my_obj.subdict["IMG_2746"]["filename"]).convert("RGB")
    test_querry = (
        "The bird sat on a tree located at the intersection of 23rd and 43rd streets."
    )
    processed_pic = (
        vis_processor["eval"](test_pic).unsqueeze(0).to(pre_multimodal_device)
    )
    processed_text = txt_processor["eval"](test_querry)

    for i, num in zip(range(10), processed_pic[0, 0, 0, 25:27].tolist()):
        assert math.isclose(num, pre_proc_pic[i], rel_tol=related_error) is True

    assert processed_text == pre_proc_text

    search_query = [
        {"text_input": test_querry},
        {"image": my_obj.subdict["IMG_2746"]["filename"]},
    ]
    multi_features_stacked = my_obj.querys_processing(
        search_query, model, txt_processor, vis_processor, pre_model
    )

    for i, num in zip(range(10), multi_features_stacked[0, 10:12].tolist()):
        assert (
            math.isclose(num, pre_extracted_feature_text[i], rel_tol=related_error)
            is True
        )

    for i, num in zip(range(10), multi_features_stacked[1, 10:12].tolist()):
        assert (
            math.isclose(num, pre_extracted_feature_img[i], rel_tol=related_error)
            is True
        )

    search_query2 = [
        {"text_input": "A bus"},
        {"image": get_path + "IMG_3758.png"},
    ]

    similarity, sorted_list = my_obj.multimodal_search(
        model,
        vis_processor,
        txt_processor,
        pre_model,
        image_keys,
        features_image_stacked,
        search_query2,
    )

    for i, num in zip(range(len(pre_simularity)), similarity.tolist()):
        for j, num2 in zip(range(len(num)), num):
            assert (
                math.isclose(num2, pre_simularity[i][j], rel_tol=100 * related_error)
                is True
            )

    for i, num in zip(range(len(pre_sorted)), sorted_list):
        for j, num2 in zip(range(2), num):
            assert num2 == pre_sorted[i][j]

    del (
        model,
        vis_processor,
        txt_processor,
        similarity,
        features_image_stacked,
        processed_pic,
        multi_features_stacked,
        my_obj,
    )
    cuda.empty_cache()


@pytest.mark.long
def test_itm(get_test_my_dict, get_path):
    search_query3 = [
        {"text_input": "A bus"},
        {"image": get_path + "IMG_3758.png"},
    ]
    image_keys = ["IMG_2746", "IMG_2809"]
    sorted_list = [[1, 0], [1, 0]]
    my_obj = ms.MultimodalSearch(get_test_my_dict)
    for itm_model in ["blip_base", "blip_large"]:
        (
            itm_scores,
            image_gradcam_with_itm,
        ) = my_obj.image_text_match_reordering(
            search_query3,
            itm_model,
            image_keys,
            sorted_list,
            batch_size=1,
            need_grad_cam=True,
        )
        for i, itm in zip(
            range(len(dict_itm_scores_for_blib[itm_model])),
            dict_itm_scores_for_blib[itm_model],
        ):
            assert (
                math.isclose(itm_scores[0].tolist()[i], itm, rel_tol=10 * related_error)
                is True
            )
        for i, grad_cam in zip(
            range(len(dict_image_gradcam_with_itm_for_blip[itm_model])),
            dict_image_gradcam_with_itm_for_blip[itm_model],
        ):
            assert (
                math.isclose(
                    image_gradcam_with_itm["A bus"]["IMG_2809"][0][0].tolist()[i],
                    grad_cam,
                    rel_tol=10 * related_error,
                )
                is True
            )
        del itm_scores, image_gradcam_with_itm
        cuda.empty_cache()


@pytest.mark.long
def test_itm_blip2_coco(get_test_my_dict, get_path):
    search_query3 = [
        {"text_input": "A bus"},
        {"image": get_path + "IMG_3758.png"},
    ]
    image_keys = ["IMG_2746", "IMG_2809"]
    sorted_list = [[1, 0], [1, 0]]
    my_obj = ms.MultimodalSearch(get_test_my_dict)

    (
        itm_scores,
        image_gradcam_with_itm,
    ) = my_obj.image_text_match_reordering(
        search_query3,
        "blip2_coco",
        image_keys,
        sorted_list,
        batch_size=1,
        need_grad_cam=False,
    )
    for i, itm in zip(
        range(len(dict_itm_scores_for_blib["blip2_coco"])),
        dict_itm_scores_for_blib["blip2_coco"],
    ):
        assert (
            math.isclose(itm_scores[0].tolist()[i], itm, rel_tol=10 * related_error)
            is True
        )
    del itm_scores, image_gradcam_with_itm
    cuda.empty_cache()
