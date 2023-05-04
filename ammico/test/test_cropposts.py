import ammico.cropposts as crpo
import numpy as np
from PIL import Image

TEST_IMAGE_1 = "pic1.png"
TEST_IMAGE_2 = "pic2.png"


def test_matching_points(get_path):
    ref_view = np.array(Image.open(get_path + TEST_IMAGE_2))
    view = np.array(Image.open(get_path + TEST_IMAGE_1))
    filtered_matches, _, _ = crpo.matching_points(ref_view, view)
    assert len(filtered_matches) > 0


def test_kp_from_matches(get_path):
    ref_view = np.array(Image.open(get_path + TEST_IMAGE_2))
    view = np.array(Image.open(get_path + TEST_IMAGE_1))
    filtered_matches, kp1, kp2 = crpo.matching_points(ref_view, view)
    kp1, kp2 = crpo.kp_from_matches(filtered_matches, kp1, kp2)

    assert kp1.shape[0] == len(filtered_matches)
    assert kp2.shape[0] == len(filtered_matches)
    assert kp1.shape[1] == 2
    assert kp2.shape[1] == 2


def test_compute_crop_corner(get_path):
    ref_view = np.array(Image.open(get_path + TEST_IMAGE_2))
    view = np.array(Image.open(get_path + TEST_IMAGE_1))
    filtered_matches, kp1, kp2 = crpo.matching_points(ref_view, view)
    corner = crpo.compute_crop_corner(filtered_matches, kp1, kp2)
    print(view.shape)
    print(corner)
    assert corner is not None
    v, h = corner
    assert 0 <= v < view.shape[0]
    assert 0 <= h < view.shape[0]


def test_crop_posts_image(get_path):
    ref_view = np.array(Image.open(get_path + TEST_IMAGE_2))
    view = np.array(Image.open(get_path + TEST_IMAGE_1))
    rte = crpo.crop_posts_image(ref_view, view)
    assert rte is not None
    crop_view, match_num = rte
    assert match_num > 0
    assert crop_view.shape[0] * crop_view.shape[1] <= view.shape[0] * view.shape[1]


def test_crop_posts_from_refs(get_path):
    ref_view = np.array(Image.open(get_path + TEST_IMAGE_2))
    view = np.array(Image.open(get_path + TEST_IMAGE_1))
    ref_views = [ref_view]
    crop_view = crpo.crop_posts_from_refs(ref_views, view)
    assert crop_view.shape[0] * crop_view.shape[1] <= view.shape[0] * view.shape[1]


def test_get_file_list(get_path):
    ref_list = []
    ref_list = crpo.get_file_list(get_path, ref_list, ext="png")
    assert len(ref_list) > 0
