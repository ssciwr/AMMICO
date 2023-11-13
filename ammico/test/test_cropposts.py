import ammico.cropposts as crpo
import cv2
import pytest
import numpy as np
import ammico.utils as utils


TEST_IMAGE_1 = "crop_test_files/pic1.png"
TEST_IMAGE_2 = "crop_test_ref_files/pic2.png"


@pytest.fixture
def open_images(get_path):
    ref_view = cv2.imread(get_path + TEST_IMAGE_2)
    view = cv2.imread(get_path + TEST_IMAGE_1)
    return ref_view, view


def test_matching_points(open_images):
    filtered_matches, _, _ = crpo.matching_points(open_images[0], open_images[1])
    assert len(filtered_matches) > 0


def test_kp_from_matches(open_images):
    filtered_matches, kp1, kp2 = crpo.matching_points(open_images[0], open_images[1])
    kp1, kp2 = crpo.kp_from_matches(filtered_matches, kp1, kp2)
    assert kp1.shape[0] == len(filtered_matches)
    assert kp2.shape[0] == len(filtered_matches)
    assert kp1.shape[1] == 2
    assert kp2.shape[1] == 2


def test_compute_crop_corner(open_images):
    filtered_matches, kp1, kp2 = crpo.matching_points(open_images[0], open_images[1])
    corner = crpo.compute_crop_corner(filtered_matches, kp1, kp2)
    assert corner is not None
    v, h = corner
    assert 0 <= v < open_images[1].shape[0]
    assert 0 <= h < open_images[1].shape[0]


def test_crop_posts_image(open_images):
    rte = crpo.crop_posts_image(open_images[0], open_images[1])
    assert rte is not None
    crop_view, match_num, _, _ = rte
    assert match_num > 0
    assert (
        crop_view.shape[0] * crop_view.shape[1]
        <= open_images[1].shape[0] * open_images[1].shape[1]
    )


def test_crop_posts_from_refs(open_images):
    crop_view = crpo.crop_posts_from_refs([open_images[0]], open_images[1])
    assert (
        crop_view.shape[0] * crop_view.shape[1]
        <= open_images[1].shape[0] * open_images[1].shape[1]
    )


def test_crop_image_from_post(open_images):
    crop_post = crpo.crop_image_from_post(open_images[0], 4)
    ref_array = np.array(
        [[220, 202, 155], [221, 204, 155], [221, 204, 155], [221, 204, 155]],
        dtype=np.uint8,
    )
    assert np.array_equal(crop_post[0], ref_array)


def test_paste_image_and_comment(open_images):
    full_post = crpo.paste_image_and_comment(open_images[0], open_images[1])
    ref_array1 = np.array([220, 202, 155], dtype=np.uint8)
    ref_array2 = np.array([74, 76, 64], dtype=np.uint8)
    assert np.array_equal(full_post[0, 0], ref_array1)
    assert np.array_equal(full_post[-1, -1], ref_array2)


def test_crop_media_posts(get_path, tmp_path):
    print(get_path)
    files = utils.find_files(path=get_path + "crop_test_files/")
    ref_files = utils.find_files(path=get_path + "crop_test_ref_files/")
    crpo.crop_media_posts(files, ref_files, tmp_path)
    assert len(list(tmp_path.iterdir())) == 1
    # now check that image in tmp_path is the cropped one
    filename = tmp_path / "pic1.png"
    cropped_image = cv2.imread(str(filename))
    ref = np.array([222, 205, 156], dtype=np.uint8)
    assert np.array_equal(cropped_image[0, 0], ref)
