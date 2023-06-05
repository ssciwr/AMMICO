import os
import ntpath
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ammico import utils
from typing import List


# use this function to visualize the matches from sift
def draw_matches(
    matches: List, img1: np.ndarray, img2: np.ndarray, kp1: tuple, kp2: tuple
) -> None:
    """Visualize the matches from SIFT."""
    MIN_MATCH_COUNT = 4
    if len(matches) > MIN_MATCH_COUNT:
        # Estimate homography between template and scene
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]
        if not isinstance(M, np.ndarray):
            print("Could not match images for drawing.")
            return

        # Draw detected template in scene image
        h = img1.shape[0]
        w = img1.shape[1]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
            -1, 1, 2
        )
        dst = cv2.perspectiveTransform(pts, M)

        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        h1 = img1.shape[0]
        h2 = img2.shape[0]
        w1 = img1.shape[1]
        w2 = img2.shape[1]
        nwidth = w1 + w2
        nheight = max(h1, h2)
        hdif = int((h2 - h1) / 2)
        newimg = np.zeros((nheight, nwidth, 3), np.uint8)

        for i in range(3):
            newimg[hdif : hdif + h1, :w1, i] = img1
            newimg[:h2, w1 : w1 + w2, i] = img2

        # Draw SIFT keypoint matches
        for m in matches:
            pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1] + hdif))
            pt2 = (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1]))
            cv2.line(newimg, pt1, pt2, (255, 0, 0))

        plt.imshow(newimg)
        plt.show()
    else:
        print("Not enough matches are found - %d/%d" % (len(matches), MIN_MATCH_COUNT))


# compute matches from sift
def matching_points(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    des1 = np.float32(des1)
    des2 = np.float32(des2)
    # Initialize and use FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    print(type(kp1))
    filtered_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            filtered_matches.append(m)

    return filtered_matches, kp1, kp2


# extract match points from matches
def kp_from_matches(matches, kp1, kp2):
    kp1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    kp2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return kp1, kp2


# estimate a crop corner for posts image via matches
def compute_crop_corner(
    # matches, kp1, kp2, region=30, h_margin=28, v_margin=5, min_match=6
    matches,
    kp1,
    kp2,
    region=30,
    # h_margin=28,
    h_margin=0,
    v_margin=5,
    min_match=6,
):
    kp1, kp2 = kp_from_matches(matches, kp1, kp2)
    ys = kp2[:, 1]
    covers = []
    for y in ys:
        ys_c = ys - y
        series = pd.Series(ys_c)
        is_between = series.between(0, region)
        covers.append(is_between.sum())

    covers = np.array(covers)
    if covers.max() < min_match:
        return None

    kp_id = ys[covers.argmax()]
    v = int(kp_id) - v_margin if int(kp_id) > v_margin else int(kp_id)
    hs = []
    for kp in kp2:
        if 0 <= kp[1] - v <= region:
            hs.append(kp[0])

    # do not use margin if h < image width/2, else use margin
    h = int(np.min(hs)) - h_margin if int(np.min(hs)) > h_margin else 0

    return v, h


# crop the posts image
def crop_posts_image(
    # ref_view, view, plt_match=False, plt_crop=False, correct_margin=700
    ref_view,
    view,
):
    """
    get file lists from dir and sub dirs


    ref_view： ref_view for crop the posts images
    view: posts image that need cropping
    rte： None - not cropped, or (crop_view, number of matches)
    """
    filtered_matches, kp1, kp2 = matching_points(ref_view, view)
    MIN_MATCH_COUNT = 6
    if len(filtered_matches) < MIN_MATCH_COUNT:
        # print("Found too few matches - {}".format(filtered_matches))
        return None

    corner = compute_crop_corner(filtered_matches, kp1, kp2)
    if corner is None:
        # print("Found no corner")
        return None
    v, h = corner

    correct_margin = 30
    if h >= view.shape[1] / 2:
        h = h - correct_margin
    else:
        h = 0

    crop_view = view[0:v, h:, :]

    return crop_view, len(filtered_matches), v, h


def crop_posts_from_refs(
    ref_views, view, plt_match=False, plt_crop=False, plt_image=False
):
    crop_view = None
    max_matchs = 0
    rte = None
    found_match = False
    print(type(view))
    for ref_view in ref_views:
        rte = crop_posts_image(ref_view, view)
        if rte is not None:
            crop_img, match_num, v, h = rte
            if match_num > max_matchs:
                crop_view = crop_img
                final_ref = ref_view
                final_v = v
                final_h = h
                max_matchs = match_num  # find the one with the most matches
                found_match = True

    # plot only the one with the most matches
    if found_match and plt_match:
        # now plot the match
        filtered_matches, kp1, kp2 = matching_points(final_ref, view)
        img1 = cv2.cvtColor(final_ref, cv2.COLOR_BGR2GRAY)
        print(type(img1))
        img2 = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
        draw_matches(filtered_matches, img1, img2, kp1, kp2)

    if found_match and plt_crop:
        view2 = view.copy()
        # now plot the cropped image
        view2[final_v, :, 0:3] = [255, 0, 0]
        view2[:, final_h, 0:3] = [255, 0, 0]
        plt.imshow(cv2.cvtColor(view2, cv2.COLOR_BGR2RGB))
        plt.show()

        plt.imshow(cv2.cvtColor(crop_view, cv2.COLOR_BGR2RGB))
        plt.show()

    # show posted image only if crop is in the left column
    if found_match and final_h >= view.shape[1] / 2:
        crop_post = crop_image_from_post(view, final_h)
        if plt_image:
            plt.imshow(cv2.cvtColor(crop_post, cv2.COLOR_BGR2RGB))
            plt.show()
        # now concatenate the two images
        crop_view = paste_image_and_comment(crop_post, crop_view)

    return crop_view


def crop_image_from_post(view, final_h):
    crop_post = view[:, 0:final_h, :]
    return crop_post


def paste_image_and_comment(crop_post, crop_view):
    h1, w1 = crop_post.shape[:2]
    h2, w2 = crop_view.shape[:2]
    image_all = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    image_all[:h1, :w1, :3] = crop_post
    image_all[:h2, w1 : w1 + w2, :3] = crop_view
    return image_all


def crop_media_posts(
    files, ref_files, save_crop_dir, plt_match=False, plt_crop=False, plt_image=False
):
    """Crop social media posts so that comments are cut off.

    Args:
        files (list): List of all the files to be cropped.
        ref_files (list): List of all the reference images that signify
            which regions should be cropped.
        save_crop_dir (str): Directory where to write the cropped images to.
        plt_match (Bool, optional): Display the matched areas on the image.
            Defaults to False.
        plt_crop (Bool, optional): Display the cropped image.
            Defaults to False.
    """

    ref_views = []
    for ref_file in ref_files:
        ref_view = cv2.imread(ref_file)
        ref_views.append(ref_view)

    for crop_file in files:
        view = cv2.imread(crop_file)
        print("Doing file {}".format(crop_file))
        crop_view = crop_posts_from_refs(
            ref_views,
            view,
            plt_match=plt_match,
            plt_crop=plt_crop,
            plt_image=plt_image,
        )
        if crop_view is not None:
            filename = ntpath.basename(crop_file)
            save_path = os.path.join(save_crop_dir, filename)
            save_path = save_path.replace("\\", "/")
            cv2.imwrite(save_path, crop_view)


if __name__ == "__main__":
    files = utils.find_files(
        path="data/test-debug/examples cropped/examples original/",
        limit=100,
    )
    ref_files = utils.find_files(path="data/ref", limit=100)
    # 103395_eng, 10071_tur, 109363_spa, 106740, 102355_eng, 102284, 104150, 102131, 103163, 106797
    crop_media_posts(
        files, ref_files, "data/crop/", plt_match=True, plt_crop=True, plt_image=True
    )
    print("done")
