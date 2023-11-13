import os
import ntpath
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Union


MIN_MATCH_COUNT = 6
FLANN_INDEX_KDTREE = 1


# use this function to visualize the matches from sift
def draw_matches(
    matches: List,
    img1: np.ndarray,
    img2: np.ndarray,
    kp1: List[cv2.KeyPoint],
    kp2: List[cv2.KeyPoint],
) -> None:
    """Visualize the matches from SIFT.

    Args:
        matches (list[cv2.Match]): List of cv2.Match matches on the image.
        img1 (np.ndarray): The reference image.
        img2 (np.ndarray): The social media post.
        kp1 (list[cv2.KeyPoint]): List of keypoints from the first image.
        kp2 (list[cv2.KeyPoint]): List of keypoints from the second image.
    """
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


def matching_points(
    img1: np.ndarray, img2: np.ndarray
) -> Tuple[cv2.DMatch, List[cv2.KeyPoint], List[cv2.KeyPoint]]:
    """Computes keypoint matches using the SIFT algorithm between two images.

    Args:
        img1 (np.ndarray): The reference image.
        img2 (np.ndarray): The social media post.
    Returns:
        cv2.DMatch: List of filtered keypoint matches.
        cv2.KeyPoint: List of keypoints from the first image.
        cv2.KeyPoint: List of keypoints from the second image.
    """
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Convert descriptors to float32
    des1 = np.float32(des1)
    des2 = np.float32(des2)
    # Initialize and use FLANN
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    filtered_matches = []
    for m, n in matches:
        # Apply ratio test to filter out ambiguous matches
        if m.distance < 0.7 * n.distance:
            filtered_matches.append(m)
    return filtered_matches, kp1, kp2


def kp_from_matches(matches, kp1: np.ndarray, kp2: np.ndarray) -> Tuple[Tuple, Tuple]:
    """Extract the match indices from the keypoints.

    Args:
        kp1 (np.ndarray): Key points of the matches,
        kp2 (np.ndarray):  Key points of the matches,
    Returns:
        tuple: Index of the descriptor in the list of train descriptors.
        tuple: index of the descriptor in the list of query descriptors.
    """
    kp1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    kp2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return kp1, kp2


def compute_crop_corner(
    matches: cv2.DMatch,
    kp1: np.ndarray,
    kp2: np.ndarray,
    region: int = 30,
    h_margin: int = 0,
    v_margin: int = 5,
    min_match: int = 6,
) -> Optional[Tuple[int, int]]:
    """Estimate the position on the image from where to crop.

    Args:
        matches (cv2.DMatch): The matched objects on the image.
        kp1 (np.ndarray): Key points of the matches for the reference image.
        kp2 (np.ndarray): Key points of the matches for the social media posts.
        region (int, optional): Area to consider around the keypoints.
            Defaults to 30.
        h_margin (int, optional): Horizontal margin to subtract from the minimum
            horizontal position. Defaults to 0.
        v_margin (int, optional): Vertical margin to subtract from the minimum
            vertical position. Defaults to 5.
        min_match: Minimum number of matches required. Defaults to 6.
    Returns:
        tuple, optional: Tuple of vertical and horizontal crop corner coordinates.
    """
    kp1, kp2 = kp_from_matches(matches, kp1, kp2)
    ys = kp2[:, 1]
    covers = []

    # Compute the number of keypoints within the region around each y-coordinate
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

    # Find the minimum x-coordinate within the region around the selected y-coordinate
    for kp in kp2:
        if 0 <= kp[1] - v <= region:
            hs.append(kp[0])
    # do not use margin if h < image width/2, else use margin
    h = int(np.min(hs)) - h_margin if int(np.min(hs)) > h_margin else 0
    return v, h


def crop_posts_image(
    ref_view: List,
    view: np.ndarray,
) -> Union[None, Tuple[np.ndarray, int, int, int]]:
    """Crop the social media post to exclude additional comments. Sometimes also crops the
    image part of the post - this is put back in later.

    Args:
        ref_views (list): List of all the reference images (as numpy arrays) that signify
            below which regions should be cropped.
        view (np.ndarray): The image to crop.
    Returns:
        np.ndarray: The cropped social media post.
    """
    filtered_matches, kp1, kp2 = matching_points(ref_view, view)
    if len(filtered_matches) < MIN_MATCH_COUNT:
        # not enough matches found
        # print("Found too few matches - {}".format(filtered_matches))
        return None
    corner = compute_crop_corner(filtered_matches, kp1, kp2)
    if corner is None:
        # no cropping corner found
        # print("Found no corner")
        return None
    v, h = corner
    # if the match is on the right-hand side of the image,
    # it is likely that there is an image to the left
    # that should not be cropped
    # in this case, we adjust the margin for the text to be
    # cropped to `correct_margin`
    # if the match is more to the left on the image, we assume
    # it starts at horizontal position 0 and do not want to
    # cut off any characters from the text
    correct_margin = 30
    if h >= view.shape[1] / 2:
        h = h - correct_margin
    else:
        h = 0
    crop_view = view[0:v, h:, :]
    return crop_view, len(filtered_matches), v, h


def crop_posts_from_refs(
    ref_views: List,
    view: np.ndarray,
    plt_match: bool = False,
    plt_crop: bool = False,
    plt_image: bool = False,
) -> np.ndarray:
    """Crop the social media post comments from the image.

    Args:
        ref_views (list): List of all the reference images (as numpy arrays) that signify
            below which regions should be cropped.
        view (np.ndarray): The image to crop.
    Returns:
        np.ndarray: The cropped social media post.
    """
    crop_view = None
    # initialize the number of found matches per reference to zero
    # so later we can select the reference with the most matches
    max_matchs = 0
    rte = None
    found_match = False
    for ref_view in ref_views:
        rte = crop_posts_image(ref_view, view)
        if rte is not None:
            crop_img, match_num, v, h = rte
            if match_num > max_matchs:
                # find the reference with the most matches to crop accordingly
                crop_view = crop_img
                final_ref = ref_view
                final_v = v
                final_h = h
                max_matchs = match_num
                found_match = True

    if found_match and plt_match:
        # plot the match
        filtered_matches, kp1, kp2 = matching_points(final_ref, view)
        img1 = cv2.cvtColor(final_ref, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
        draw_matches(filtered_matches, img1, img2, kp1, kp2)

    if found_match and plt_crop:
        # plot the cropped image
        view2 = view.copy()
        view2[final_v, :, 0:3] = [255, 0, 0]
        view2[:, final_h, 0:3] = [255, 0, 0]
        plt.imshow(cv2.cvtColor(view2, cv2.COLOR_BGR2RGB))
        plt.show()
        plt.imshow(cv2.cvtColor(crop_view, cv2.COLOR_BGR2RGB))
        plt.show()

    if found_match and final_h >= view.shape[1] / 2:
        # here it would crop the actual image from the social media post
        # to avoid this, we check the position from where it would crop
        # if > than half of the width of the image, also keep all that is
        # on the left-hand side of the crop
        crop_post = crop_image_from_post(view, final_h)
        if plt_image:
            # plot the image part of the social media post
            plt.imshow(cv2.cvtColor(crop_post, cv2.COLOR_BGR2RGB))
            plt.show()
        # now concatenate the image and the text part
        crop_view = paste_image_and_comment(crop_post, crop_view)
    return crop_view


def crop_image_from_post(view: np.ndarray, final_h: int) -> np.ndarray:
    """Crop the image part from the social media post.

    Args:
        view (np.ndarray): The image to be cropped.
        final_h: The horizontal position up to which should be cropped.
    Returns:
        np.ndarray: The cropped image part."""
    crop_post = view[:, 0:final_h, :]
    return crop_post


def paste_image_and_comment(crop_post: np.ndarray, crop_view: np.ndarray) -> np.ndarray:
    """Paste the image part and the text part together without the unecessary comments.

    Args:
        crop_post (np.ndarray): The cropped image part of the social media post.
        crop_view (np.ndarray): The cropped text part of the social media post.
    Returns:
        np.ndarray: The image and text part of the social media post in one image."""
    h1, w1 = crop_post.shape[:2]
    h2, w2 = crop_view.shape[:2]
    image_all = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    image_all[:h1, :w1, :3] = crop_post
    image_all[:h2, w1 : w1 + w2, :3] = crop_view
    return image_all


def crop_media_posts(
    files, ref_files, save_crop_dir, plt_match=False, plt_crop=False, plt_image=False
) -> None:
    """Crop social media posts so that comments beyond the first comment/post are cut off.

    Args:
        files (list): List of all the files to be cropped.
        ref_files (list): List of all the reference images that signify
            below which regions should be cropped.
        save_crop_dir (str): Directory where to write the cropped social media posts to.
        plt_match (Bool, optional): Display the matched areas on the social media post.
            Defaults to False.
        plt_crop (Bool, optional): Display the cropped text part of the social media post.
            Defaults to False.
        plt_image (Bool, optional): Display the image part of the social media post.
            Defaults to False.
    """

    # get the reference images with regions that signify areas to crop
    ref_views = []
    for ref_file in ref_files.values():
        ref_file_path = ref_file["filename"]
        ref_view = cv2.imread(ref_file_path)
        ref_views.append(ref_view)
    # parse through the social media posts to be cropped
    for crop_file in files.values():
        crop_file_path = crop_file["filename"]
        view = cv2.imread(crop_file_path)
        print("Doing file {}".format(crop_file_path))
        crop_view = crop_posts_from_refs(
            ref_views,
            view,
            plt_match=plt_match,
            plt_crop=plt_crop,
            plt_image=plt_image,
        )
        if crop_view is not None:
            # save the image to the provided folder
            filename = ntpath.basename(crop_file_path)
            save_path = os.path.join(save_crop_dir, filename)
            save_path = save_path.replace("\\", "/")
            cv2.imwrite(save_path, crop_view)
