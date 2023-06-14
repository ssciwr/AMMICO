import os
import ntpath
from PIL import Image
from matplotlib.patches import ConnectionPatch
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# use this function to visualize the matches
def plot_matches(img1, img2, keypoints1, keypoints2):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # draw images
    axes[0].imshow(img1)
    axes[1].imshow(img2)

    # draw matches
    for kp1, kp2 in zip(keypoints1, keypoints2):
        c = np.random.rand(3)
        con = ConnectionPatch(
            xyA=kp1,
            coordsA=axes[0].transData,
            xyB=kp2,
            coordsB=axes[1].transData,
            color=c,
        )
        fig.add_artist(con)
        axes[0].plot(*kp1, color=c, marker="x")
        axes[1].plot(*kp2, color=c, marker="x")

    plt.show()


# use this function to visualize the matches from sift


def draw_matches(matches, img1, img2, kp1, kp2):
    MIN_MATCH_COUNT = 4

    if len(matches) > MIN_MATCH_COUNT:
        # Estimate homography between template and scene
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]

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

        # Combine the two images side by side
        for i in range(3):
            newimg[hdif : hdif + h1, :w1, i] = img1
            newimg[:h2, w1 : w1 + w2, i] = img2

        # Draw SIFT keypoint matches
        for m in matches:
            pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1] + hdif))
            pt2 = (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1]))
            cv2.line(newimg, pt1, pt2, (255, 0, 0))

        # Display the resulting image with keypoint matches
        plt.imshow(newimg)
        plt.show()
    else:
        print("Not enough matches are found - %d/%d" % (len(matches), MIN_MATCH_COUNT))


# compute matches from sift


def matching_points(img1, img2):
    # Convert images to grayscale
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Create SIFT object and compute keypoints and descriptors
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Convert descriptors to float32
    des1 = np.float32(des1)
    des2 = np.float32(des2)

    # Initialize and use FLANN for matching keypoints
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    filtered_matches = []
    for m, n in matches:
        # Apply ratio test to filter out ambiguous matches
        if m.distance < 0.7 * n.distance:
            filtered_matches.append(m)

    # Uncomment the line below to visualize the matches using draw_matches function
    # draw_matches(filtered_matches, img1, img2, kp1, kp2)

    return filtered_matches, kp1, kp2


# extract match points from matches


def kp_from_matches(
    matches: list[cv2.DMatch], kp1: list[cv2.KeyPoint], kp2: list[cv2.KeyPoint]
) -> tuple[np.ndarray, np.ndarray]:
    kp1_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    kp2_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
    return kp1_pts, kp2_pts


def compute_crop_corner(
    matches: list[cv2.DMatch],
    kp1: np.ndarray,
    kp2: np.ndarray,
    region: int = 30,
    h_margin: int = 28,
    v_margin: int = 5,
    min_match: int = 6,
):
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

    # Determine the y-coordinate with the maximum number of keypoints
    kp_id = ys[covers.argmax()]
    v = int(kp_id) - v_margin if int(kp_id) > v_margin else int(kp_id)

    hs = []

    # Find the minimum x-coordinate within the region around the selected y-coordinate
    for kp in kp2:
        if 0 <= kp[1] - v <= region:
            hs.append(kp[0])

    h = int(np.min(hs)) - h_margin if int(np.min(hs)) > h_margin else 0

    return v, h


# crop the posts image


def crop_posts_image(
    ref_view,
    view,
    plt_match: bool = False,
    plt_crop: bool = False,
    correct_margin: int = 700,
):
    # Compute the filtered matches and keypoints between reference view and posts image
    filtered_matches, kp1, kp2 = matching_points(ref_view, view)

    MIN_MATCH_COUNT = 6
    if len(filtered_matches) < MIN_MATCH_COUNT:
        return None

    if plt_match:
        # Convert reference view and posts image to grayscale
        img1 = cv2.cvtColor(ref_view, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)

        # Plot the matches between the grayscale images
        draw_matches(filtered_matches, img1, img2, kp1, kp2)

    # Compute the crop corner coordinates
    corner = compute_crop_corner(filtered_matches, kp1, kp2)
    if corner is None:
        return None
    v, h = corner

    # Adjust the horizontal coordinate to ensure sufficient margin
    if view.shape[1] - h > correct_margin:
        h = view.shape[1] - ref_view.shape[1]
    if view.shape[1] - h < ref_view.shape[1]:
        h = view.shape[1] - ref_view.shape[1]

    # Crop the posts image based on the computed corner coordinates
    crop_view = view[0:v, h:, :]

    if plt_crop:
        # Mark the crop boundary with red color in the original posts image
        view[v, :, 0:3] = [255, 0, 0]
        view[:, h, 0:3] = [255, 0, 0]
        plt.imshow(view)
        plt.show()

        # Display the cropped view
        plt.imshow(crop_view)
        plt.show()

    return crop_view, len(filtered_matches)


def get_file_list(
    dir: str, filelist: list[str], ext: str = None, convert_unix: bool = True
) -> list[str]:
    if os.path.isfile(dir):
        # If dir is a file path
        if ext is None:
            # If no extension is specified, include the file in the list
            filelist.append(dir)
        else:
            # If an extension is specified, check if the file matches the extension and include it in the list
            if ext in dir[-3:]:
                filelist.append(dir)

    elif os.path.isdir(dir):
        # If dir is a directory
        for s in os.listdir(dir):
            new_dir = os.path.join(dir, s)
            # Recursively call get_file_list to retrieve files from the subdirectories
            get_file_list(new_dir, filelist, ext)

    if convert_unix:
        # Convert file paths to Unix-style if convert_unix is True
        new_filelist = []
        for file_ in filelist:
            file_ = file_.replace("\\", "/")
            new_filelist.append(file_)
        return new_filelist
    else:
        return filelist


def crop_posts_from_refs(
    ref_views: list[str], view: str, plt_match: bool = False, plt_crop: bool = False
):
    crop_view = None
    max_matches = 0

    for ref_view in ref_views:
        rte = crop_posts_image(ref_view, view, plt_match=plt_match, plt_crop=plt_crop)
        if rte is not None:
            crop_img, match_num = rte
            if match_num > max_matches:
                crop_view = crop_img
                max_matches = match_num
                # print("match_num = ", match_num)

    return crop_view


def crop_posts_from_files(
    ref_dir: str,
    crop_dir: str,
    save_crop_dir: str,
    plt_match: bool = False,
    plt_crop: bool = False,
) -> None:
    ref_list = []
    ref_list = get_file_list(ref_dir, ref_list, ext="png")

    ref_views = []
    for ref_file in ref_list:
        ref_view = np.array(Image.open(ref_file))
        ref_views.append(ref_view)

    crop_list = []
    crop_list = get_file_list(crop_dir, crop_list, ext="png")

    for crop_file in crop_list:
        view = np.array(Image.open(crop_file))
        crop_view = crop_posts_from_refs(
            ref_views, view, plt_match=plt_match, plt_crop=plt_crop
        )
        if crop_view is not None:
            filename = ntpath.basename(crop_file)
            save_path = os.path.join(save_crop_dir, filename)
            save_path = save_path.replace("\\", "/")
            cv2.imwrite(save_path, crop_view)


def test_crop_from_file() -> None:
    # Load images
    view1 = np.array(Image.open("data/ref/ref-06.png"))
    view2 = np.array(Image.open("data/napsa/102956_eng.png"))
    crop_view, _ = crop_posts_image(view1, view2, plt_match=True, plt_crop=True)
    cv2.imwrite("data/crop_100489_ind.png", crop_view)


def test_crop_from_folder():
    ref_dir = "./data/ref"
    crop_dir = "./data/apsa"
    save_crop_dir = "data/crop"
    crop_posts_from_files(
        ref_dir, crop_dir, save_crop_dir, plt_match=False, plt_crop=False
    )
