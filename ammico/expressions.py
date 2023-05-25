import numpy as np
import webcolors
import pandas as pd
from collections import defaultdict

import colorgram

import colour

from ammico.utils import get_color_table


def rgb2name(c, merge_color=True):
    """This function takes a rgb color as input and returns the closest color name from the CSS3 color list.

    :param c: RGB value as list or tuple
    :type c: list or tuple
    :param merge_color: Wether color name should be reduced, defaults to True
    :type merge_color: bool, optional
    :return: color name
    :rtype: str
    """
    h_color = "#{:02x}{:02x}{:02x}".format(int(c[0]), int(c[1]), int(c[2]))
    try:
        output_color = webcolors.hex_to_name(h_color, spec="css3")
    except ValueError:
        delta_e_lst = []
        filtered_colors = webcolors.CSS3_NAMES_TO_HEX

        for img_clr, img_hex in filtered_colors.items():
            cur_clr = webcolors.hex_to_rgb(img_hex)

            # calculate color Delta-E

            delta_e = colour.delta_E(c, cur_clr, method="CIE 2000")

            delta_e_lst.append(delta_e)

        # find lowest dealta-e
        min_diff = np.argsort(delta_e_lst)[0]

        output_color = (
            str(list(filtered_colors.items())[min_diff][0])
            .lower()
            .replace("grey", "gray")
        )

    # match color to reduced list:
    if merge_color:
        for reduced_key, reduced_color_sub_list in get_color_table().items():
            if str(output_color).lower() in [
                str(color_name).lower()
                for color_name in reduced_color_sub_list["ColorName"]
            ]:
                output_color = reduced_key
                break

    return output_color


def analyze_images(image_paths, n_colors=100, reduce_colors=True):
    """This function takes a list of image paths as input and returns a dataframe with the percentage of each color in the images.
    It uses the colorgram library to extract the n most common colors from the images.
    One problem is, that the most common colors are taken before beeing categorized,
    so for small values it might occur that the ten most common colors are shades of grey,
    while other colors are present but will be ignored. Because of this n_colors=100 was chosen as default.

    The colors are then matched to the closest color in the CSS3 color list using the delta-e metric.
    They are then merged into one data frame.
    The colors can be reduced to a smaller list of colors using the get_color_table function.
    These colors are: "red", "green", "blue", "yellow","cyan", "orange", "purple", "pink", "brown", "grey", "white", "black"


    Args:
        image_paths (list): list of image paths
        n_colors (int, optional): number of colors to extract from each image.
        Note that the total amount of colors can be higher if multiple images are processed.
        Defaults to 10.
        reduce_colors (bool, optional): whether to merge the colors into a reduced color list. Defaults to True.

    """
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    # the default dict makes adding percentages easier
    merged_images = defaultdict(lambda: 0)

    for image_path in image_paths:
        colors = colorgram.extract(image_path, n_colors)
        for color in colors:
            rgb_name = rgb2name(color.rgb, merge_color=reduce_colors)
            merged_images[rgb_name] += color.proportion

    df = pd.DataFrame(merged_images, index=["sum"])

    percentage_row = pd.DataFrame(
        {color: df[color].values / df.sum(axis=1).values for color in df.columns},
        index=["percentage"],
    )
    label_row = pd.DataFrame({color: color for color in df.columns}, index=["label"])
    df = pd.concat([percentage_row, label_row, df.loc[:]])
    return df
