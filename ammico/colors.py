import numpy as np
import webcolors
import pandas as pd
from collections import defaultdict
import colorgram
import colour
from ammico.utils import get_color_table, AnalysisMethod

COLOR_SCHEMES = [
    "CIE 1976",
    "CIE 1994",
    "CIE 2000",
    "CMC",
    "ITP",
    "CAM02-LCD",
    "CAM02-SCD",
    "CAM02-UCS",
    "CAM16-LCD",
    "CAM16-SCD",
    "CAM16-UCS",
    "DIN99",
]


class ColorDetector(AnalysisMethod):
    def __init__(
        self,
        subdict: dict,
        delta_e_method: str = "CIE 1976",
    ) -> None:
        """Color Analysis class, analyse hue and identify named colors.

        Args:
            subdict (dict): The dictionary containing the image path.
            delta_e_method (str): The calculation method used for assigning the
                closest color name, defaults to "CIE 1976".
                The available options are: 'CIE 1976', 'CIE 1994', 'CIE 2000',
                'CMC', 'ITP', 'CAM02-LCD', 'CAM02-SCD', 'CAM02-UCS', 'CAM16-LCD',
                'CAM16-SCD', 'CAM16-UCS', 'DIN99'
        """
        super().__init__(subdict)
        self.subdict.update(self.set_keys())
        self.merge_color = True
        self.n_colors = 100
        if delta_e_method not in COLOR_SCHEMES:
            raise ValueError(
                "Invalid selection for assigning the color name. Please select one of {}".format(
                    COLOR_SCHEMES
                )
            )
        self.delta_e_method = delta_e_method

    def set_keys(self) -> dict:
        colors = {
            "red": 0,
            "green": 0,
            "blue": 0,
            "yellow": 0,
            "cyan": 0,
            "orange": 0,
            "purple": 0,
            "pink": 0,
            "brown": 0,
            "grey": 0,
            "white": 0,
            "black": 0,
        }
        return colors

    def analyse_image(self):
        """
        Uses the colorgram library to extract the n most common colors from the images.
        One problem is, that the most common colors are taken before beeing categorized,
        so for small values it might occur that the ten most common colors are shades of grey,
        while other colors are present but will be ignored. Because of this n_colors=100 was chosen as default.

        The colors are then matched to the closest color in the CSS3 color list using the delta-e metric.
        They are then merged into one data frame.
        The colors can be reduced to a smaller list of colors using the get_color_table function.
        These colors are: "red", "green", "blue", "yellow","cyan", "orange", "purple", "pink", "brown", "grey", "white", "black".

        Returns:
            dict: Dictionary with color names as keys and percentage of color in image as values.
        """
        filename = self.subdict["filename"]

        colors = colorgram.extract(filename, self.n_colors)
        for color in colors:
            rgb_name = self.rgb2name(
                color.rgb,
                merge_color=self.merge_color,
                delta_e_method=self.delta_e_method,
            )
            self.subdict[rgb_name] += color.proportion

        # ensure color rounding
        for key in self.set_keys().keys():
            if self.subdict[key]:
                self.subdict[key] = round(self.subdict[key], 2)

        return self.subdict

    def rgb2name(
        self, c, merge_color: bool = True, delta_e_method: str = "CIE 1976"
    ) -> str:
        """Take an rgb color as input and return the closest color name from the CSS3 color list.

        Args:
            c (Union[List,tuple]): RGB value.
            merge_color (bool, Optional): Whether color name should be reduced, defaults to True.
        Returns:
            str: Closest matching color name.
        """
        if len(c) != 3:
            raise ValueError("Input color must be a list or tuple of length 3 (RGB).")

        h_color = "#{:02x}{:02x}{:02x}".format(int(c[0]), int(c[1]), int(c[2]))
        try:
            output_color = webcolors.hex_to_name(h_color, spec="css3")
            output_color = output_color.lower().replace("grey", "gray")
        except ValueError:
            delta_e_lst = []
            filtered_colors = webcolors._definitions._CSS3_NAMES_TO_HEX

            for _, img_hex in filtered_colors.items():
                cur_clr = webcolors.hex_to_rgb(img_hex)
                # calculate color Delta-E
                delta_e = colour.delta_E(c, cur_clr, method=delta_e_method)
                delta_e_lst.append(delta_e)
            # find lowest delta-e
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
                    output_color = reduced_key.lower()
                    break
        return output_color
