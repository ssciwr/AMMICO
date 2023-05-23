from ammico.expressions import rgb2name, analyze_images
from ammico.display import show_piechart
import pandas as pd


def test_rgb2name(get_path):
    assert rgb2name([0, 0, 0]) == "black"
    assert rgb2name([255, 255, 255]) == "white"
    assert rgb2name([205, 133, 63]) == "brown"

    assert rgb2name([255, 255, 255], merge_color=False) == "white"
    assert rgb2name([0, 0, 0], merge_color=False) == "black"
    assert rgb2name([205, 133, 63], merge_color=False) == "peru"


def test_analyze_images(get_path):
    path_img_1 = get_path + "IMG_2809.png"
    path_img_2 = get_path + "IMG_2746.png"

    df_list = analyze_images([path_img_1], n_colors=10, reduce_colors=True)

    df_string = analyze_images(path_img_1, n_colors=10, reduce_colors=True)

    pd.testing.assert_frame_equal(df_list, df_string)

    df = analyze_images([path_img_1, path_img_2], n_colors=10, reduce_colors=True)
    assert df["green"].loc["sum"] == 0.4704398648321647
    assert df.shape == (3, 7)

    df = analyze_images([path_img_1, path_img_2], n_colors=10, reduce_colors=False)
    assert df["purple"].loc["sum"] == 0.029468227931013505
    assert df.shape == (3, 17)

    df = analyze_images([path_img_1, path_img_2], n_colors=2, reduce_colors=False)
    assert df.shape == (3, 4)
