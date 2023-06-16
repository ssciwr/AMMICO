from ammico.utils import find_files, initialize_dict, append_data_to_dict, dump_df
from ammico.text import TextDetector


if __name__ == "__main__":
    images = find_files(path=".")
    mydict = initialize_dict(images)
    for key in mydict:
        mydict[key] = TextDetector(mydict[key], analyse_text=True).analyse_image()
    print(mydict)
    outdict = append_data_to_dict(mydict)
    df = dump_df(outdict)
    df.to_csv("data_out.csv")
