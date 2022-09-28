from misinformation import utils
from misinformation import faces


if __name__ == "__main__":
    files = utils.find_files(
        path="/home/inga/projects/misinformation-project/misinformation/misinformation/test/data/Europe APRMAY20 visual data/cropped images"
    )
    mydict = utils.initialize_dict(files)
    outdict = {}
    outdict = utils.append_data_to_dict(mydict)
    # analyze faces
    image_ids = [key for key in mydict.keys()]
    for i in image_ids:
        mydict[i] = faces.EmotionDetector(mydict[i]).analyse_image()

    df = utils.dump_df(outdict)
    print(df.head(10))

    # example of LabelManager for loading csv data to dict
    lm = utils.LabelManager()
    # get the desired label numbers automatically
    orders = lm.get_orders()
    lm.filter_from_order([1, 2, 3] + orders)
    # map the output to our output - or the other way around?

    labels = lm.gen_dict()
    print(labels)
