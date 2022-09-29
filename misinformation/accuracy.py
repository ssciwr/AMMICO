import pandas as pd
import json

from misinformation import utils
from misinformation import faces


class LabelManager:
    def __init__(self):
        self.labels_code = None
        self.labels = None
        self.f_labels = None
        self.f_labels_code = None
        self.load()

    def load(self):
        self.labels_code = pd.read_excel(
            "./misinformation/test/data/EUROPE_APRMAY20_data_variable_labels_coding.xlsx",
            sheet_name="variable_labels_codings",
        )
        self.labels = pd.read_csv(
            "./misinformation/test/data/Europe_APRMAY20data190722.csv",
            sep=",",
            decimal=".",
        )
        self.map = self.read_json("./misinformation/data/map_test_set.json")

    def read_json(self, name):
        with open("{}".format(name)) as f:
            mydict = json.load(f)
        return mydict

    def get_orders(self):
        return [i["order"] for i in self.map.values()]

    def filter_from_order(self, orders: list):
        cols = []
        for order in orders:
            col = self.labels_code.iloc[order - 1, 1]
            cols.append(col.lower())

        self.f_labels_code = self.labels_code.loc[
            self.labels_code["order"].isin(orders)
        ]
        self.f_labels = self.labels[cols]

    def gen_dict(self):
        labels_dict = {}
        if self.f_labels is None:
            print("No filtered labels found")
            return labels_dict

        cols = self.f_labels.columns.tolist()
        for index, row in self.f_labels.iterrows():
            row_dict = {}
            for col in cols:
                row_dict[col] = row[col]
            labels_dict[row["pic_id"]] = row_dict

        return labels_dict


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
    lm = LabelManager()
    # get the desired label numbers automatically
    orders = lm.get_orders()
    lm.filter_from_order([1, 2, 3] + orders)
    # map the output to our output - or the other way around?

    labels = lm.gen_dict()
    # print(labels)
