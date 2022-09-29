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

    def map_dict(self, mydict):
        mapped_dict = {}
        for id, subdict in mydict.items():
            mapped_subdict = {}
            mapped_subdict["id"] = id[0:-2]
            mapped_subdict["pic_order"] = id[-1] if id[-2] == "0" else id[-2::]
            mapped_subdict["pic_id"] = id
            for key in self.map.keys():
                # get the key name
                mydict_name = self.map[key]["variable_mydict"]
                mydict_value = self.map[key]["value_mydict"]
                # find out which value was set
                mydict_current = subdict[mydict_name]
                # now map to new key-value pair
                mapped_subdict[key] = 1 if mydict_current == mydict_value else 0
                # substitute the values that are not boolean
                if self.map[key]["variable_coding"] != "Bool":
                    mapped_subdict[key] = mydict_current
            mapped_dict[id] = mapped_subdict
        return mapped_dict


# 10056701: {'id': 100567, 'pic_order': 1, 'pic_id': 10056701, 'v9_4': 1, 'v9_5a': 0.0, 'v9_5b': 1, 'v9_6': 1, 'v9_7': 0, 'v9_8': 0, 'v9_8a': 0, 'v9_9': 1, 'v9_10': 0, 'v9_11': 1, 'v9_12': 0, 'v9_13': 0, 'v9_13_text': nan, 'v11_3': 0}
# Yes,No,1,['No'],[32],['Woman'],['white'],"[('neutral', 91.75465703010559)]",['Neutral']
# 1, 0, 1,


if __name__ == "__main__":
    files = utils.find_files(
        path="/home/inga/projects/misinformation-project/misinformation/misinformation/test/data/Europe APRMAY20 visual data/cropped images"
    )
    mydict = utils.initialize_dict(files)
    # analyze faces
    image_ids = [key for key in mydict.keys()]
    for i in image_ids:
        mydict[i] = faces.EmotionDetector(mydict[i]).analyse_image()

    outdict = utils.append_data_to_dict(mydict)
    df = utils.dump_df(outdict)
    # print(df.head(10))
    df.to_csv("mydict_out.csv")

    # example of LabelManager for loading csv data to dict
    lm = LabelManager()
    # get the desired label numbers automatically
    orders = lm.get_orders()
    # map mydict to the specified variable names and values
    mydict_map = lm.map_dict(mydict)
    print(mydict_map)
    lm.filter_from_order([1, 2, 3] + orders)

    labels = lm.gen_dict()
    print(labels)
