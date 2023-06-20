import ammico

if __name__ == "__main__":
    images = ammico.find_files(path=".")
    mydict = ammico.initialize_dict(images)
    for key in mydict:
        mydict[key] = ammico.TextDetector(
            mydict[key], analyse_text=True
        ).analyse_image()
    print(mydict)
    outdict = ammico.append_data_to_dict(mydict)
    df = ammico.dump_df(outdict)
    df.to_csv("data_out.csv")
