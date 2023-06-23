import ammico

if __name__ == "__main__":
    images = ammico.find_files(
        path="../data/test/facebook_screenshots/", pattern="*.jpg"
    )
    mydict = ammico.initialize_dict(images)
    obj = ammico.SummaryDetector()
    # summary_m, summary_v = obj.load_model("base")
    # (
    #     summary_vqa_model,
    #     summary_vqa_vis_processors,
    #     summary_vqa_txt_processors,
    # ) = obj.load_vqa_model()
    for key in mydict:
        mydict[key] = ammico.SummaryDetector(
            mydict[key],
            analysis_type="summary_and_questions",
            # summary_model=summary_m,
            # summary_vis_processors=summary_v,
            # summary_vqa_model=summary_vqa_model,
            # summary_vqa_vis_processors=summary_vqa_vis_processors,
            # summary_vqa_txt_processors=summary_vqa_txt_processors,
        ).analyse_image()
    print(mydict)
    outdict = ammico.append_data_to_dict(mydict)
    df = ammico.dump_df(outdict)
    df.to_csv("data_out4.csv")
