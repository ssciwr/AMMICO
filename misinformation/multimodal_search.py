from misinformation.utils import AnalysisMethod
import torch
import torch.nn.functional as Func
import requests
import lavis
from PIL import Image
from IPython.display import display
from lavis.models import load_model_and_preprocess


class MultimodalSearch(AnalysisMethod):
    def __init__(self, subdict: dict) -> None:
        super().__init__(subdict)
        # self.subdict.update(self.set_keys())

    multimodal_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_feature_extractor_model(device, model_type):
        if model_type == "blip2":
            model, vis_processors, txt_processors = load_model_and_preprocess(
                name="blip2_feature_extractor",
                model_type="pretrain",
                is_eval=True,
                device=device,
            )
        elif model_type == "blip":
            model, vis_processors, txt_processors = load_model_and_preprocess(
                name="blip_feature_extractor",
                model_type="base",
                is_eval=True,
                device=device,
            )
        elif model_type == "albef":
            model, vis_processors, txt_processors = load_model_and_preprocess(
                name="albef_feature_extractor",
                model_type="base",
                is_eval=True,
                device=device,
            )
        elif model_type == "clip_base":
            model, vis_processors, txt_processors = load_model_and_preprocess(
                name="clip_feature_extractor",
                model_type="base",
                is_eval=True,
                device=device,
            )
        elif model_type == "clip_rn50":
            model, vis_processors, txt_processors = load_model_and_preprocess(
                name="clip_feature_extractor",
                model_type="RN50",
                is_eval=True,
                device=device,
            )
        elif model_type == "clip_vitl14":
            model, vis_processors, txt_processors = load_model_and_preprocess(
                name="clip_feature_extractor",
                model_type="ViT-L-14",
                is_eval=True,
                device=device,
            )
        else:
            print(
                "Please, use one of the following models: blip2, blip, albef, clip_base, clip_rn50, clip_vitl14"
            )

        return model, vis_processors, txt_processors

    def read_img(filepath):
        raw_image = Image.open(filepath).convert("RGB")
        return raw_image

    def read_and_process_images(image_paths, vis_processor):
        raw_images = [MultimodalSearch.read_img(path) for path in image_paths]
        images = [
            vis_processor["eval"](r_img)
            .unsqueeze(0)
            .to(MultimodalSearch.multimodal_device)
            for r_img in raw_images
        ]
        images_tensors = torch.stack(images)

        return raw_images, images_tensors

    def extract_image_features(model, images_tensors, model_type):
        if model_type == "blip2":
            with torch.cuda.amp.autocast(
                enabled=(MultimodalSearch.multimodal_device != torch.device("cpu"))
            ):
                features_image = [
                    model.extract_features(
                        {"image": ten, "text_input": ""}, mode="image"
                    )
                    for ten in images_tensors
                ]
                features_image_stacked = torch.stack(
                    [
                        feat.image_embeds_proj[:, 0, :].squeeze(0)
                        for feat in features_image
                    ]
                )
        elif model_type in ("clip_base", "clip_rn50", "clip_vitl14"):
            features_image = [
                model.extract_features({"image": ten}) for ten in images_tensors
            ]
            features_image_stacked = torch.stack(
                [
                    Func.normalize(feat.float(), dim=-1).squeeze(0)
                    for feat in features_image
                ]
            )
        else:
            features_image = [
                model.extract_features({"image": ten, "text_input": ""}, mode="image")
                for ten in images_tensors
            ]
            features_image_stacked = torch.stack(
                [feat.image_embeds_proj[:, 0, :].squeeze(0) for feat in features_image]
            )
        return features_image_stacked

    def save_tensors(
        model_type, features_image_stacked, name="saved_features_image.pt"
    ):
        with open(
            str(len(features_image_stacked)) + "_" + model_type + "_" + name, "wb"
        ) as f:
            torch.save(features_image_stacked, f)
        return name

    def load_tensors(name="saved_features_image.pt"):
        features_image_stacked = torch.load(name)
        return features_image_stacked

    def extract_text_features(model, text_input):
        sample_text = {"text_input": [text_input]}
        features_text = model.extract_features(sample_text, mode="text")

        return features_text

    def parsing_images(self, model_type):

        if model_type in ("clip_base", "clip_rn50", "clip_vitl14"):
            path_to_lib = lavis.__file__[:-11] + "models/clip_models/"
            url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/lavis/models/clip_models/bpe_simple_vocab_16e6.txt.gz"
            r = requests.get(url, allow_redirects=False)
            open(path_to_lib + "bpe_simple_vocab_16e6.txt.gz", "wb").write(r.content)

        image_keys = sorted(self.keys())
        image_names = [self[k]["filename"] for k in image_keys]

        (
            model,
            vis_processors,
            txt_processors,
        ) = MultimodalSearch.load_feature_extractor_model(
            MultimodalSearch.multimodal_device, model_type
        )
        raw_images, images_tensors = MultimodalSearch.read_and_process_images(
            image_names, vis_processors
        )
        features_image_stacked = MultimodalSearch.extract_image_features(
            model, images_tensors, model_type
        )
        MultimodalSearch.save_tensors(model_type, features_image_stacked)

        return (
            model,
            vis_processors,
            txt_processors,
            image_keys,
            image_names,
            features_image_stacked,
        )

    def querys_processing(
        self, search_query, model, txt_processors, vis_processors, model_type
    ):
        for query in search_query:
            if not (len(query) == 1) and (query in ("image", "text_input")):
                raise SyntaxError(
                    'Each querry must contain either an "image" or a "text_input"'
                )
        multi_sample = []
        for query in search_query:
            if "text_input" in query.keys():
                text_processing = txt_processors["eval"](query["text_input"])
                images_tensors = ""
            elif "image" in query.keys():
                _, images_tensors = MultimodalSearch.read_and_process_images(
                    [query["image"]], vis_processors
                )
                text_processing = ""
            multi_sample.append(
                {"image": images_tensors, "text_input": text_processing}
            )

        if model_type in ("clip_base", "clip_rn50", "clip_vitl14"):
            multi_features_query = []
            for query in multi_sample:
                if query["image"] == "":
                    features = model.extract_features(
                        {"text_input": query["text_input"]}
                    )

                    features_squeeze = features.squeeze(0).to(
                        MultimodalSearch.multimodal_device
                    )
                    multi_features_query.append(
                        Func.normalize(features_squeeze, dim=-1)
                    )
                if query["text_input"] == "":
                    multi_features_query.append(
                        MultimodalSearch.extract_image_features(
                            model, query["image"], model_type
                        )
                    )

        else:
            multi_features_query = []
            for query in multi_sample:
                if query["image"] == "":
                    features = model.extract_features(query, mode="text")
                    features_squeeze = (
                        features.text_embeds_proj[:, 0, :]
                        .squeeze(0)
                        .to(MultimodalSearch.multimodal_device)
                    )
                    multi_features_query.append(features_squeeze)
                if query["text_input"] == "":
                    multi_features_query.append(
                        MultimodalSearch.extract_image_features(
                            model, query["image"], model_type
                        )
                    )

        multi_features_stacked = torch.stack(
            [query.squeeze(0) for query in multi_features_query]
        ).to(MultimodalSearch.multimodal_device)

        return multi_features_stacked

    def multimodal_search(
        self,
        model,
        vis_processors,
        txt_processors,
        model_type,
        image_keys,
        features_image_stacked,
        search_query,
    ):
        features_image_stacked.to(MultimodalSearch.multimodal_device)

        multi_features_stacked = MultimodalSearch.querys_processing(
            self, search_query, model, txt_processors, vis_processors, model_type
        )

        similarity = features_image_stacked @ multi_features_stacked.t()
        sorted_lists = [
            sorted(range(len(similarity)), key=lambda k: similarity[k, i], reverse=True)
            for i in range(len(similarity[0]))
        ]
        places = [[item.index(i) for i in range(len(item))] for item in sorted_lists]

        for q in range(len(search_query)):
            for i, key in zip(range(len(image_keys)), image_keys):
                self[key]["rank " + list(search_query[q].values())[0]] = places[q][i]
                self[key][list(search_query[q].values())[0]] = similarity[i][q].item()

        return similarity

    def show_results(self, query):
        if "image" in query.keys():
            pic = Image.open(query["image"]).convert("RGB")
            pic.thumbnail((400, 400))
            display(
                "Your search query: ",
                pic,
                "--------------------------------------------------",
                "Results:",
            )
        elif "text_input" in query.keys():
            display(
                "Your search query: " + query["text_input"],
                "--------------------------------------------------",
                "Results:",
            )
        for s in sorted(
            self.items(), key=lambda t: t[1][list(query.values())[0]], reverse=True
        ):
            p1 = Image.open(s[1]["filename"]).convert("RGB")
            p1.thumbnail((400, 400))
            display(
                p1,
                "Rank: "
                + str(s[1]["rank " + list(query.values())[0]])
                + " Val: "
                + str(s[1][list(query.values())[0]]),
            )
