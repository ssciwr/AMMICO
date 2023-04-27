from ammico.utils import AnalysisMethod
import torch
import torch.nn.functional as Func
import requests
import lavis
import os
import numpy as np
from PIL import Image
from skimage import transform as skimage_transform
from scipy.ndimage import filters
from matplotlib import pyplot as plt
from IPython.display import display
from lavis.models import load_model_and_preprocess, load_model, BlipBase
from lavis.processors import load_processor


class MultimodalSearch(AnalysisMethod):
    def __init__(self, subdict: dict) -> None:
        super().__init__(subdict)

    multimodal_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_feature_extractor_model_blip2(self, device):
        model, vis_processors, txt_processors = load_model_and_preprocess(
            name="blip2_feature_extractor",
            model_type="pretrain",
            is_eval=True,
            device=device,
        )
        return model, vis_processors, txt_processors

    def load_feature_extractor_model_blip(self, device):
        model, vis_processors, txt_processors = load_model_and_preprocess(
            name="blip_feature_extractor",
            model_type="base",
            is_eval=True,
            device=device,
        )
        return model, vis_processors, txt_processors

    def load_feature_extractor_model_albef(self, device):
        model, vis_processors, txt_processors = load_model_and_preprocess(
            name="albef_feature_extractor",
            model_type="base",
            is_eval=True,
            device=device,
        )
        return model, vis_processors, txt_processors

    def load_feature_extractor_model_clip_base(self, device):
        model, vis_processors, txt_processors = load_model_and_preprocess(
            name="clip_feature_extractor",
            model_type="base",
            is_eval=True,
            device=device,
        )
        return model, vis_processors, txt_processors

    def load_feature_extractor_model_clip_vitl14(self, device):
        model, vis_processors, txt_processors = load_model_and_preprocess(
            name="clip_feature_extractor",
            model_type="ViT-L-14",
            is_eval=True,
            device=device,
        )
        return model, vis_processors, txt_processors

    def load_feature_extractor_model_clip_vitl14_336(self, device):
        model, vis_processors, txt_processors = load_model_and_preprocess(
            name="clip_feature_extractor",
            model_type="ViT-L-14-336",
            is_eval=True,
            device=device,
        )
        return model, vis_processors, txt_processors

    def read_img(self, filepath):
        raw_image = Image.open(filepath).convert("RGB")
        return raw_image

    def read_and_process_images(self, image_paths, vis_processor):
        raw_images = [MultimodalSearch.read_img(self, path) for path in image_paths]
        images = [
            vis_processor["eval"](r_img)
            .unsqueeze(0)
            .to(MultimodalSearch.multimodal_device)
            for r_img in raw_images
        ]
        images_tensors = torch.stack(images)

        return raw_images, images_tensors

    def extract_image_features_blip2(self, model, images_tensors):
        with torch.cuda.amp.autocast(
            enabled=(MultimodalSearch.multimodal_device != torch.device("cpu"))
        ):
            features_image = [
                model.extract_features({"image": ten, "text_input": ""}, mode="image")
                for ten in images_tensors
            ]
            features_image_stacked = torch.stack(
                [feat.image_embeds_proj[:, 0, :].squeeze(0) for feat in features_image]
            )
        return features_image_stacked

    def extract_image_features_clip(self, model, images_tensors):
        features_image = [
            model.extract_features({"image": ten}) for ten in images_tensors
        ]
        features_image_stacked = torch.stack(
            [Func.normalize(feat.float(), dim=-1).squeeze(0) for feat in features_image]
        )
        return features_image_stacked

    def extract_image_features_basic(self, model, images_tensors):
        features_image = [
            model.extract_features({"image": ten, "text_input": ""}, mode="image")
            for ten in images_tensors
        ]
        features_image_stacked = torch.stack(
            [feat.image_embeds_proj[:, 0, :].squeeze(0) for feat in features_image]
        )
        return features_image_stacked

    def save_tensors(
        self,
        model_type,
        features_image_stacked,
        name="saved_features_image.pt",
        path="./saved_tensors/",
    ):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(
            str(path)
            + str(len(features_image_stacked))
            + "_"
            + model_type
            + "_"
            + name,
            "wb",
        ) as f:
            torch.save(features_image_stacked, f)
        return name

    def load_tensors(self, name):
        features_image_stacked = torch.load(name)
        return features_image_stacked

    def extract_text_features(self, model, text_input):
        sample_text = {"text_input": [text_input]}
        features_text = model.extract_features(sample_text, mode="text")

        return features_text

    def parsing_images(
        self,
        model_type,
        path_to_saved_tensors="./saved_tensors/",
        path_to_load_tensors=None,
    ):
        if model_type in ("clip_base", "clip_vitl14_336", "clip_vitl14"):
            path_to_lib = lavis.__file__[:-11] + "models/clip_models/"
            url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/lavis/models/clip_models/bpe_simple_vocab_16e6.txt.gz"
            r = requests.get(url, allow_redirects=False)
            open(path_to_lib + "bpe_simple_vocab_16e6.txt.gz", "wb").write(r.content)

        image_keys = sorted(self.keys())
        image_names = [self[k]["filename"] for k in image_keys]

        select_model = {
            "blip2": MultimodalSearch.load_feature_extractor_model_blip2,
            "blip": MultimodalSearch.load_feature_extractor_model_blip,
            "albef": MultimodalSearch.load_feature_extractor_model_albef,
            "clip_base": MultimodalSearch.load_feature_extractor_model_clip_base,
            "clip_vitl14": MultimodalSearch.load_feature_extractor_model_clip_vitl14,
            "clip_vitl14_336": MultimodalSearch.load_feature_extractor_model_clip_vitl14_336,
        }

        select_extract_image_features = {
            "blip2": MultimodalSearch.extract_image_features_blip2,
            "blip": MultimodalSearch.extract_image_features_basic,
            "albef": MultimodalSearch.extract_image_features_basic,
            "clip_base": MultimodalSearch.extract_image_features_clip,
            "clip_vitl14": MultimodalSearch.extract_image_features_clip,
            "clip_vitl14_336": MultimodalSearch.extract_image_features_clip,
        }

        if model_type in select_model.keys():
            (
                model,
                vis_processors,
                txt_processors,
            ) = select_model[
                model_type
            ](self, MultimodalSearch.multimodal_device)
        else:
            raise SyntaxError(
                "Please, use one of the following models: blip2, blip, albef, clip_base, clip_vitl14, clip_vitl14_336"
            )

        _, images_tensors = MultimodalSearch.read_and_process_images(
            self, image_names, vis_processors
        )
        if path_to_load_tensors is None:
            with torch.no_grad():
                features_image_stacked = select_extract_image_features[model_type](
                    self, model, images_tensors
                )
            MultimodalSearch.save_tensors(
                self, model_type, features_image_stacked, path=path_to_saved_tensors
            )
        else:
            features_image_stacked = MultimodalSearch.load_tensors(
                self, str(path_to_load_tensors)
            )

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
        select_extract_image_features = {
            "blip2": MultimodalSearch.extract_image_features_blip2,
            "blip": MultimodalSearch.extract_image_features_basic,
            "albef": MultimodalSearch.extract_image_features_basic,
            "clip_base": MultimodalSearch.extract_image_features_clip,
            "clip_vitl14": MultimodalSearch.extract_image_features_clip,
            "clip_vitl14_336": MultimodalSearch.extract_image_features_clip,
        }

        for query in search_query:
            if len(query) != 1 and (query in ("image", "text_input")):
                raise SyntaxError(
                    'Each query must contain either an "image" or a "text_input"'
                )
        multi_sample = []
        for query in search_query:
            if "text_input" in query.keys():
                text_processing = txt_processors["eval"](query["text_input"])
                images_tensors = ""
            elif "image" in query.keys():
                _, images_tensors = MultimodalSearch.read_and_process_images(
                    self, [query["image"]], vis_processors
                )
                text_processing = ""
            multi_sample.append(
                {"image": images_tensors, "text_input": text_processing}
            )

        multi_features_query = []
        for query in multi_sample:
            if query["image"] == "":
                if model_type in ("clip_base", "clip_vitl14_336", "clip_vitl14"):
                    features = model.extract_features(
                        {"text_input": query["text_input"]}
                    )

                    features_squeeze = features.squeeze(0).to(
                        MultimodalSearch.multimodal_device
                    )
                    multi_features_query.append(
                        Func.normalize(features_squeeze, dim=-1)
                    )
                else:
                    features = model.extract_features(query, mode="text")
                    features_squeeze = (
                        features.text_embeds_proj[:, 0, :]
                        .squeeze(0)
                        .to(MultimodalSearch.multimodal_device)
                    )
                    multi_features_query.append(features_squeeze)
            if query["text_input"] == "":
                multi_features_query.append(
                    select_extract_image_features[model_type](
                        self, model, query["image"]
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
        filter_number_of_images=None,
        filter_val_limit=None,
        filter_rel_error=None,
    ):
        if filter_number_of_images is None:
            filter_number_of_images = len(self)
        if filter_val_limit is None:
            filter_val_limit = 0
        if filter_rel_error is None:
            filter_rel_error = 1e10

        features_image_stacked.to(MultimodalSearch.multimodal_device)

        with torch.no_grad():
            multi_features_stacked = MultimodalSearch.querys_processing(
                self, search_query, model, txt_processors, vis_processors, model_type
            )

        similarity = features_image_stacked @ multi_features_stacked.t()
        sorted_lists = [
            sorted(
                range(len(similarity)),
                key=lambda k, value=i: similarity[k, value],
                reverse=True,
            )
            for i in range(len(similarity[0]))
        ]
        places = [[item.index(i) for i in range(len(item))] for item in sorted_lists]

        for q in range(len(search_query)):
            max_val = similarity[sorted_lists[q][0]][q].item()
            for i, key in zip(range(len(image_keys)), sorted_lists[q]):
                if (
                    i < filter_number_of_images
                    and similarity[key][q].item() > filter_val_limit
                    and 100 * abs(max_val - similarity[key][q].item()) / max_val
                    < filter_rel_error
                ):
                    self[image_keys[key]][
                        "rank " + list(search_query[q].values())[0]
                    ] = places[q][key]
                    self[image_keys[key]][
                        list(search_query[q].values())[0]
                    ] = similarity[key][q].item()
                else:
                    self[image_keys[key]][
                        "rank " + list(search_query[q].values())[0]
                    ] = None
                    self[image_keys[key]][list(search_query[q].values())[0]] = 0
        return similarity, sorted_lists

    def itm_text_precessing(self, search_query):
        for query in search_query:
            if (len(query) != 1) and (query in ("image", "text_input")):
                raise SyntaxError(
                    'Each querry must contain either an "image" or a "text_input"'
                )
        text_query_index = []
        for i, query in zip(range(len(search_query)), search_query):
            if "text_input" in query.keys():
                text_query_index.append(i)

        return text_query_index

    def get_pathes_from_query(self, query):
        paths = []
        image_names = []
        for s in sorted(
            self.items(), key=lambda t: t[1][list(query.values())[0]], reverse=True
        ):
            if s[1]["rank " + list(query.values())[0]] is None:
                break
            paths.append(s[1]["filename"])
            image_names.append(s[0])
        return paths, image_names

    def read_and_process_images_itm(self, image_paths, vis_processor):
        raw_images = [MultimodalSearch.read_img(self, path) for path in image_paths]
        images = [vis_processor(r_img) for r_img in raw_images]
        images_tensors = torch.stack(images).to(MultimodalSearch.multimodal_device)

        return raw_images, images_tensors

    def compute_gradcam_batch(
        self,
        model,
        visual_input,
        text_input,
        tokenized_text,
        block_num=6,
    ):
        model.text_encoder.base_model.base_model.encoder.layer[
            block_num
        ].crossattention.self.save_attention = True

        output = model(
            {"image": visual_input, "text_input": text_input}, match_head="itm"
        )
        loss = output[:, 1].sum()

        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            mask = tokenized_text.attention_mask.view(
                tokenized_text.attention_mask.size(0), 1, -1, 1, 1
            )  # (bsz,1,token_len, 1,1)
            token_length = mask.sum() - 2
            token_length = token_length.cpu()
            # grads and cams [bsz, num_head, seq_len, image_patch]
            grads = model.text_encoder.base_model.base_model.encoder.layer[
                block_num
            ].crossattention.self.get_attn_gradients()
            cams = model.text_encoder.base_model.base_model.encoder.layer[
                block_num
            ].crossattention.self.get_attention_map()

            # assume using vit large with 576 num image patch
            cams = (
                cams[:, :, :, 1:].reshape(visual_input.size(0), 12, -1, 24, 24) * mask
            )
            grads = (
                grads[:, :, :, 1:]
                .clamp(0)
                .reshape(visual_input.size(0), 12, -1, 24, 24)
                * mask
            )

            gradcam = cams * grads
            # [enc token gradcam, average gradcam across token, gradcam for individual token]
            # gradcam = torch.cat((gradcam[0:1,:], gradcam[1:token_length+1, :].sum(dim=0, keepdim=True)/token_length, gradcam[1:, :]))
            gradcam = gradcam.mean(1).cpu().detach()
            gradcam = (
                gradcam[:, 1 : token_length + 1, :].sum(dim=1, keepdim=True)
                / token_length
            )

        return gradcam, output

    def resize_img(self, raw_img):
        w, h = raw_img.size
        scaling_factor = 240 / w
        resized_image = raw_img.resize(
            (int(w * scaling_factor), int(h * scaling_factor))
        )
        return resized_image

    def get_att_map(self, img, att_map, blur=True, overlap=True):
        att_map -= att_map.min()
        if att_map.max() > 0:
            att_map /= att_map.max()
        att_map = skimage_transform.resize(
            att_map, (img.shape[:2]), order=3, mode="constant"
        )
        if blur:
            att_map = filters.gaussian_filter(att_map, 0.02 * max(img.shape[:2]))
            att_map -= att_map.min()
            att_map /= att_map.max()
        cmap = plt.get_cmap("jet")
        att_mapv = cmap(att_map)
        att_mapv = np.delete(att_mapv, 3, 2)
        if overlap:
            att_map = (
                1 * (1 - att_map**0.7).reshape(att_map.shape + (1,)) * img
                + (att_map**0.7).reshape(att_map.shape + (1,)) * att_mapv
            )
        return att_map

    def upload_model_blip2_coco(self):
        itm_model = load_model(
            "blip2_image_text_matching",
            "coco",
            is_eval=True,
            device=MultimodalSearch.multimodal_device,
        )
        vis_processor = load_processor("blip_image_eval").build(image_size=364)
        return itm_model, vis_processor

    def upload_model_blip_base(self):
        itm_model = load_model(
            "blip_image_text_matching",
            "base",
            is_eval=True,
            device=MultimodalSearch.multimodal_device,
        )
        vis_processor = load_processor("blip_image_eval").build(image_size=384)
        return itm_model, vis_processor

    def upload_model_blip_large(self):
        itm_model = load_model(
            "blip_image_text_matching",
            "large",
            is_eval=True,
            device=MultimodalSearch.multimodal_device,
        )
        vis_processor = load_processor("blip_image_eval").build(image_size=384)
        return itm_model, vis_processor

    def image_text_match_reordering(
        self,
        search_query,
        itm_model_type,
        image_keys,
        sorted_lists,
        batch_size=1,
        need_grad_cam=False,
    ):
        if itm_model_type == "blip2_coco" and need_grad_cam is True:
            raise SyntaxError(
                "The blip2_coco model does not yet work with gradcam. Please set need_grad_cam to False"
            )

        choose_model = {
            "blip_base": MultimodalSearch.upload_model_blip_base,
            "blip_large": MultimodalSearch.upload_model_blip_large,
            "blip2_coco": MultimodalSearch.upload_model_blip2_coco,
        }

        itm_model, vis_processor_itm = choose_model[itm_model_type](self)
        text_processor = load_processor("blip_caption")
        tokenizer = BlipBase.init_tokenizer()

        if itm_model_type == "blip2_coco":
            need_grad_cam = False

        text_query_index = MultimodalSearch.itm_text_precessing(self, search_query)

        avg_gradcams = []
        itm_scores = []
        itm_scores2 = []
        image_gradcam_with_itm = {}

        for index_text_query in text_query_index:
            query = search_query[index_text_query]
            pathes, image_names = MultimodalSearch.get_pathes_from_query(self, query)
            num_batches = int(len(pathes) / batch_size)
            num_batches_residue = len(pathes) % batch_size

            local_itm_scores = []
            local_avg_gradcams = []

            if num_batches_residue != 0:
                num_batches = num_batches + 1
            for i in range(num_batches):
                filenames_in_batch = pathes[i * batch_size : (i + 1) * batch_size]
                current_len = len(filenames_in_batch)
                raw_images, images = MultimodalSearch.read_and_process_images_itm(
                    self, filenames_in_batch, vis_processor_itm
                )
                queries_batch = [text_processor(query["text_input"])] * current_len
                queries_tok_batch = tokenizer(queries_batch, return_tensors="pt").to(
                    MultimodalSearch.multimodal_device
                )

                if need_grad_cam:
                    gradcam, itm_output = MultimodalSearch.compute_gradcam_batch(
                        self,
                        itm_model,
                        images,
                        queries_batch,
                        queries_tok_batch,
                    )
                    norm_imgs = [np.float32(r_img) / 255 for r_img in raw_images]

                    for norm_img, grad_cam in zip(norm_imgs, gradcam):
                        avg_gradcam = MultimodalSearch.get_att_map(
                            self, norm_img, np.float32(grad_cam[0]), blur=True
                        )
                        local_avg_gradcams.append(avg_gradcam)

                else:
                    itm_output = itm_model(
                        {"image": images, "text_input": queries_batch}, match_head="itm"
                    )

                with torch.no_grad():
                    itm_score = torch.nn.functional.softmax(itm_output, dim=1)

                local_itm_scores.append(itm_score)

            local_itm_scores2 = torch.cat(local_itm_scores)[:, 1]
            if need_grad_cam:
                localimage_gradcam_with_itm = {
                    n: i * 255 for n, i in zip(image_names, local_avg_gradcams)
                }
            else:
                localimage_gradcam_with_itm = ""
            image_names_with_itm = {
                n: i.item() for n, i in zip(image_names, local_itm_scores2)
            }
            itm_rank = torch.argsort(local_itm_scores2, descending=True)
            image_names_with_new_rank = {
                image_names[i.item()]: rank
                for i, rank in zip(itm_rank, range(len(itm_rank)))
            }
            for i, key in zip(range(len(image_keys)), sorted_lists[index_text_query]):
                if image_keys[key] in image_names:
                    self[image_keys[key]][
                        "itm " + list(search_query[index_text_query].values())[0]
                    ] = image_names_with_itm[image_keys[key]]
                    self[image_keys[key]][
                        "itm_rank " + list(search_query[index_text_query].values())[0]
                    ] = image_names_with_new_rank[image_keys[key]]
                else:
                    self[image_keys[key]][
                        "itm " + list(search_query[index_text_query].values())[0]
                    ] = 0
                    self[image_keys[key]][
                        "itm_rank " + list(search_query[index_text_query].values())[0]
                    ] = None

            avg_gradcams.append(local_avg_gradcams)
            itm_scores.append(local_itm_scores)
            itm_scores2.append(local_itm_scores2)
            image_gradcam_with_itm[
                list(search_query[index_text_query].values())[0]
            ] = localimage_gradcam_with_itm
        del (
            itm_model,
            vis_processor_itm,
            text_processor,
            raw_images,
            images,
            tokenizer,
            queries_batch,
            queries_tok_batch,
            itm_score,
        )
        if need_grad_cam:
            del itm_output, gradcam, norm_img, grad_cam, avg_gradcam
        torch.cuda.empty_cache()
        return itm_scores2, image_gradcam_with_itm

    def show_results(self, query, itm=False, image_gradcam_with_itm=False):
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
        if itm:
            current_querry_val = "itm " + list(query.values())[0]
            current_querry_rank = "itm_rank " + list(query.values())[0]
        else:
            current_querry_val = list(query.values())[0]
            current_querry_rank = "rank " + list(query.values())[0]

        for s in sorted(
            self.items(), key=lambda t: t[1][current_querry_val], reverse=True
        ):
            if s[1][current_querry_rank] is None:
                break
            if image_gradcam_with_itm is False:
                p1 = Image.open(s[1]["filename"]).convert("RGB")
            else:
                image = image_gradcam_with_itm[list(query.values())[0]][s[0]]
                p1 = Image.fromarray(image.astype("uint8"), "RGB")
            p1.thumbnail((400, 400))
            display(
                "Rank: "
                + str(s[1][current_querry_rank])
                + " Val: "
                + str(s[1][current_querry_val]),
                s[0],
                p1,
            )
            display(
                "--------------------------------------------------",
            )
