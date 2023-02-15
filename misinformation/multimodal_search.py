from misinformation.utils import AnalysisMethod
import torch
from PIL import Image
from IPython.display import display
from lavis.models import load_model_and_preprocess

class MultimodalSearch(AnalysisMethod):
    def __init__(self, subdict: dict) -> None:
        super().__init__(subdict)
        #self.subdict.update(self.set_keys())

    multimodal_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_feature_extractor_model(device, model_type):
        if (model_type == "blip2"):
            model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)
        elif (model_type == "blip"):
            model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=True, device=device)    
        elif (model_type == "albef"):
            model, vis_processors, txt_processors = load_model_and_preprocess(name="albef_feature_extractor", model_type="base", is_eval=True, device=device)
        elif (model_type == "clip_base"):
            model, vis_processors, txt_processors = load_model_and_preprocess(name="clip_feature_extractor", model_type="base", is_eval=True, device=device)
        elif (model_type == "clip_rn50"):
            model, vis_processors, txt_processors = load_model_and_preprocess(name="clip_feature_extractor", model_type="RN50", is_eval=True, device=device)
        elif (model_type == "clip_vitl14"):
            model, vis_processors, txt_processors = load_model_and_preprocess(name="clip_feature_extractor", model_type="ViT-L-14", is_eval=True, device=device)
        else:
            print("Please, use one of the following models: blip2, blip, albef, clip_base, clip_rn50, clip_vitl14")
        
        return model, vis_processors, txt_processors

    def read_img(filepath):
        raw_image = Image.open(filepath).convert("RGB")
        return raw_image
    
    def read_and_process_images(image_paths, vis_processor):
        raw_images = [MultimodalSearch.read_img(path) for path in image_paths]
        images = [vis_processor["eval"](r_img).unsqueeze(0).to(MultimodalSearch.multimodal_device) for r_img in raw_images]
        images_tensors = torch.stack(images)

        return raw_images, images_tensors

    def extract_image_features(model, images_tensors, model_type):
        if (model_type == "blip2"):
            with torch.cuda.amp.autocast(enabled=(MultimodalSearch.multimodal_device != torch.device("cpu"))):
                features_image = [model.extract_features({"image": ten}, mode="image") for ten in images_tensors]
        else:
            features_image = [model.extract_features({"image": ten}, mode="image") for ten in images_tensors]
        
        features_image_stacked = torch.stack([feat.image_embeds_proj[:,0,:].squeeze(0) for feat in features_image])
        
        return features_image_stacked

    def save_tensors(features_image_stacked, name = 'saved_features_image.pt'):
        with open(name,'wb') as f:
            torch.save(features_image_stacked, f)
        return name
    
    def load_tensors(name = 'saved_features_image.pt'):
        features_image_stacked = torch.load(name)
        return features_image_stacked

    def extract_text_features(model, text_input):
        sample_text = {"text_input": [text_input]}
        features_text = model.extract_features(sample_text, mode="text")        
        
        return features_text
    
    def parsing_images(self, model_type):
        image_keys = sorted(self.keys())
        image_names = [self[k]['filename'] for k in image_keys]
        model, vis_processors, txt_processors = MultimodalSearch.load_feature_extractor_model(MultimodalSearch.multimodal_device, model_type)
        raw_images, images_tensors = MultimodalSearch.read_and_process_images(image_names, vis_processors)
        features_image_stacked = MultimodalSearch.extract_image_features(model, images_tensors, model_type)
        MultimodalSearch.save_tensors(features_image_stacked)
        
        return image_keys, image_names, features_image_stacked

    def multimodal_search(self, model_type, image_keys, features_image_stacked, search_query):
        features_image_stacked.to(MultimodalSearch.multimodal_device)
        model, vis_processors, txt_processors = MultimodalSearch.load_feature_extractor_model(MultimodalSearch.multimodal_device, model_type)
        multi_text_input = [txt_processors["eval"](query) for query in search_query]
        multi_sample = [ {"text_input": [query]} for query in multi_text_input]
        multi_features_text = [model.extract_features(sample, mode="text") for sample in multi_sample]
        multi_features_text_stacked = torch.stack([features.text_embeds_proj[:, 0, :].squeeze(0) for features in multi_features_text]).to(MultimodalSearch.multimodal_device)
        similarity = features_image_stacked @ multi_features_text_stacked.t()
        sorted_lists = [ sorted(range(len(similarity)), key=lambda k: similarity[k,i], reverse=True) for i in range(len(similarity[0])) ]
        places = [ [item.index(i) for i in range(len(item))] for item in sorted_lists ]

        for q in range(len(search_query)): 
            for i,key in zip(range(len(image_keys)),image_keys):
                self[key]['rank ' + search_query[q]] = places[q][i]
                self[key][search_query[q]] = similarity[i][q].item()

        return self

    def show_results(self,query):
        for s in sorted(self.items(), key=lambda t:t[1][query], reverse=True):
            p1 = Image.open(s[1]['filename']).convert("RGB")
            p1.thumbnail((400, 400))
            display(p1,s[1][query])