from misinformation.utils import AnalysisMethod
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess


class SummaryDetector(AnalysisMethod):
    def __init__(self, subdict: dict) -> None:
        super().__init__(subdict)
        self.subdict.update(self.set_keys())
        self.image_summary = {
            "const_image_summary": None,
            "3_non-deterministic summary": None,
        }

    summary_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    summary_model, summary_vis_processors, _ = load_model_and_preprocess(
        name="blip_caption", model_type="base_coco", is_eval=True, device=summary_device
    )

    def set_keys(self) -> dict:
        params = {
            "const_image_summary": None,
            "3_non-deterministic summary": None,
        }
        return params

    def analyse_image(self):

        path = self.subdict["filename"]
        raw_image = Image.open(path).convert("RGB")
        image = (
            self.summary_vis_processors["eval"](raw_image)
            .unsqueeze(0)
            .to(self.summary_device)
        )
        self.image_summary["const_image_summary"] = self.summary_model.generate(
            {"image": image}
        )[0]
        self.image_summary["3_non-deterministic summary"] = self.summary_model.generate(
            {"image": image}, use_nucleus_sampling=True, num_captions=3
        )
        for key in self.image_summary:
            self.subdict[key] = self.image_summary[key]
        return self.subdict

    (
        summary_VQA_model,
        summary_VQA_vis_processors,
        summary_VQA_txt_processors,
    ) = load_model_and_preprocess(
        name="blip_vqa", model_type="vqav2", is_eval=True, device=summary_device
    )

    def analyse_questions(self, list_of_questions):
        if len(list_of_questions) > 0:
            path = self.subdict["filename"]
            raw_image = Image.open(path).convert("RGB")
            image = (
                self.summary_VQA_vis_processors["eval"](raw_image)
                .unsqueeze(0)
                .to(self.summary_device)
            )
            question_batch = []
            for quest in list_of_questions:
                question_batch.append(self.summary_VQA_txt_processors["eval"](quest))
            batch_size = len(list_of_questions)
            image_batch = image.repeat(batch_size, 1, 1, 1)

            answers_batch = self.summary_VQA_model.predict_answers(
                samples={"image": image_batch, "text_input": question_batch},
                inference_method="generate",
            )

            for q, a in zip(question_batch, answers_batch):
                self.image_summary[q] = a

            for key in self.image_summary:
                self.subdict[key] = self.image_summary[key]
        else:
            print("Please, enter list of questions")
        return self.subdict
