from misinformation.utils import AnalysisMethod
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess


class SummaryDetector(AnalysisMethod):
    def __init__(self, subdict: dict) -> None:
        super().__init__(subdict)

    summary_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary_model, summary_vis_processors, _ = load_model_and_preprocess(
        name="blip_caption",
        model_type="base_coco",
        is_eval=True,
        device=summary_device,
    )

    def analyse_image(self, summary_model=None, summary_vis_processors=None):

        if summary_model is None and summary_vis_processors is None:
            summary_model = SummaryDetector.summary_model
            summary_vis_processors = SummaryDetector.summary_vis_processors

        path = self.subdict["filename"]
        raw_image = Image.open(path).convert("RGB")
        image = (
            summary_vis_processors["eval"](raw_image)
            .unsqueeze(0)
            .to(self.summary_device)
        )
        with torch.no_grad():
            self.subdict["const_image_summary"] = summary_model.generate(
                {"image": image}
            )[0]
            self.subdict["3_non-deterministic summary"] = summary_model.generate(
                {"image": image}, use_nucleus_sampling=True, num_captions=3
            )
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

            with torch.no_grad():
                answers_batch = self.summary_VQA_model.predict_answers(
                    samples={"image": image_batch, "text_input": question_batch},
                    inference_method="generate",
                )

            for q, a in zip(list_of_questions, answers_batch):
                self.subdict[q] = a

        else:
            print("Please, enter list of questions")
        return self.subdict
