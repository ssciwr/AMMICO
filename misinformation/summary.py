from misinformation.utils import AnalysisMethod
from torch import device, cuda, no_grad
from PIL import Image
from lavis.models import load_model_and_preprocess


class SummaryDetector(AnalysisMethod):
    def __init__(self, subdict: dict) -> None:
        super().__init__(subdict)

    summary_device = device("cuda" if cuda.is_available() else "cpu")
    # summary_model, summary_vis_processors, _ = load_model_and_preprocess(
    #     name="blip_caption",
    #     model_type="base_coco",
    #     is_eval=True,
    #     device=summary_device,
    # )

    def load_model_base(self):
        summary_device = device("cuda" if cuda.is_available() else "cpu")
        summary_model, summary_vis_processors, _ = load_model_and_preprocess(
            name="blip_caption",
            model_type="base_coco",
            is_eval=True,
            device=summary_device,
        )
        return summary_model, summary_vis_processors

    def load_model_large(self):
        summary_device = device("cuda" if cuda.is_available() else "cpu")
        summary_model, summary_vis_processors, _ = load_model_and_preprocess(
            name="blip_caption",
            model_type="large_coco",
            is_eval=True,
            device=summary_device,
        )
        return summary_model, summary_vis_processors

    def load_model(self, model_type):
        select_model = {
            "base": SummaryDetector.load_model_base,
            "large": SummaryDetector.load_model_large,
        }
        summary_model, summary_vis_processors = select_model[model_type](self)
        return summary_model, summary_vis_processors

    def analyse_image(self, summary_model=None, summary_vis_processors=None):

        if summary_model is None and summary_vis_processors is None:
            summary_model, summary_vis_processors = self.load_model_base()

        path = self.subdict["filename"]
        raw_image = Image.open(path).convert("RGB")
        image = (
            summary_vis_processors["eval"](raw_image)
            .unsqueeze(0)
            .to(self.summary_device)
        )
        with no_grad():
            self.subdict["const_image_summary"] = summary_model.generate(
                {"image": image}
            )[0]
            self.subdict["3_non-deterministic summary"] = summary_model.generate(
                {"image": image}, use_nucleus_sampling=True, num_captions=3
            )
        return self.subdict

    # (
    #     summary_VQA_model,
    #     summary_VQA_vis_processors,
    #     summary_VQA_txt_processors,
    # ) = load_model_and_preprocess(
    #     name="blip_vqa", model_type="vqav2", is_eval=True, device=summary_device
    # )

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

            with no_grad():
                answers_batch = self.summary_VQA_model.predict_answers(
                    samples={"image": image_batch, "text_input": question_batch},
                    inference_method="generate",
                )

            for q, a in zip(list_of_questions, answers_batch):
                self.subdict[q] = a

        else:
            print("Please, enter list of questions")
        return self.subdict
