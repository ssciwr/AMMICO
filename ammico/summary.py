from ammico.utils import AnalysisMethod
from torch import cuda, no_grad
from PIL import Image
from lavis.models import load_model_and_preprocess


class SummaryDetector(AnalysisMethod):
    def __init__(self, subdict: dict) -> None:
        super().__init__(subdict)
        self.summary_device = "cuda" if cuda.is_available() else "cpu"

    def load_model_base(self):
        """
        Load base_coco blip_caption model and preprocessors for visual inputs from lavis.models.

        Args:

        Returns:
            model (torch.nn.Module): model.
            vis_processors (dict): preprocessors for visual inputs.
        """
        summary_model, summary_vis_processors, _ = load_model_and_preprocess(
            name="blip_caption",
            model_type="base_coco",
            is_eval=True,
            device=self.summary_device,
        )
        return summary_model, summary_vis_processors

    def load_model_large(self):
        """
        Load large_coco blip_caption model and preprocessors for visual inputs from lavis.models.

        Args:

        Returns:
            model (torch.nn.Module): model.
            vis_processors (dict): preprocessors for visual inputs.
        """
        summary_model, summary_vis_processors, _ = load_model_and_preprocess(
            name="blip_caption",
            model_type="large_coco",
            is_eval=True,
            device=self.summary_device,
        )
        return summary_model, summary_vis_processors

    def load_model(self, model_type: str):
        """
        Load blip_caption model and preprocessors for visual inputs from lavis.models.

        Args:
            model_type (str): type of the model.

        Returns:
            model (torch.nn.Module): model.
            vis_processors (dict): preprocessors for visual inputs.
        """
        select_model = {
            "base": SummaryDetector.load_model_base,
            "large": SummaryDetector.load_model_large,
        }
        summary_model, summary_vis_processors = select_model[model_type](self)
        return summary_model, summary_vis_processors

    def analyse_image(self, summary_model=None, summary_vis_processors=None):
        """
        Create 1 constant and 3 non deterministic captions for image.

        Args:
            summary_model (str): model.
            summary_vis_processors (str): preprocessors for visual inputs.

        Returns:
            self.subdict (dict): dictionary with constant image summary and 3 non deterministic summary.
        """
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

    def analyse_questions(self, list_of_questions: list[str]) -> dict:
        """
        Generate answers to free-form questions about image written in natural language.

        Args:
            list_of_questions (list[str]): list of questions.

        Returns:
            self.subdict (dict): dictionary with answers to questions.
        """
        (
            summary_vqa_model,
            summary_vqa_vis_processors,
            summary_vqa_txt_processors,
        ) = load_model_and_preprocess(
            name="blip_vqa",
            model_type="vqav2",
            is_eval=True,
            device=self.summary_device,
        )
        if len(list_of_questions) > 0:
            path = self.subdict["filename"]
            raw_image = Image.open(path).convert("RGB")
            image = (
                summary_vqa_vis_processors["eval"](raw_image)
                .unsqueeze(0)
                .to(self.summary_device)
            )
            question_batch = []
            for quest in list_of_questions:
                question_batch.append(summary_vqa_txt_processors["eval"](quest))
            batch_size = len(list_of_questions)
            image_batch = image.repeat(batch_size, 1, 1, 1)

            with no_grad():
                answers_batch = summary_vqa_model.predict_answers(
                    samples={"image": image_batch, "text_input": question_batch},
                    inference_method="generate",
                )

            for q, a in zip(list_of_questions, answers_batch):
                self.subdict[q] = a

        else:
            print("Please, enter list of questions")
        return self.subdict
