from ammico.utils import AnalysisMethod
from torch import cuda, no_grad
from PIL import Image
from lavis.models import load_model_and_preprocess


class SummaryDetector(AnalysisMethod):
    def __init__(
        self,
        subdict: dict = {},
        model_type: str = "base",
        analysis_type: str = "summary_and_questions",
        list_of_questions: str = None,
        summary_model=None,
        summary_vis_processors=None,
        summary_vqa_model=None,
        summary_vqa_vis_processors=None,
        summary_vqa_txt_processors=None,
        summary_vqa_model_new=None,
        summary_vqa_vis_processors_new=None,
        summary_vqa_txt_processors_new=None,
        device_type: str = None,
    ) -> None:
        """
        SummaryDetector class for analysing images using the blip_caption model.

        Args:
            subdict (dict, optional): Dictionary containing the image to be analysed. Defaults to {}.
            model_type (str, optional): Type of blip_caption model to use. Can be "base" or "large". Defaults to "base".
            analysis_type (str, optional): Type of analysis to perform. Can be "summary", "questions" or "summary_and_questions". Defaults to "summary_and_questions".
            list_of_questions (list, optional): List of questions to answer. Defaults to ["Are there people in the image?", "What is this picture about?"].
            summary_model ([type], optional): blip_caption model. Defaults to None.
            summary_vis_processors ([type], optional): Preprocessors for visual inputs. Defaults to None.
            summary_vqa_model ([type], optional): blip_vqa model. Defaults to None.
            summary_vqa_vis_processors ([type], optional): Preprocessors for vqa visual inputs. Defaults to None.
            summary_vqa_txt_processors ([type], optional): Preprocessors for vqa text inputs. Defaults to None.

        Raises:
            ValueError: If analysis_type is not one of "summary", "questions" or "summary_and_questions".

        Returns:
            None.
        """

        super().__init__(subdict)
        allowed_analysis_types = ["summary", "questions", "summary_and_questions"]
        allowed_new_analysis_types = [
            "new_summary",
            "new_questions",
            "new_summary_and_questions",
        ]
        all_allowed_analysis_types = allowed_analysis_types + allowed_new_analysis_types
        if analysis_type not in all_allowed_analysis_types:
            raise ValueError(
                "analysis_type must be one of {}".format(all_allowed_analysis_types)
            )
        if device_type is None:
            self.summary_device = "cuda" if cuda.is_available() else "cpu"
        else:
            self.summary_device = device_type
        allowed_model_types = [
            "base",
            "large",
        ]
        allowed_new_model_types = [
            "blip2_t5_pretrain_flant5xxl",
            "blip2_t5_pretrain_flant5xl",
            "blip2_t5_caption_coco_flant5xl",
            "blip2_opt_pretrain_opt2.7b",
            "blip2_opt_pretrain_opt6.7b",
            "blip2_opt_caption_coco_opt2.7b",
            "blip2_opt_caption_coco_opt6.7b",
        ]
        all_allowed_model_types = allowed_model_types + allowed_new_model_types
        if model_type not in all_allowed_model_types:
            raise ValueError(
                "Model type is not allowed - please select one of {}".format(
                    all_allowed_model_types
                )
            )
        self.model_type = model_type
        self.analysis_type = analysis_type
        if list_of_questions is None and analysis_type in allowed_analysis_types:
            self.list_of_questions = [
                "Are there people in the image?",
                "What is this picture about?",
            ]
        elif list_of_questions is None and analysis_type in allowed_new_analysis_types:
            self.list_of_questions = [
                "Question: Are there people in the image? Answer:",
                "Question: What is this picture about? Answer:",
            ]
        elif (not isinstance(list_of_questions, list)) or (
            not all(isinstance(i, str) for i in list_of_questions)
        ):
            raise ValueError("list_of_questions must be a list of string (questions)")
        else:
            self.list_of_questions = list_of_questions
        if (
            (summary_model is None)
            and (summary_vis_processors is None)
            and (analysis_type == "summary" or analysis_type == "summary_and_questions")
        ):
            self.summary_model, self.summary_vis_processors = self.load_model(
                model_type=model_type
            )
        else:
            self.summary_model = summary_model
            self.summary_vis_processors = summary_vis_processors
        if (
            (summary_vqa_model is None)
            and (summary_vqa_vis_processors is None)
            and (summary_vqa_txt_processors is None)
            and (
                analysis_type == "questions" or analysis_type == "summary_and_questions"
            )
        ):
            (
                self.summary_vqa_model,
                self.summary_vqa_vis_processors,
                self.summary_vqa_txt_processors,
            ) = self.load_vqa_model()
        else:
            self.summary_vqa_model = summary_vqa_model
            self.summary_vqa_vis_processors = summary_vqa_vis_processors
            self.summary_vqa_txt_processors = summary_vqa_txt_processors
        if (
            (summary_vqa_model_new is None)
            and (summary_vqa_vis_processors_new is None)
            and (summary_vqa_txt_processors_new is None)
            and (analysis_type in allowed_new_analysis_types)
        ):
            (
                self.summary_vqa_model_new,
                self.summary_vqa_vis_processors_new,
                self.summary_vqa_txt_processors_new,
            ) = self.load_new_model(model_type=model_type)
        else:
            self.summary_vqa_model_new = summary_vqa_model_new
            self.summary_vqa_vis_processors_new = summary_vqa_vis_processors_new
            self.summary_vqa_txt_processors_new = summary_vqa_txt_processors_new

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

    def load_vqa_model(self):
        """
        Load blip_vqa model and preprocessors for visual and text inputs from lavis.models.

        Args:

        Returns:
            model (torch.nn.Module): model.
            vis_processors (dict): preprocessors for visual inputs.
            txt_processors (dict): preprocessors for text inputs.

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
        return summary_vqa_model, summary_vqa_vis_processors, summary_vqa_txt_processors

    def analyse_image(
        self,
        analysis_type: str = None,
        subdict: dict = {},
        list_of_questions: list[str] = None,
    ):
        """
        Analyse image with blip_caption model.

        Args:

        Returns:
            self.subdict (dict): dictionary with analysis results.
        """
        if analysis_type is not None:
            self.analysis_type = analysis_type
        if bool(subdict):
            self.subdict = subdict
        if list_of_questions is not None:
            self.list_of_questions = list_of_questions
        if self.analysis_type == "summary_and_questions":
            self.analyse_summary()
            self.analyse_questions(self.list_of_questions)
        elif self.analysis_type == "summary":
            self.analyse_summary()
        elif self.analysis_type == "questions":
            self.analyse_questions(self.list_of_questions)
        elif self.analysis_type == "new_summary":
            self.analyse_summary_new()
        elif self.analysis_type == "new_questions":
            self.analyse_questions_new(self.list_of_questions)
        elif self.analysis_type == "new_summary_and_questions":
            self.analyse_summary_new()
            self.analyse_questions_new(self.list_of_questions)
        return self.subdict

    def analyse_summary(self):
        """
        Create 1 constant and 3 non deterministic captions for image.

        Args:

        Returns:
            self.subdict (dict): dictionary with analysis results.
        """

        path = self.subdict["filename"]
        raw_image = Image.open(path).convert("RGB")
        image = (
            self.summary_vis_processors["eval"](raw_image)
            .unsqueeze(0)
            .to(self.summary_device)
        )
        with no_grad():
            self.subdict["const_image_summary"] = self.summary_model.generate(
                {"image": image}
            )[0]
            self.subdict["3_non-deterministic summary"] = self.summary_model.generate(
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
        if (
            (self.summary_vqa_model is None)
            and (self.summary_vqa_vis_processors is None)
            and (self.summary_vqa_txt_processors is None)
        ):
            (
                self.summary_vqa_model,
                self.summary_vqa_vis_processors,
                self.summary_vqa_txt_processors,
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
                self.summary_vqa_vis_processors["eval"](raw_image)
                .unsqueeze(0)
                .to(self.summary_device)
            )
            question_batch = []
            for quest in list_of_questions:
                question_batch.append(self.summary_vqa_txt_processors["eval"](quest))
            batch_size = len(list_of_questions)
            image_batch = image.repeat(batch_size, 1, 1, 1)

            with no_grad():
                answers_batch = self.summary_vqa_model.predict_answers(
                    samples={"image": image_batch, "text_input": question_batch},
                    inference_method="generate",
                )

            for q, a in zip(list_of_questions, answers_batch):
                self.subdict[q] = a

        else:
            print("Please, enter list of questions")
        return self.subdict

    def load_new_model(self, model_type: str):
        """
        Load blip_caption model and preprocessors for visual inputs from lavis.models.

        Args:
            model_type (str): type of the model.

        Returns:
            model (torch.nn.Module): model.
            vis_processors (dict): preprocessors for visual inputs.
        """
        select_model = {
            "blip2_t5_pretrain_flant5xxl": SummaryDetector.load_model_blip2_t5_pretrain_flant5xxl,
            "blip2_t5_pretrain_flant5xl": SummaryDetector.load_model_blip2_t5_pretrain_flant5xl,
            "blip2_t5_caption_coco_flant5xl": SummaryDetector.load_model_blip2_t5_caption_coco_flant5xl,
            "blip2_opt_pretrain_opt2.7b": SummaryDetector.load_model_blip2_opt_pretrain_opt2,
            "blip2_opt_pretrain_opt6.7b": SummaryDetector.load_model_base_blip2_opt_pretrain_opt67b,
            "blip2_opt_caption_coco_opt2.7b": SummaryDetector.load_model_blip2_opt_caption_coco_opt27b,
            "blip2_opt_caption_coco_opt6.7b": SummaryDetector.load_model_base_blip2_opt_caption_coco_opt67b,
        }
        (
            summary_vqa_model,
            summary_vqa_vis_processors,
            summary_vqa_txt_processors,
        ) = select_model[model_type](self)
        return summary_vqa_model, summary_vqa_vis_processors, summary_vqa_txt_processors

    def load_model_blip2_t5_pretrain_flant5xxl(self):
        (
            summary_vqa_model,
            summary_vqa_vis_processors,
            summary_vqa_txt_processors,
        ) = load_model_and_preprocess(
            name="blip2_t5",
            model_type="pretrain_flant5xxl",
            is_eval=True,
            device=self.summary_device,
        )
        return summary_vqa_model, summary_vqa_vis_processors, summary_vqa_txt_processors

    def load_model_blip2_t5_pretrain_flant5xl(self):
        (
            summary_vqa_model,
            summary_vqa_vis_processors,
            summary_vqa_txt_processors,
        ) = load_model_and_preprocess(
            name="blip2_t5",
            model_type="pretrain_flant5xl",
            is_eval=True,
            device=self.summary_device,
        )
        return summary_vqa_model, summary_vqa_vis_processors, summary_vqa_txt_processors

    def load_model_blip2_t5_caption_coco_flant5xl(self):
        (
            summary_vqa_model,
            summary_vqa_vis_processors,
            summary_vqa_txt_processors,
        ) = load_model_and_preprocess(
            name="blip2_t5",
            model_type="caption_coco_flant5xl",
            is_eval=True,
            device=self.summary_device,
        )
        return summary_vqa_model, summary_vqa_vis_processors, summary_vqa_txt_processors

    def load_model_blip2_opt_pretrain_opt2(self):
        (
            summary_vqa_model,
            summary_vqa_vis_processors,
            summary_vqa_txt_processors,
        ) = load_model_and_preprocess(
            name="blip2_opt",
            model_type="pretrain_opt2",
            is_eval=True,
            device=self.summary_device,
        )
        return summary_vqa_model, summary_vqa_vis_processors, summary_vqa_txt_processors

    def load_model_base_blip2_opt_pretrain_opt67b(self):
        (
            summary_vqa_model,
            summary_vqa_vis_processors,
            summary_vqa_txt_processors,
        ) = load_model_and_preprocess(
            name="blip2_opt",
            model_type="pretrain_opt6.7b",
            is_eval=True,
            device=self.summary_device,
        )
        return summary_vqa_model, summary_vqa_vis_processors, summary_vqa_txt_processors

    def load_model_blip2_opt_caption_coco_opt27b(self):
        (
            summary_vqa_model,
            summary_vqa_vis_processors,
            summary_vqa_txt_processors,
        ) = load_model_and_preprocess(
            name="blip2_opt",
            model_type="caption_coco_opt2.7b",
            is_eval=True,
            device=self.summary_device,
        )
        return summary_vqa_model, summary_vqa_vis_processors, summary_vqa_txt_processors

    def load_model_base_blip2_opt_caption_coco_opt67b(self):
        (
            summary_vqa_model,
            summary_vqa_vis_processors,
            summary_vqa_txt_processors,
        ) = load_model_and_preprocess(
            name="blip2_opt",
            model_type="caption_coco_opt6.7b",
            is_eval=True,
            device=self.summary_device,
        )
        return summary_vqa_model, summary_vqa_vis_processors, summary_vqa_txt_processors

    def analyse_questions_new(self, list_of_questions: list[str]) -> dict:
        """
        Generate answers to free-form questions about image written in natural language.

        Args:
            list_of_questions (list[str]): list of questions.

        Returns:
            self.subdict (dict): dictionary with answers to questions.
        """
        if len(list_of_questions) > 0:
            path = self.subdict["filename"]
            raw_image = Image.open(path).convert("RGB")
            image = (
                self.summary_vqa_vis_processors_new["eval"](raw_image)
                .unsqueeze(0)
                .to(self.summary_device)
            )
            question_batch = []
            answers_batch = []
            for quest in list_of_questions:
                question = (str)(quest)
                question_batch.append(question)
                answer = self.summary_vqa_model_new.generate(
                    {"image": image, "prompt": question}
                )
                answers_batch.append(answer)

            for q, a in zip(list_of_questions, answers_batch):
                self.subdict[q] = a[0]
        else:
            print("Please, enter list of questions")
        return self.subdict

    def analyse_summary_new(self):
        """
        Create 1 constant and 3 non deterministic captions for image.

        Args:

        Returns:
            self.subdict (dict): dictionary with analysis results.
        """

        path = self.subdict["filename"]
        raw_image = Image.open(path).convert("RGB")
        image = (
            self.summary_vqa_vis_processors_new["eval"](raw_image)
            .unsqueeze(0)
            .to(self.summary_device)
        )
        self.subdict["const_image_summary"] = self.summary_vqa_model_new.generate(
            {"image": image}
        )[0]
        self.subdict[
            "3_non-deterministic summary"
        ] = self.summary_vqa_model_new.generate(
            {"image": image}, use_nucleus_sampling=True, num_captions=3
        )
        if not bool(self.subdict["const_image_summary"].strip()):
            for sum in self.subdict["3_non-deterministic summary"]:
                if not bool(sum.strip()):
                    self.subdict["const_image_summary"] = sum
                    break
        return self.subdict
