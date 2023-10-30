from ammico.utils import AnalysisMethod
from torch import cuda, no_grad
from PIL import Image
from lavis.models import load_model_and_preprocess
from typing import Optional


class SummaryDetector(AnalysisMethod):
    allowed_model_types = [
        "base",
        "large",
        "vqa",
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
    allowed_analysis_types = ["summary", "questions", "summary_and_questions"]

    def __init__(
        self,
        subdict: dict = {},
        model_type: str = "base",
        analysis_type: str = "summary_and_questions",
        list_of_questions: Optional[list[str]] = None,
        summary_model=None,
        summary_vis_processors=None,
        summary_vqa_model=None,
        summary_vqa_vis_processors=None,
        summary_vqa_txt_processors=None,
        summary_vqa_model_new=None,
        summary_vqa_vis_processors_new=None,
        summary_vqa_txt_processors_new=None,
        device_type: Optional[str] = None,
    ) -> None:
        """
        SummaryDetector class for analysing images using the blip_caption model.

        Args:
            subdict (dict, optional): Dictionary containing the image to be analysed. Defaults to {}.

            model_type (str, optional): Type of model to use. Can be "base" or "large" or "vqa" for blip_caption and VQA. Or can be one of the new models:
                "blip2_t5_pretrain_flant5xxl",
                "blip2_t5_pretrain_flant5xl",
                "blip2_t5_caption_coco_flant5xl",
                "blip2_opt_pretrain_opt2.7b",
                "blip2_opt_pretrain_opt6.7b",
                "blip2_opt_caption_coco_opt2.7b",
                "blip2_opt_caption_coco_opt6.7b". Defaults to "base".
            analysis_type (str, optional): Type of analysis to perform. Can be "summary", "questions" or "summary_and_questions". Defaults to "summary_and_questions".
            list_of_questions (list, optional): List of questions to answer. Defaults to ["Are there people in the image?", "What is this picture about?"].
            summary_model ([type], optional): blip_caption model. Defaults to None.
            summary_vis_processors ([type], optional): Preprocessors for visual inputs. Defaults to None.
            summary_vqa_model ([type], optional): blip_vqa model. Defaults to None.
            summary_vqa_vis_processors ([type], optional): Preprocessors for vqa visual inputs. Defaults to None.
            summary_vqa_txt_processors ([type], optional): Preprocessors for vqa text inputs. Defaults to None.
            summary_vqa_model_new ([type], optional): new_vqa model. Defaults to None.
            summary_vqa_vis_processors_new ([type], optional): Preprocessors for vqa visual inputs. Defaults to None.
            summary_vqa_txt_processors_new ([type], optional): Preprocessors for vqa text inputs. Defaults to None.

        Raises:
            ValueError: If analysis_type is not one of "summary", "questions" or "summary_and_questions".

        Returns:
            None.
        """

        super().__init__(subdict)
        # check if analysis_type is valid
        if analysis_type not in self.allowed_analysis_types:
            raise ValueError(
                "analysis_type must be one of {}".format(self.allowed_analysis_types)
            )
        # check if device_type is valid
        if device_type is None:
            self.summary_device = "cuda" if cuda.is_available() else "cpu"
        elif device_type not in ["cuda", "cpu"]:
            raise ValueError("device_type must be one of {}".format(["cuda", "cpu"]))
        else:
            self.summary_device = device_type
        # check if model_type is valid
        if model_type not in self.all_allowed_model_types:
            raise ValueError(
                "Model type is not allowed - please select one of {}".format(
                    self.all_allowed_model_types
                )
            )
        self.model_type = model_type
        self.analysis_type = analysis_type
        # check if list_of_questions is valid
        if list_of_questions is None and model_type in self.allowed_model_types:
            self.list_of_questions = [
                "Are there people in the image?",
                "What is this picture about?",
            ]
        elif list_of_questions is None and model_type in self.allowed_new_model_types:
            self.list_of_questions = [
                "Question: Are there people in the image? Answer:",
                "Question: What is this picture about? Answer:",
            ]
        elif (not isinstance(list_of_questions, list)) or (
            not all(isinstance(i, str) for i in list_of_questions)
        ):
            raise ValueError(
                "list_of_questions must be a list of string (questions)"
            )  # add sequence of questions
        else:
            self.list_of_questions = list_of_questions
        # load models and preprocessors
        if (
            model_type in self.allowed_model_types
            and (summary_model is None)
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
            model_type in self.allowed_model_types
            and (summary_vqa_model is None)
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
            model_type in self.allowed_new_model_types
            and (summary_vqa_model_new is None)
            and (summary_vqa_vis_processors_new is None)
            and (summary_vqa_txt_processors_new is None)
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
            summary_model (torch.nn.Module): model.
            summary_vis_processors (dict): preprocessors for visual inputs.
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
            summary_model (torch.nn.Module): model.
            summary_vis_processors (dict): preprocessors for visual inputs.
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
            summary_model (torch.nn.Module): model.
            summary_vis_processors (dict): preprocessors for visual inputs.
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
            summary_vqa_model (torch.nn.Module): model.
            summary_vqa_vis_processors (dict): preprocessors for visual inputs.
            summary_vqa_txt_processors (dict): preprocessors for text inputs.

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
        subdict: dict = None,
        analysis_type: Optional[str] = None,
        list_of_questions: Optional[list[str]] = None,
        consequential_questions: bool = False,
    ):
        """
        Analyse image with blip_caption model.

        Args:
            analysis_type (str): type of the analysis.
            subdict (dict): dictionary with analising pictures.
            list_of_questions (list[str]): list of questions.
            consequential_questions (bool): whether to ask consequential questions. Works only for new BLIP2 models.

        Returns:
            self.subdict (dict): dictionary with analysis results.
        """
        if analysis_type is None:
            analysis_type = self.analysis_type
        if subdict is not None:
            self.subdict = subdict
        if list_of_questions is not None:
            self.list_of_questions = list_of_questions

        if analysis_type == "summary_and_questions":
            if (
                self.model_type in self.allowed_model_types
                and self.analysis_type != "summary_and_questions"
            ):  # if model_type is not new and required model is absent
                if self.summary_model is None:  # load summary model if it is not loaded
                    self.summary_model, self.summary_vis_processors = self.load_model(
                        model_type=self.model_type
                    )
                elif (
                    self.summary_vqa_model is None
                ):  # load vqa model if it is not loaded
                    (
                        self.summary_vqa_model,
                        self.summary_vqa_vis_processors,
                        self.summary_vqa_txt_processors,
                    ) = self.load_vqa_model()
                self.analysis_type = "summary_and_questions"  # now all models are loaded, so you can perform any analysis
            self.analyse_summary(nondeterministic_summaries=True)
            self.analyse_questions(self.list_of_questions, consequential_questions)
        elif analysis_type == "summary":
            if (
                (self.model_type in self.allowed_model_types)
                and (self.analysis_type == "questions")
                and (self.summary_model is None)
            ):  # if model_type is not new and required model is absent
                (
                    self.summary_model,
                    self.summary_vis_processors,
                ) = self.load_model(  # load summary model if it is not loaded
                    model_type=self.model_type
                )
                self.analysis_type = "summary_and_questions"  # now all models are loaded, so you can perform any analysis
            self.analyse_summary(nondeterministic_summaries=True)
        elif analysis_type == "questions":
            if (
                (self.model_type in self.allowed_model_types)
                and (self.analysis_type == "summary")
                and (self.summary_vqa_model is None)
            ):  # if model_type is not new and required model is absent
                (
                    self.summary_vqa_model,  # load vqa model if it is not loaded
                    self.summary_vqa_vis_processors,
                    self.summary_vqa_txt_processors,
                ) = self.load_vqa_model()
                self.analysis_type = "summary_and_questions"  # now all models are loaded, so you can perform any analysis
            self.analyse_questions(self.list_of_questions, consequential_questions)
        else:
            raise ValueError(
                "analysis_type must be one of {}".format(self.allowed_analysis_types)
            )
        return self.subdict

    def analyse_summary(self, nondeterministic_summaries: bool = True):
        """
        Create 1 constant and 3 non deterministic captions for image.

        Args:
            nondeterministic_summaries (bool): whether to create 3 non deterministic captions.

        Returns:
            self.subdict (dict): dictionary with analysis results.
        """
        if self.model_type in self.allowed_model_types:
            vis_processors = self.summary_vis_processors
            model = self.summary_model
        elif self.model_type in self.allowed_new_model_types:
            vis_processors = self.summary_vqa_vis_processors_new
            model = self.summary_vqa_model_new
        else:
            raise ValueError(
                "Model type is not allowed - please select one of {}".format(
                    self.all_allowed_model_types
                )
            )
        path = self.subdict["filename"]
        raw_image = Image.open(path).convert("RGB")
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(self.summary_device)
        with no_grad():
            self.subdict["const_image_summary"] = model.generate({"image": image})[0]
            if nondeterministic_summaries:
                self.subdict["3_non-deterministic_summary"] = model.generate(
                    {"image": image}, use_nucleus_sampling=True, num_captions=3
                )
        return self.subdict

    def analyse_questions(
        self, list_of_questions: list[str], consequential_questions: bool = False
    ) -> dict:
        """
        Generate answers to free-form questions about image written in natural language.

        Args:
            list_of_questions (list[str]): list of questions.
            consequential_questions (bool): whether to ask consequential questions. Works only for new BLIP2 models.

        Returns:
            self.subdict (dict): dictionary with answers to questions.
        """
        model, vis_processors, txt_processors, model_old = self.check_model()
        if len(list_of_questions) > 0:
            path = self.subdict["filename"]
            raw_image = Image.open(path).convert("RGB")
            image = (
                vis_processors["eval"](raw_image).unsqueeze(0).to(self.summary_device)
            )
            question_batch = []
            list_of_questions_processed = []

            if model_old:
                for quest in list_of_questions:
                    list_of_questions_processed.append(txt_processors["eval"](quest))
            else:
                for quest in list_of_questions:
                    list_of_questions_processed.append((str)(quest))

            for quest in list_of_questions_processed:
                question_batch.append(quest)
            batch_size = len(list_of_questions)
            image_batch = image.repeat(batch_size, 1, 1, 1)

            if not consequential_questions:
                with no_grad():
                    if model_old:
                        answers_batch = model.predict_answers(
                            samples={
                                "image": image_batch,
                                "text_input": question_batch,
                            },
                            inference_method="generate",
                        )
                    else:
                        answers_batch = model.generate(
                            {"image": image_batch, "prompt": question_batch}
                        )

                for q, a in zip(list_of_questions, answers_batch):
                    self.subdict[q] = a

            if consequential_questions and not model_old:
                query_with_context = ""
                for quest in question_batch:
                    query_with_context = query_with_context + quest
                    with no_grad():
                        answer = model.generate(
                            {"image": image, "prompt": query_with_context}
                        )
                    self.subdict[query_with_context] = answer[0]
                    query_with_context = query_with_context + " " + answer[0] + ". "
            elif consequential_questions and model_old:
                raise ValueError(
                    "Consequential questions are not allowed for old models"
                )
        else:
            print("Please, enter list of questions")
        return self.subdict

    def check_model(self):
        """
        Check model type and return appropriate model and preprocessors.

        Args:

        Returns:
            model (nn.Module): model.
            vis_processors (dict): visual preprocessor.
            txt_processors (dict): text preprocessor.
            model_old (bool): whether model is old or new.
        """
        if self.model_type in self.allowed_model_types:
            vis_processors = self.summary_vqa_vis_processors
            model = self.summary_vqa_model
            txt_processors = self.summary_vqa_txt_processors
            model_old = True
        elif self.model_type in self.allowed_new_model_types:
            vis_processors = self.summary_vqa_vis_processors_new
            model = self.summary_vqa_model_new
            txt_processors = self.summary_vqa_txt_processors_new
            model_old = False
        else:
            raise ValueError(
                "Model type is not allowed - please select one of {}".format(
                    self.all_allowed_model_types
                )
            )

        return model, vis_processors, txt_processors, model_old

    def load_new_model(self, model_type: str):
        """
        Load new BLIP2 models.

        Args:
            model_type (str): type of the model.

        Returns:
            model (torch.nn.Module): model.
            vis_processors (dict): preprocessors for visual inputs.
            txt_processors (dict): preprocessors for text inputs.
        """
        select_model = {
            "blip2_t5_pretrain_flant5xxl": SummaryDetector.load_model_blip2_t5_pretrain_flant5xxl,
            "blip2_t5_pretrain_flant5xl": SummaryDetector.load_model_blip2_t5_pretrain_flant5xl,
            "blip2_t5_caption_coco_flant5xl": SummaryDetector.load_model_blip2_t5_caption_coco_flant5xl,
            "blip2_opt_pretrain_opt2.7b": SummaryDetector.load_model_blip2_opt_pretrain_opt27b,
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
        """
        Load BLIP2 model with FLAN-T5 XXL architecture.

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
            name="blip2_t5",
            model_type="pretrain_flant5xxl",
            is_eval=True,
            device=self.summary_device,
        )
        return summary_vqa_model, summary_vqa_vis_processors, summary_vqa_txt_processors

    def load_model_blip2_t5_pretrain_flant5xl(self):
        """
        Load BLIP2 model with FLAN-T5 XL architecture.

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
            name="blip2_t5",
            model_type="pretrain_flant5xl",
            is_eval=True,
            device=self.summary_device,
        )
        return summary_vqa_model, summary_vqa_vis_processors, summary_vqa_txt_processors

    def load_model_blip2_t5_caption_coco_flant5xl(self):
        """
        Load BLIP2 model with caption_coco_flant5xl architecture.

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
            name="blip2_t5",
            model_type="caption_coco_flant5xl",
            is_eval=True,
            device=self.summary_device,
        )
        return summary_vqa_model, summary_vqa_vis_processors, summary_vqa_txt_processors

    def load_model_blip2_opt_pretrain_opt27b(self):
        """
        Load BLIP2 model with pretrain_opt2 architecture.

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
            name="blip2_opt",
            model_type="pretrain_opt2.7b",
            is_eval=True,
            device=self.summary_device,
        )
        return summary_vqa_model, summary_vqa_vis_processors, summary_vqa_txt_processors

    def load_model_base_blip2_opt_pretrain_opt67b(self):
        """
        Load BLIP2 model with pretrain_opt6.7b architecture.

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
            name="blip2_opt",
            model_type="pretrain_opt6.7b",
            is_eval=True,
            device=self.summary_device,
        )
        return summary_vqa_model, summary_vqa_vis_processors, summary_vqa_txt_processors

    def load_model_blip2_opt_caption_coco_opt27b(self):
        """
        Load BLIP2 model with caption_coco_opt2.7b architecture.

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
            name="blip2_opt",
            model_type="caption_coco_opt2.7b",
            is_eval=True,
            device=self.summary_device,
        )
        return summary_vqa_model, summary_vqa_vis_processors, summary_vqa_txt_processors

    def load_model_base_blip2_opt_caption_coco_opt67b(self):
        """
        Load BLIP2 model with caption_coco_opt6.7b architecture.

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
            name="blip2_opt",
            model_type="caption_coco_opt6.7b",
            is_eval=True,
            device=self.summary_device,
        )
        return summary_vqa_model, summary_vqa_vis_processors, summary_vqa_txt_processors
