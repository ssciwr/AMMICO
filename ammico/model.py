from ammico.utils import (
    resolve_model_device,
    resolve_model_size,
    get_supported_whisperx_languages,
)

import torch
import warnings
import whisperx
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    AutoTokenizer,
)
from typing import Optional, List, Union
from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np


class MultimodalSummaryModel:
    DEFAULT_CUDA_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
    DEFAULT_CPU_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        """
        Class for QWEN-2.5-VL model loading and inference.
        Args:
            model_id: Type of model to load, defaults to a smaller version for CPU if device is "cpu".
            device: "cuda" or "cpu" (auto-detected when None).
            cache_dir: huggingface cache dir (optional).
        """
        self.device = resolve_model_device(device)

        if model_id is not None and model_id not in (
            self.DEFAULT_CUDA_MODEL,
            self.DEFAULT_CPU_MODEL,
        ):
            raise ValueError(
                f"model_id must be one of {self.DEFAULT_CUDA_MODEL} or {self.DEFAULT_CPU_MODEL}"
            )

        self.model_id = model_id or (
            self.DEFAULT_CUDA_MODEL if self.device == "cuda" else self.DEFAULT_CPU_MODEL
        )

        self.cache_dir = cache_dir
        self._trust_remote_code = True
        self._quantize = True

        self.model = None
        self.processor = None
        self.tokenizer = None

        self._load_model_and_processor()

    def _load_model_and_processor(self):
        load_kwargs = {"trust_remote_code": self._trust_remote_code, "use_cache": True}
        if self.cache_dir:
            load_kwargs["cache_dir"] = self.cache_dir

        self.processor = AutoProcessor.from_pretrained(
            self.model_id, padding_side="left", **load_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **load_kwargs)

        if self.device == "cuda":
            compute_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            )
            load_kwargs["quantization_config"] = bnb_config
            load_kwargs["device_map"] = "auto"

        else:
            load_kwargs.pop("quantization_config", None)
            load_kwargs.pop("device_map", None)

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, **load_kwargs
        )
        self.model.eval()

    def close(self) -> None:
        """Free model resources (helpful in long-running processes)."""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            if self.processor is not None:
                del self.processor
                self.processor = None
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
        finally:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                warnings.warn(
                    "Failed to empty CUDA cache. This is not critical, but may lead to memory lingering: "
                    f"{e!r}",
                    RuntimeWarning,
                    stacklevel=2,
                )


class AudioToTextModel:
    def __init__(
        self,
        model_size: str = "large",
        device: Optional[str] = None,
        language: Optional[str] = None,
    ) -> None:
        """
        Class for WhisperX model loading and inference.
        Args:
            model_size: Size of Whisper model to load (small, base, large).
            device: "cuda" or "cpu" (auto-detected when None).
            language: ISO-639-1 language code (e.g., "en", "fr", "de").
                     If None, language will be detected automatically.
                     Set this to avoid unreliable detection on small clips.
        """
        self.device = resolve_model_device(device)

        self.model_size = resolve_model_size(model_size)

        self.model = None

        self.language = self._validate_language(language)

        self._load_model()

    def _validate_language(self, language: Optional[str]) -> Optional[str]:
        """

        Validate the provided language code against whisperx's supported languages.
        Args:
            language: ISO-639-1 language code (e.g., "en", "fr", "de").
        Returns:
            Validated language code or None.
        Raises:
            ValueError: If the language code is invalid or unsupported.
        """

        if not language:
            return None

        language = language.strip().lower()
        supported_languages = get_supported_whisperx_languages()

        if len(language) != 2:
            raise ValueError(
                f"Invalid language code: '{language}'. Language codes must be 2 letters."
            )

        if not language.isalpha():
            raise ValueError(
                f"Invalid language code: '{language}'. Language codes must contain only alphabetic characters."
            )

        if language not in supported_languages:
            raise ValueError(
                f"Unsupported language code: '{language}'. Supported: {sorted(supported_languages)}"
            )

        return language

    def _load_model(self):
        if self.device == "cuda":
            self.model = whisperx.load_model(
                self.model_size,
                device=self.device,
                compute_type="float16",
                language=self.language,
            )
        else:
            self.model = whisperx.load_model(
                self.model_size,
                device=self.device,
                compute_type="int8",
                language=self.language,
            )

    def close(self) -> None:
        """Free model resources (helpful in long-running processes)."""
        try:
            if self.model is not None:
                del self.model
                self.model = None
        finally:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                warnings.warn(
                    "Failed to empty CUDA cache. This is not critical, but may lead to memory lingering: "
                    f"{e!r}",
                    RuntimeWarning,
                    stacklevel=2,
                )


class MultimodalEmbeddingsModel:
    def __init__(
        self,
        device: Optional[str] = None,
    ) -> None:
        """
        Class for Multimodal Embeddings model loading and inference. Uses Jina CLIP-V2 model.
        Args:
            device: "cuda" or "cpu" (auto-detected when None).
        """
        self.device = resolve_model_device(device)

        model_id = "jinaai/jina-clip-v2"

        self.model = SentenceTransformer(
            model_id,
            device=self.device,
            trust_remote_code=True,
            model_kwargs={"dtype": "auto"},
        )

        self.model.eval()

        self.embedding_dim = 1024

    @torch.inference_mode()
    def encode_text(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 64,
        truncate_dim: Optional[int] = None,
    ) -> Union[torch.Tensor, np.ndarray]:
        if isinstance(texts, str):
            texts = [texts]

        convert_to_tensor = self.device == "cuda"
        convert_to_numpy = not convert_to_tensor

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=convert_to_tensor,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=True,
        )

        if truncate_dim is not None:
            if not (64 <= truncate_dim <= self.embedding_dim):
                raise ValueError(
                    f"truncate_dim must be between 64 and {self.embedding_dim}"
                )
            embeddings = embeddings[:, :truncate_dim]
        return embeddings

    @torch.inference_mode()
    def encode_image(
        self,
        images: Union[Image.Image, List[Image.Image]],
        batch_size: int = 32,
        truncate_dim: Optional[int] = None,
    ) -> Union[torch.Tensor, np.ndarray]:
        if not isinstance(images, (Image.Image, list)):
            raise ValueError(
                "images must be a PIL.Image or a list of PIL.Image objects. Please load images properly."
            )

        convert_to_tensor = self.device == "cuda"
        convert_to_numpy = not convert_to_tensor

        embeddings = self.model.encode(
            images if isinstance(images, list) else [images],
            batch_size=batch_size,
            convert_to_tensor=convert_to_tensor,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=True,
        )

        if truncate_dim is not None:
            if not (64 <= truncate_dim <= self.embedding_dim):
                raise ValueError(
                    f"truncate_dim must be between 64 and {self.embedding_dim}"
                )
            embeddings = embeddings[:, :truncate_dim]

        return embeddings
