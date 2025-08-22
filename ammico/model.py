import torch
import warnings
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    AutoTokenizer,
)
from typing import Optional


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
        self.device = self._resolve_device(device)
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

    @staticmethod
    def _resolve_device(device: Optional[str]) -> str:
        if device is None:
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device.lower() not in ("cuda", "cpu"):
            raise ValueError("device must be 'cuda' or 'cpu'")
        if device.lower() == "cuda" and not torch.cuda.is_available():
            warnings.warn(
                "Although 'cuda' was requested, no CUDA device is available. Using CPU instead.",
                RuntimeWarning,
                stacklevel=2,
            )
            return "cpu"
        return device.lower()

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

    def _close(self) -> None:
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

    def close(self) -> None:
        """Free model resources (helpful in long-running processes)."""
        self._close()
