from ammico.utils import resolve_model_device

import torch
import warnings
from typing import Optional, List, Union
from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np


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
            model_kwargs={"torch_dtype": "auto"},
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
