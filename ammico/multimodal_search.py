from ammico.model import MultimodalEmbeddingsModel
from ammico.utils import AnalysisMethod, load_image, prepare_image, find_files
import torch
import faiss
from faiss.contrib import torch_utils
from typing import Optional, List, Union, Tuple, Any, Dict
from PIL import Image
import numpy as np
from pathlib import Path
import json


class MultimodalSearch(AnalysisMethod):
    def __init__(
        self,
        model: MultimodalEmbeddingsModel,
        subdict: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Class for multimodal search using embeddings.
        """
        if subdict is None:
            subdict = {}

        super().__init__(subdict)

        self.model = model
        self.image_paths: List[str] = []
        self.image_embeddings: Optional[Union[torch.Tensor, np.ndarray]] = None
        self.faiss_index = None
        self._default_save_dir: Optional[Path] = None

    def _prepare_images(
        self,
        images: Union[str, Path, List[Union[str, Path]]],
    ) -> tuple[List[Image.Image], List[str]]:
        """
        Load and prepare images from file paths or directory.
        Args:
            images: List of image file paths or a directory containing images.
        Returns:
            Tuple of (list of PIL Images, list of image paths).
        """

        if isinstance(images, (str, Path)):
            images_path = Path(images)
            if images_path.is_dir():
                images = find_files(images_path, return_as_list=True)
            else:
                images = [images_path]

        paths = [Path(img).resolve() for img in images]
        pil_images = [prepare_image(load_image(p)) for p in paths]

        return pil_images, [str(p) for p in paths]

    def _save_embeddings(
        self,
        embeddings: Union[torch.Tensor, np.ndarray],
        save_path: Optional[Union[str, Path]] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Save embeddings to a .npy file if embeddings are numpy array,
        or to a torch binary file if embeddings are torch tensor.
        Args:
            embeddings: Image embeddings to save.
            save_path: Directory to save embeddings.
            overwrite: Whether to overwrite existing files.
        """
        save_path = Path(save_path or self._default_save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        if isinstance(embeddings, torch.Tensor):
            filename = "image_embeddings.pt"
            out_path = save_path / filename
            if out_path.exists() and not overwrite:
                raise FileExistsError(
                    f"File already exists: {out_path}. Use overwrite=True to overwrite."
                )
            torch.save(embeddings, str(out_path))
        elif isinstance(embeddings, np.ndarray):
            filename = "image_embeddings.npy"
            out_path = save_path / filename
            if out_path.exists() and not overwrite:
                raise FileExistsError(
                    f"File already exists: {out_path}. Use overwrite=True to overwrite."
                )
            np.save(str(out_path), embeddings)
        else:
            raise ValueError(
                "Embeddings must be either a torch.Tensor or a numpy.ndarray."
            )

    def _save_faiss_index(
        self,
        save_path: Optional[Union[str, Path]] = None,
        filename: str = "faiss_index.index",
        overwrite: bool = False,
    ) -> None:
        """
        Save the Faiss index to a file.
        Args:
            save_path: Directory to save the Faiss index.
            filename: Name of the Faiss index file.
            overwrite: Whether to overwrite existing files.
        """
        if self.faiss_index is None:
            raise ValueError("Faiss index is not built yet.")

        save_path = Path(save_path or self._default_save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        out_path = save_path / filename
        if out_path.exists() and not overwrite:
            raise FileExistsError(
                f"File already exists: {out_path}. Use overwrite=True to overwrite."
            )

        if self.model.device == "cuda":
            faiss_index = faiss.index_gpu_to_cpu(
                self.faiss_index
            )  # convert to cpu index for saving
        else:
            faiss_index = self.faiss_index

        faiss.write_index(faiss_index, str(out_path))

    def _save_image_paths(
        self,
        save_path: Optional[Union[str, Path]] = None,
        filename: str = "image_paths.json",
        overwrite: bool = False,
    ) -> None:
        """Save image paths mapping to a JSON file.
        Args:
            save_path: Directory to save the image paths file.
            filename: Name of the image paths file.
            overwrite: Whether to overwrite existing files.
        """

        if not self.image_paths:
            raise ValueError("No image paths to save")

        save_path = Path(save_path or self._default_save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        out_path = save_path / filename
        if out_path.exists() and not overwrite:
            raise FileExistsError(
                f"File already exists: {out_path}. Use overwrite=True to overwrite."
            )

        with open(out_path, "w") as f:
            json.dump({"image_paths": self.image_paths}, f, indent=2)

    def _build_faiss_index(self, embeddings: Union[torch.Tensor, np.ndarray]) -> None:
        """Build a Faiss index from image embeddings."""

        if self.model.device == "cuda":
            res = faiss.StandardGpuResources()
            flat_config = faiss.GpuIndexFlatConfig()
            try:
                flat_config.device = torch.cuda.current_device()
            except Exception as e:
                flat_config.device = 0
            faiss_index = faiss.GpuIndexFlatIP(res, embeddings.shape[1], flat_config)
            faiss_index.add(embeddings.float().contiguous())
        else:
            faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
            faiss_index.add(embeddings.astype("float32"))

        self.faiss_index = faiss_index

    def index_images(
        self,
        images: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        batch_size: int = 32,
        truncate_dim: Optional[int] = None,
        save_embeddings_and_indexes: bool = False,
        save_path: Optional[Union[str, Path]] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Encode and index images given a list of image paths or a directory.
        If images is None, uses all images in self.subdict if available.
        Saves embeddings and Faiss index to disk if specified.
        Args:
            images: List of image file paths or a directory containing images.
            batch_size: Batch size for encoding images.
            truncate_dim: Dimension to truncate embeddings to.
            save_embeddings_and_indexes: Whether to save embeddings to disk.
            save_path: Path to save embeddings and index.
        """
        if save_path and not Path(save_path).exists():
            raise FileNotFoundError(f"Save path does not exist: {save_path}")

        if not images:
            if not self.subdict:
                raise ValueError("No images provided and subdict is empty")
            images = [
                v["filename"]
                for v in self.subdict.values()
                if isinstance(v, dict) and "filename" in v
            ]

        images_list, image_paths = self._prepare_images(images)
        self.image_paths = image_paths
        self._default_save_dir = (
            Path(save_path).resolve() if save_path else Path(image_paths[0]).parent
        )

        embeddings = self.model.encode_image(
            images=images_list,
            batch_size=batch_size,
            truncate_dim=truncate_dim,
        )

        self.image_embeddings = embeddings
        self._build_faiss_index(self.image_embeddings)
        if save_embeddings_and_indexes:
            self._save_embeddings(
                self.image_embeddings, self._default_save_dir, overwrite=overwrite
            )
            self._save_faiss_index(self._default_save_dir, overwrite=overwrite)
            self._save_image_paths(self._default_save_dir, overwrite=overwrite)

    def _load_image_paths(self, load_path: Union[str, Path]) -> None:
        """Load image paths mapping."""

        path = Path(load_path)
        path = path / "image_paths.json" if path.is_dir() else path
        if not path.exists():
            raise FileNotFoundError(f"Image paths not found: {path}")

        with open(path, "r") as f:
            data = json.load(f)

        self.image_paths = data["image_paths"]

    def _load_embeddings(
        self,
        embedding_path: Union[str, Path],
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Load precomputed image embeddings from a .npy or .pt file.
        - returns image_embeddings
        """
        path = Path(embedding_path)
        if not path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {path}")

        if path.suffix == ".npy":
            emb = np.load(str(path))
            if not isinstance(emb, np.ndarray) or emb.ndim != 2:
                raise ValueError("Loaded embeddings should be a 2D numpy array.")
            return emb
        elif path.suffix == ".pt":
            emb = torch.load(str(path))
            if not isinstance(emb, torch.Tensor) or emb.ndim != 2:
                raise ValueError("Loaded embeddings should be a 2D torch tensor.")
            return emb
        elif path.is_dir():
            np_path = path / "image_embeddings.npy"
            pt_path = path / "image_embeddings.pt"
            # We consider .npy only for cpu and .pt only for cuda because of logic of other parts of code
            if np_path.exists() and self.model.device == "cpu":
                emb = np.load(str(np_path))
                if not isinstance(emb, np.ndarray) or emb.ndim != 2:
                    raise ValueError("Loaded embeddings should be a 2D numpy array.")
                return emb
            elif pt_path.exists() and self.model.device == "cuda":
                emb = torch.load(str(pt_path))
                if not isinstance(emb, torch.Tensor) or emb.ndim != 2:
                    raise ValueError("Loaded embeddings should be a 2D torch tensor.")
                return emb
            else:
                raise FileNotFoundError(
                    f"No embeddings file found in directory: {path}"
                )
        else:
            raise ValueError(
                "Unsupported embedding file format. Use .npy or .pt files."
            )

    def _load_faiss_index(
        self,
        index_path: Union[str, Path],
        filename: str = "faiss_index.index",
    ) -> None:
        """
        Load a Faiss index from a file.
        """
        path = Path(index_path)
        path = path / filename if path.is_dir() else path
        if not path.exists():
            raise FileNotFoundError(f"Faiss index file not found: {path}")

        if self.model.device == "cpu":
            self.faiss_index = faiss.read_index(str(path))
        elif self.model.device == "cuda":
            res = faiss.StandardGpuResources()
            flat_config = faiss.GpuIndexFlatConfig()
            try:
                flat_config.device = torch.cuda.current_device()
            except Exception as e:
                flat_config.device = 0
            cpu_index = faiss.read_index(str(path))
            self.faiss_index = faiss.index_cpu_to_gpu(
                res, flat_config.device, cpu_index
            )

    def _load_indexes(
        self,
        load_path: Union[str, Path],
    ) -> None:
        """
        Load precomputed image embeddings and Faiss index from disk.
        Args:
            load_path: Directory or file path to load embeddings and index from.
        """
        if load_path is None:
            load_path = getattr(self, "_default_save_dir", None)
        if load_path is None:
            raise ValueError("load_path must be specified or set during indexing")

        path = Path(load_path)
        if not path.exists():
            raise FileNotFoundError(f"Load path does not exist: {path}")

        self._load_image_paths(path)
        # self.image_embeddings = self._load_embeddings(path) # since we have FAISS index, no need to load embeddings, it is possible to load it directly if needed
        self._load_faiss_index(path)

    def _encode_query(
        self,
        query: Union[str, Path, Image.Image],
        query_type: str = "text",
        batch_size: int = 64,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Encode a text or image query into an embedding.
        Args:
            query: Text string or image file path or PIL Image.
            query_type: "text" or "image".
            batch_size: Batch size for encoding.
        Returns:
            Query embedding as a torch.Tensor or numpy.ndarray.
        """
        if query_type == "text":
            emb = self.model.encode_text(query, batch_size=batch_size)
        elif query_type == "image":
            pil_images, _ = self._prepare_images(query)
            emb = self.model.encode_image(pil_images, batch_size=batch_size)
        else:
            raise ValueError("query_type must be either 'text' or 'image'")
        return emb

    def _search_embeddings(
        self,
        query_embeddings: Union[torch.Tensor, np.ndarray],
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        return_paths: bool = True,
    ) -> Union[
        Tuple[List[int], List[float]],
        Tuple[List[str], List[float]],
    ]:
        """
        Given query embeddings  find top_k matches in dataset embeddings.
        Returns (indices_per_query, scores_per_query)
        """

        if self.model.device == "cuda":
            query_gpu = query_embeddings.float().contiguous()
            distances, indices = self.faiss_index.search(query_gpu, top_k)
            distances = distances.cpu().numpy()
            indices = indices.cpu().numpy()
        else:
            q = query_embeddings.astype("float32")
            distances, indices = self.faiss_index.search(q, top_k)

        results = []
        for d_row, i_row in zip(distances, indices):
            items = [
                (int(i), float(d))
                for i, d in zip(i_row, d_row)
                if score_threshold is None or d >= score_threshold
            ]
            results.append(items)

        if return_paths:
            return (
                [[self.image_paths[i] for i, _ in r] for r in results],
                [[s for _, s in r] for r in results],
            )

        return (
            [[i for i, _ in r] for r in results],
            [[s for _, s in r] for r in results],
        )

    def multimodal_search(
        self,
        query: Union[str, Path, Image.Image],
        query_type: str = "text",
        top_k: int = 10,
        batch_size: int = 64,
        score_threshold: Optional[float] = None,
        load_indexes: bool = False,
        load_path: Optional[Union[str, Path]] = None,
        return_paths: bool = True,
    ) -> Union[
        Tuple[List[int], List[float]],
        Tuple[List[str], List[float]],
    ]:
        """
        Perform multimodal search given a query.
        Args:
            query: Text string or image file path or PIL Image.
            query_type: "text" or "image".
            top_k: Number of top matches to return.
            batch_size: Batch size for encoding the query.
            score_threshold: Minimum score threshold for matches.
            load_indexes: Whether to load precomputed indexes from disk.
            load_path: Path to load indexes from.
            return_paths: Whether to return image paths instead of indices.
        Returns:
            Tuple of (list of indices or paths, list of scores).
        """
        if load_indexes:
            self._load_indexes(load_path)

        if self.faiss_index is None:
            raise RuntimeError(
                "No image faiss_indexes found. Call index_images first or set load_indexes to True."
            )

        if return_paths and not self.image_paths:
            raise RuntimeError(
                "No image paths found. Cannot return paths. Ensure image paths are loaded."
            )

        q_emb = self._encode_query(
            query=query,
            query_type=query_type,
            batch_size=batch_size,
        )

        return self._search_embeddings(
            q_emb,
            top_k=top_k,
            score_threshold=score_threshold,
            return_paths=return_paths,
        )
