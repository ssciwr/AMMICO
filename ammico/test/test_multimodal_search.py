import pytest
from unittest.mock import patch
from ammico.multimodal_search import MultimodalSearch
from ammico.model import MultimodalEmbeddingsModel

from PIL import Image
import numpy as np
from pathlib import Path


def test_index_images(multimodal_search_mock, tmp_path):
    # Mock image preparation
    with (
        patch("ammico.multimodal_search.prepare_image") as mock_prepare_image,
        patch("ammico.multimodal_search.load_image") as mock_load_image,
    ):
        mock_prepare_image.side_effect = lambda x: Image.new("RGB", (100, 100))
        mock_load_image.side_effect = lambda x: Image.new("RGB", (100, 100))

        # Test indexing images
        image_paths = [tmp_path / f"image_{i}.jpg" for i in range(10)]
        for path in image_paths:
            path.touch()

        multimodal_search_mock.index_images(images=image_paths)

        assert len(multimodal_search_mock.image_paths) == 10
        assert multimodal_search_mock.image_embeddings is not None
        assert multimodal_search_mock.faiss_index is not None


def test_save_embeddings(multimodal_search_mock, tmp_path):
    embeddings = np.random.rand(10, 128).astype("float32")
    multimodal_search_mock._save_embeddings(embeddings, save_path=tmp_path)

    saved_file = tmp_path / "image_embeddings.npy"
    assert saved_file.exists()


def test_save_faiss_index(multimodal_search_mock, tmp_path):
    embeddings = np.random.rand(10, 128).astype("float32")
    multimodal_search_mock._build_faiss_index(embeddings)
    multimodal_search_mock._save_faiss_index(save_path=tmp_path)

    saved_file = tmp_path / "faiss_index.index"
    assert saved_file.exists()


def test_multimodal_search_text_query(multimodal_search_mock):
    query = "example text query"
    embeddings = np.random.rand(10, 128).astype("float32")
    multimodal_search_mock._build_faiss_index(embeddings)
    items, scores = multimodal_search_mock.multimodal_search(
        query=query, query_type="text", return_paths=False
    )

    assert isinstance(items, list)
    assert isinstance(scores, list)


def test_multimodal_search_image_query(multimodal_search_mock, tmp_path):
    # Mock image preparation
    embeddings = np.random.rand(10, 128).astype("float32")
    multimodal_search_mock._build_faiss_index(embeddings)
    with (
        patch("ammico.multimodal_search.prepare_image") as mock_prepare_image,
        patch("ammico.multimodal_search.load_image") as mock_load_image,
    ):
        mock_prepare_image.side_effect = lambda x: Image.new("RGB", (100, 100))
        mock_load_image.side_effect = lambda x: Image.new("RGB", (100, 100))

        query_image = tmp_path / "query_image.jpg"
        query_image.touch()

        items, scores = multimodal_search_mock.multimodal_search(
            query=query_image, query_type="image", return_paths=False
        )

        assert isinstance(items, list)
        assert isinstance(scores, list)


@pytest.mark.long
def test_multimodal_search_combined_query(get_path):
    model = MultimodalEmbeddingsModel()
    mms = MultimodalSearch(model=model)
    mms.index_images(images=get_path, save_embeddings_and_indexes=False)
    queries = [
        {"text": "A person wearing a mask"},
        {"image": Path(get_path) / "IMG_2809.png"},
    ]
    results = mms.multimodal_batch_search(queries=queries, top_k=3)

    assert len(results) == 2
    for query_idx, (items, scores) in enumerate(results):
        assert len(items) > 0
        assert len(items) == len(scores)

    assert any("pexels-maksgelatin-4750169.jpg" in item for item in results[0][0])
    assert any("IMG_2809.png" in item for item in results[1][0])
    assert any("IMG_3758.png" in item for item in results[1][0])
