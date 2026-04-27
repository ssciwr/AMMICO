import pytest
from PIL import Image
import numpy as np
import torch


def test_multimodal_embeddings_model_init(multimodal_embeddings_model_cpu):
    assert multimodal_embeddings_model_cpu.model is not None
    assert multimodal_embeddings_model_cpu.device == "cpu"
    assert multimodal_embeddings_model_cpu.embedding_dim == 1024


@pytest.mark.long
def test_multimodal_embeddings_model_init_cuda(multimodal_embeddings_model_cuda):
    assert multimodal_embeddings_model_cuda.model is not None
    assert multimodal_embeddings_model_cuda.device == "cuda"
    assert multimodal_embeddings_model_cuda.embedding_dim == 1024


def test_multimodal_embeddings_model_encode_text(multimodal_embeddings_model_cpu):
    texts = ["This is a test sentence.", "Another test sentence."]
    embeddings = multimodal_embeddings_model_cpu.encode_text(texts)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 1024)


@pytest.mark.long
def test_multimodal_embeddings_model_encode_text_cuda(multimodal_embeddings_model_cuda):
    texts = ["This is a test sentence.", "Another test sentence."]
    embeddings = multimodal_embeddings_model_cuda.encode_text(texts)
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (2, 1024)


def test_multimodal_embeddings_model_encode_text_truncate(
    multimodal_embeddings_model_cpu,
):
    texts = ["This is a test sentence."]
    embeddings = multimodal_embeddings_model_cpu.encode_text(texts, truncate_dim=128)
    assert embeddings.shape == (1, 128)


@pytest.mark.long
def test_multimodal_embeddings_model_encode_text_truncate_cuda(
    multimodal_embeddings_model_cuda,
):
    texts = ["This is a test sentence."]
    embeddings = multimodal_embeddings_model_cuda.encode_text(texts, truncate_dim=128)
    assert embeddings.shape == (1, 128)


def test_multimodal_embeddings_model_encode_text_invalid_truncate(
    multimodal_embeddings_model_cpu,
):
    texts = ["This is a test sentence."]
    with pytest.raises(ValueError):
        multimodal_embeddings_model_cpu.encode_text(texts, truncate_dim=32)


def test_multimodal_embeddings_model_encode_image(multimodal_embeddings_model_cpu):
    image = Image.new("RGB", (224, 224))
    embeddings = multimodal_embeddings_model_cpu.encode_image(image)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (1, 1024)


@pytest.mark.long
def test_multimodal_embeddings_model_encode_image_cuda(
    multimodal_embeddings_model_cuda,
):
    image = Image.new("RGB", (224, 224))
    embeddings = multimodal_embeddings_model_cuda.encode_image(image)
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (1, 1024)


def test_multimodal_embeddings_model_encode_image_truncate(
    multimodal_embeddings_model_cpu,
):
    image = Image.new("RGB", (224, 224))
    embeddings = multimodal_embeddings_model_cpu.encode_image(image, truncate_dim=128)
    assert embeddings.shape == (1, 128)


@pytest.mark.long
def test_multimodal_embeddings_model_encode_image_truncate_cuda(
    multimodal_embeddings_model_cuda,
):
    image = Image.new("RGB", (224, 224))
    embeddings = multimodal_embeddings_model_cuda.encode_image(image, truncate_dim=128)
    assert embeddings.shape == (1, 128)


def test_multimodal_embeddings_model_encode_image_invalid_truncate(
    multimodal_embeddings_model_cpu,
):
    image = Image.new("RGB", (224, 224))
    with pytest.raises(ValueError):
        multimodal_embeddings_model_cpu.encode_image(image, truncate_dim=32)


def test_multimodal_embeddings_model_encode_image_invalid_input(
    multimodal_embeddings_model_cpu,
):
    with pytest.raises(ValueError):
        multimodal_embeddings_model_cpu.encode_image("not_an_image")
