import os

# Set dummy environment variables so Pydantic validation passes during module import time
os.environ.setdefault("LLM_API_KEY", "test-llm-key")
os.environ.setdefault("EMBEDDING_API_KEY", "test-embedding-key")

import pytest
from langchain_core.documents import Document
from app.core.errors import EmbeddingError

# Adjust the import path according to where your embedding functions are located
from app.pipeline.embeddings import (
    get_embedding_model,
    embed_documents,
    embed_query,
)


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the lru_cache for get_embedding_model before and after each test."""
    get_embedding_model.cache_clear()
    yield
    get_embedding_model.cache_clear()


# ==========================================
# TESTS FOR get_embedding_model
# ==========================================

def test_get_embedding_model_success(mocker):
    """Test successful initialization and caching of the embedding model."""
    mock_hf = mocker.patch("app.pipeline.embeddings.HuggingFaceEmbeddings")

    model1 = get_embedding_model("test-model")
    model2 = get_embedding_model("test-model")

    # Verify HuggingFaceEmbeddings was instantiated correctly once (due to lru_cache)
    mock_hf.assert_called_once_with(
        model_name="test-model",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )
    assert model1 == model2


def test_get_embedding_model_error_wrapping(mocker):
    """Test that initialization failures are wrapped into an EmbeddingError."""
    mocker.patch(
        "app.pipeline.embeddings.HuggingFaceEmbeddings",
        side_effect=Exception("Model load failed")
    )

    with pytest.raises(EmbeddingError) as exc_info:
        get_embedding_model("invalid-model")
    assert "Failed to initialize embedding model" in str(exc_info.value)


# ==========================================
# TESTS FOR embed_documents
# ==========================================

def test_embed_documents_success(mocker):
    """Test successful embedding generation for a list of documents."""
    mock_model_instance = mocker.Mock()
    mock_model_instance.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
    mocker.patch("app.pipeline.embeddings.get_embedding_model", return_value=mock_model_instance)

    docs = [
        Document(page_content="Text one"),
        Document(page_content="Text two"),
    ]

    vectors = embed_documents(docs, model_name="test-model")

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    mock_model_instance.embed_documents.assert_called_once_with(["Text one", "Text two"])


def test_embed_documents_empty_list_error():
    """Test that passing an empty document list raises an EmbeddingError."""
    with pytest.raises(EmbeddingError) as exc_info:
        embed_documents([])
    assert "Cannot embed an empty document list" in str(exc_info.value)


def test_embed_documents_exception_handling(mocker):
    """Test that errors during document embedding are wrapped into an EmbeddingError."""
    mock_model_instance = mocker.Mock()
    mock_model_instance.embed_documents.side_effect = Exception("Batch timeout")
    mocker.patch("app.pipeline.embeddings.get_embedding_model", return_value=mock_model_instance)

    docs = [Document(page_content="Some text")]

    with pytest.raises(EmbeddingError) as exc_info:
        embed_documents(docs)
    assert "Failed to generate document embeddings" in str(exc_info.value)


# ==========================================
# TESTS FOR embed_query
# ==========================================

def test_embed_query_success(mocker):
    """Test successful embedding generation for a search query."""
    mock_model_instance = mocker.Mock()
    mock_model_instance.embed_query.return_value = [0.5, 0.6, 0.7]
    mocker.patch("app.pipeline.embeddings.get_embedding_model", return_value=mock_model_instance)

    vector = embed_query("What is AI?", model_name="test-model")

    assert vector == [0.5, 0.6, 0.7]
    mock_model_instance.embed_query.assert_called_once_with("What is AI?")


@pytest.mark.parametrize("empty_query", ["", "   ", "\t\n"])
def test_embed_query_empty_query_error(empty_query):
    """Test that an empty or whitespace-only query raises an EmbeddingError."""
    with pytest.raises(EmbeddingError) as exc_info:
        embed_query(empty_query)
    assert "Query cannot be empty" in str(exc_info.value)


def test_embed_query_exception_handling(mocker):
    """Test that errors during query embedding are wrapped into an EmbeddingError."""
    mock_model_instance = mocker.Mock()
    mock_model_instance.embed_query.side_effect = Exception("Vector generation failure")
    mocker.patch("app.pipeline.embeddings.get_embedding_model", return_value=mock_model_instance)

    with pytest.raises(EmbeddingError) as exc_info:
        embed_query("Valid query")
    assert "Failed to generate query embedding" in str(exc_info.value)