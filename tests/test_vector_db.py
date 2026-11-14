import os

# Set dummy environment variables so Pydantic validation passes during module import time
os.environ.setdefault("LLM_API_KEY", "test-llm-key")
os.environ.setdefault("EMBEDDING_API_KEY", "test-embedding-key")

from pathlib import Path
import pytest
from langchain_core.documents import Document
from app.core.errors import VectorStoreError

# Corrected import to use vector_db instead of vector_store
from app.pipeline.vector_db import (
    VectorDatabase,
    _stable_chunk_id,
    _sanitize_metadata,
)


# ==========================================
# TESTS FOR HELPERS (_stable_chunk_id & _sanitize_metadata)
# ==========================================

def test_stable_chunk_id_prefers_chunk_id():
    """Test that _stable_chunk_id returns the existing chunk_id if present."""
    doc = Document(
        page_content="Some content",
        metadata={"chunk_id": "custom_chunk_123"}
    )
    assert _stable_chunk_id(doc) == "custom_chunk_123"


def test_stable_chunk_id_generates_hash():
    """Test that _stable_chunk_id generates a fallback hash when chunk_id is missing."""
    doc = Document(
        page_content="Some content",
        metadata={"doc_id": "doc_1", "chunk_index": 0}
    )
    chunk_id = _stable_chunk_id(doc)
    assert isinstance(chunk_id, str)
    assert len(chunk_id) == 64  # SHA-256 hex digest length


def test_sanitize_metadata():
    """Test that metadata is correctly sanitized for Chroma compatibility."""
    raw_metadata = {
        "valid_str": "hello",
        "valid_int": 42,
        "valid_float": 3.14,
        "valid_bool": True,
        "none_value": None,                  # Should be excluded
        "list_value": [1, 2, 3],             # Should be JSON serialized
        "dict_value": {"nested": "data"},    # Should be JSON serialized
        "other_value": Path("/tmp"),         # Should be cast to string
    }

    sanitized = _sanitize_metadata(raw_metadata)

    assert sanitized["valid_str"] == "hello"
    assert sanitized["valid_int"] == 42
    assert sanitized["valid_float"] == 3.14
    assert sanitized["valid_bool"] is True
    assert "none_value" not in sanitized
    assert sanitized["list_value"] == "[1, 2, 3]"
    assert sanitized["dict_value"] == '{"nested": "data"}'
    assert sanitized["other_value"] == "/tmp"


# ==========================================
# TESTS FOR VectorDatabase
# ==========================================

def test_vector_database_initialization_error(mocker):
    """Test that initialization failures are wrapped into a VectorStoreError."""
    mocker.patch(
        "app.pipeline.vector_db.get_embedding_model",
        side_effect=Exception("Model loading error")
    )

    with pytest.raises(VectorStoreError) as exc_info:
        VectorDatabase(persist_directory="/tmp/fake_dir")
    assert "Failed to initialize ChromaDB" in str(exc_info.value)


def test_add_documents_empty_error(tmp_path, mocker):
    """Test that adding an empty document list raises a VectorStoreError."""
    mocker.patch("app.pipeline.vector_db.get_embedding_model")
    mocker.patch("app.pipeline.vector_db.Chroma")

    db = VectorDatabase(persist_directory=str(tmp_path))

    with pytest.raises(VectorStoreError) as exc_info:
        db.add_documents([])
    assert "Cannot insert an empty document list" in str(exc_info.value)


def test_add_documents_success(tmp_path, mocker):
    """Test successful document insertion and id generation."""
    mock_chroma_instance = mocker.Mock()
    mocker.patch("app.pipeline.vector_db.Chroma", return_value=mock_chroma_instance)
    mocker.patch("app.pipeline.vector_db.get_embedding_model")

    db = VectorDatabase(persist_directory=str(tmp_path))

    docs = [
        Document(page_content="Doc 1 content", metadata={"chunk_id": "chunk_1", "doc_id": "doc_1"}),
        Document(page_content="Doc 2 content", metadata={"doc_id": "doc_2"}),  # Will use hash ID
    ]

    ids = db.add_documents(docs)

    assert len(ids) == 2
    assert ids[0] == "chunk_1"
    mock_chroma_instance.add_documents.assert_called_once()


def test_similarity_search_validation(tmp_path, mocker):
    """Test query and k validation rules during similarity search."""
    mocker.patch("app.pipeline.vector_db.Chroma")
    mocker.patch("app.pipeline.vector_db.get_embedding_model")

    db = VectorDatabase(persist_directory=str(tmp_path))

    # Empty query check
    with pytest.raises(VectorStoreError) as exc_info:
        db.similarity_search("   ", k=5)
    assert "Search query cannot be empty" in str(exc_info.value)

    # Invalid k check
    with pytest.raises(VectorStoreError) as exc_info:
        db.similarity_search("valid query", k=0)
    assert "k must be positive" in str(exc_info.value)


def test_similarity_search_success_with_filter(tmp_path, mocker):
    """Test successful similarity search with doc_id filtering."""
    mock_chroma_instance = mocker.Mock()
    mock_chroma_instance.similarity_search.return_value = [Document(page_content="Match")]
    mocker.patch("app.pipeline.vector_db.Chroma", return_value=mock_chroma_instance)
    mocker.patch("app.pipeline.vector_db.get_embedding_model")

    db = VectorDatabase(persist_directory=str(tmp_path))
    results = db.similarity_search("test query", k=3, doc_id="doc_123")

    assert len(results) == 1
    mock_chroma_instance.similarity_search.assert_called_once_with(
        query="test query",
        k=3,
        filter={"doc_id": "doc_123"},
    )


def test_delete_document_success(tmp_path, mocker):
    """Test successful document deletion by doc_id."""
    mock_chroma_instance = mocker.Mock()
    mocker.patch("app.pipeline.vector_db.Chroma", return_value=mock_chroma_instance)
    mocker.patch("app.pipeline.vector_db.get_embedding_model")

    db = VectorDatabase(persist_directory=str(tmp_path))
    db.delete_document("doc_123")

    mock_chroma_instance.delete.assert_called_once_with(where={"doc_id": "doc_123"})


def test_get_document_chunks_success(tmp_path, mocker):
    """Test successful retrieval of document chunks."""
    mock_chroma_instance = mocker.Mock()
    mock_chroma_instance.get.return_value = {
        "documents": ["Chunk 1", "Chunk 2"],
        "metadatas": [{"chunk_index": 0}, {"chunk_index": 1}],
    }
    mocker.patch("app.pipeline.vector_db.Chroma", return_value=mock_chroma_instance)
    mocker.patch("app.pipeline.vector_db.get_embedding_model")

    db = VectorDatabase(persist_directory=str(tmp_path))
    chunks = db.get_document_chunks("doc_123")

    assert len(chunks) == 2
    assert chunks[0].page_content == "Chunk 1"
    assert chunks[0].metadata["chunk_index"] == 0
    mock_chroma_instance.get.assert_called_once_with(
        where={"doc_id": "doc_123"},
        include=["documents", "metadatas"],
    )