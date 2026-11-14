import os

# Set dummy environment variables so Pydantic validation passes during module import time
os.environ.setdefault("LLM_API_KEY", "test-llm-key")
os.environ.setdefault("EMBEDDING_API_KEY", "test-embedding-key")

import pytest
from langchain_core.documents import Document
from app.core.errors import ChunkingError

# Adjust the import path according to where your chunking function is located
from app.pipeline.chunker import chunk_documents


# ==========================================
# TESTS FOR VALIDATION
# ==========================================

def test_chunk_documents_invalid_chunk_size():
    """Test that a non-positive chunk size raises ChunkingError."""
    docs = [Document(page_content="Some text", metadata={"doc_id": "1"})]

    with pytest.raises(ChunkingError) as exc_info:
        chunk_documents(docs, chunk_size=0)
    assert "Chunk size must be positive" in str(exc_info.value)

    with pytest.raises(ChunkingError) as exc_info:
        chunk_documents(docs, chunk_size=-10)
    assert "Chunk size must be positive" in str(exc_info.value)


@pytest.mark.parametrize("size, overlap", [
    (100, -5),   # Negative overlap
    (100, 100),  # Overlap equal to chunk size
    (100, 150),  # Overlap greater than chunk size
])
def test_chunk_documents_invalid_chunk_overlap(size, overlap):
    """Test that invalid chunk overlaps raise ChunkingError."""
    docs = [Document(page_content="Some text", metadata={"doc_id": "1"})]

    with pytest.raises(ChunkingError) as exc_info:
        chunk_documents(docs, chunk_size=size, chunk_overlap=overlap)
    assert "Chunk overlap must be non-negative and smaller than chunk size" in str(exc_info.value)


# ==========================================
# TESTS FOR SUCCESSFUL CHUNKING & LOGIC
# ==========================================

def test_chunk_documents_success_with_headings_and_tables():
    """Test successful text chunking, heading section memory per doc_id, and table preservation."""
    docs = [
        Document(page_content="Chapter 1: Intro", metadata={"doc_id": "doc_a", "chunk_type": "heading"}),
        Document(page_content="This is a long sentence that needs splitting into smaller pieces.", metadata={"doc_id": "doc_a", "chunk_type": "text"}),
        Document(page_content="Col1 | Col2\nVal1 | Val2", metadata={"doc_id": "doc_b", "chunk_type": "table"}),
    ]

    chunks = chunk_documents(docs, chunk_size=25, chunk_overlap=5)

    assert len(chunks) >= 2

    text_chunk = next(c for c in chunks if "long sentence" in c.page_content)
    assert text_chunk.metadata["section"] == "Chapter 1: Intro"
    assert text_chunk.metadata["chunk_type"] == "text"
    assert text_chunk.metadata["chunk_id"].startswith("doc_a_")

    table_chunk = next(c for c in chunks if "Col1 | Col2" in c.page_content)
    assert table_chunk.metadata["chunk_type"] == "table"
    assert table_chunk.metadata["section"] == "Unknown"
    assert table_chunk.metadata["chunk_id"].startswith("doc_b_")


def test_chunk_documents_no_chunks_error():
    """Test that empty chunks resulting from blank page contents raise ChunkingError."""
    docs = [
        Document(page_content="   ", metadata={"doc_id": "1", "chunk_type": "text"}),
    ]

    with pytest.raises(ChunkingError) as exc_info:
        chunk_documents(docs)
    # Since whitespace-only results in zero pieces, it hits the outer try-except wrapper
    assert "Failed to split documents into chunks" in str(exc_info.value)


# ==========================================
# TESTS FOR EXCEPTION HANDLING
# ==========================================

def test_chunk_documents_exception_handling(mocker):
    """Test that unexpected errors during execution get wrapped into a ChunkingError."""
    # Patch split_documents at the source library module where it is defined
    mocker.patch(
        "langchain_text_splitters.RecursiveCharacterTextSplitter.split_documents",
        side_effect=Exception("Splitter failure")
    )

    docs = [Document(page_content="Normal text content", metadata={"doc_id": "1", "chunk_type": "text"})]

    with pytest.raises(ChunkingError) as exc_info:
        chunk_documents(docs)
    assert "Failed to split documents into chunks" in str(exc_info.value)