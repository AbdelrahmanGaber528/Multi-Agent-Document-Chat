import os

# Set dummy environment variables so Pydantic validation passes during module import time
os.environ.setdefault("LLM_API_KEY", "test-llm-key")
os.environ.setdefault("EMBEDDING_API_KEY", "test-embedding-key")

import pytest
from langchain_core.documents import Document
from app.core.errors import ParsingError

from app.pipeline.parser import detect_heading, parse_documents


# ==========================================
# TESTS FOR detect_heading
# ==========================================

@pytest.mark.parametrize(
    "content, style, expected",
    [
        ("Introduction", "heading 1", True),         # Style-based match
        ("Title Page", "title", True),               # Style-title match
        ("1. Background", "normal", True),           # Numbered pattern match
        ("Chapter 2: Methods", "", True),            # Chapter pattern match
        ("Section 3.1: Analysis", "body", True),     # Section pattern match
        ("This is just a regular sentence.", "normal", False),  # Normal text
        ("", "heading 1", True),                     # Style takes precedence over empty text
        ("A" * 151, "heading 1", True),              # Style takes precedence over length check
    ],
)
def test_detect_heading(content, style, expected):
    """Test heading detection across different styles, lengths, and regex patterns."""
    doc = Document(page_content=content, metadata={"style": style})
    assert detect_heading(doc) == expected


# ==========================================
# TESTS FOR parse_documents
# ==========================================

def test_parse_documents_success():
    """Test successful parsing, section tracking, metadata defaults, and chunk type classification."""
    docs = [
        Document(page_content="   ", metadata={}),  # Empty block - should be skipped
        Document(page_content="Chapter 1: Overview", metadata={"doc_name": "report.pdf", "style": "heading 1"}),
        Document(page_content="This is introductory paragraph text.", metadata={"doc_id": "uuid-1"}),
        Document(page_content="Col1 | Col2\nVal1 | Val2", metadata={"block_type": "table"}),
    ]

    parsed = parse_documents(docs)

    assert len(parsed) == 3

    # 1. Check Heading Block
    assert parsed[0].page_content == "Chapter 1: Overview"
    assert parsed[0].metadata["chunk_type"] == "heading"
    assert parsed[0].metadata["section"] == "Chapter 1: Overview"
    assert parsed[0].metadata["block_index"] == 1
    assert parsed[0].metadata["doc_name"] == "report.pdf"
    assert parsed[0].metadata["doc_id"] is None  # Default fallback applied

    # 2. Check Text Block (inherits section from Chapter 1)
    assert parsed[1].page_content == "This is introductory paragraph text."
    assert parsed[1].metadata["chunk_type"] == "text"
    assert parsed[1].metadata["section"] == "Chapter 1: Overview"
    assert parsed[1].metadata["block_index"] == 2
    assert parsed[1].metadata["doc_id"] == "uuid-1"
    assert parsed[1].metadata["doc_name"] == "unknown"  # Default fallback applied
    assert parsed[1].metadata["page"] is None          # Default fallback applied

    # 3. Check Table Block
    assert parsed[2].metadata["chunk_type"] == "table"
    assert parsed[2].metadata["section"] == "Chapter 1: Overview"
    assert parsed[2].metadata["block_index"] == 3


def test_parse_documents_empty_content_raises_error():
    """Test that providing empty or whitespace-only documents raises ParsingError."""
    docs = [
        Document(page_content="", metadata={}),
        Document(page_content="   ", metadata={}),
    ]

    with pytest.raises(ParsingError) as exc_info:
        parse_documents(docs)
    assert "Document parsing failed" in str(exc_info.value)


def test_parse_documents_exception_handling(mocker):
    """Test that unexpected errors during parsing get wrapped into a ParsingError."""
    bad_doc = mocker.Mock()
    type(bad_doc).page_content = mocker.PropertyMock(side_effect=Exception("Unexpected read error"))

    with pytest.raises(ParsingError) as exc_info:
        parse_documents([bad_doc])
    assert "Document parsing failed" in str(exc_info.value)