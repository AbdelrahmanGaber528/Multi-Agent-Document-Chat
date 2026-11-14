import os

# Set dummy environment variables so Pydantic validation passes during module import time
os.environ.setdefault("LLM_API_KEY", "test-llm-key")
os.environ.setdefault("EMBEDDING_API_KEY", "test-embedding-key")

import pytest
from langchain_core.documents import Document
from app.core.errors import ParsingError

# Adjust the import path according to where your cleaner functions are located
from app.pipeline.cleaner import clean_text, clean_documents


# ==========================================
# TESTS FOR clean_text
# ==========================================

@pytest.mark.parametrize(
    "input_text, expected",
    [
        ("Hello   World\t\tfrom   here", "Hello World from here"),                 # Horizontal whitespace collapsing
        ("Line 1\r\nLine 2\rLine 3", "Line 1\nLine 2\nLine 3"),                     # Line ending normalization
        ("Hello\x00World", "HelloWorld"),                                           # Null character removal
        ("Paragraph 1\n\n\n\n\nParagraph 2", "Paragraph 1\n\nParagraph 2"),         # Excessive blank lines collapsing
        ("   Leading and trailing spaces   ", "Leading and trailing spaces"),         # Stripping
    ],
)
def test_clean_text_success(input_text, expected):
    """Test various text cleaning transformations."""
    assert clean_text(input_text) == expected


def test_clean_text_unicode_normalization():
    """Test that Unicode characters are normalized using NFC form."""
    # 'e' + combining acute accent (NFD) -> normalized to single 'é' (NFC)
    unnormalized = "e\u0301"
    normalized = "é"
    assert clean_text(unnormalized) == normalized


def test_clean_text_invalid_type():
    """Test that passing a non-string type raises a TypeError."""
    with pytest.raises(TypeError) as exc_info:
        clean_text(12345)  # type: ignore
    assert "Document content must be a string" in str(exc_info.value)


# ==========================================
# TESTS FOR clean_documents
# ==========================================

def test_clean_documents_success():
    """Test successful cleaning of a list of documents and filtering out empty docs."""
    docs = [
        Document(page_content="   Hello   World   ", metadata={"source": "doc1.txt"}),
        Document(page_content="   \n\n   ", metadata={"source": "empty.txt"}),  # Should be skipped
        Document(page_content="Test\r\nContent", metadata={"source": "doc2.txt"}),
    ]

    cleaned = clean_documents(docs)

    # Expect 2 documents (the empty whitespace doc should be filtered out)
    assert len(cleaned) == 2

    assert cleaned[0].page_content == "Hello World"
    assert cleaned[0].metadata["source"] == "doc1.txt"

    assert cleaned[1].page_content == "Test\nContent"
    assert cleaned[1].metadata["source"] == "doc2.txt"


def test_clean_documents_exception_handling(mocker):
    """Test that unexpected errors during document cleaning get wrapped into a ParsingError."""
    # Force page_content access to raise an unexpected exception
    bad_doc = mocker.Mock()
    type(bad_doc).page_content = mocker.PropertyMock(side_effect=Exception("Unexpected memory error"))

    with pytest.raises(ParsingError) as exc_info:
        clean_documents([bad_doc])
    assert "Document cleaning failed" in str(exc_info.value)