import os

# Set dummy environment variables so Pydantic validation passes during module import time
os.environ.setdefault("LLM_API_KEY", "test-llm-key")
os.environ.setdefault("EMBEDDING_API_KEY", "test-embedding-key")

from pathlib import Path
from unittest.mock import patch, Mock
import pytest
from langchain_core.documents import Document

from app.core.errors import (
    UnsupportedFileError,
    FileTooLargeError,
    ParsingError,
    EmptyDocumentError,
)

from app.pipeline.loader import (
    validate_file,
    load_pdf,
    load_txt,
    load_docx,
    load_document,
)


@pytest.fixture
def mock_settings(monkeypatch):
    """Fixture to mock MAX_FILE_SIZE and SUPPORTED_EXTENSIONS."""
    monkeypatch.setattr("app.pipeline.loader.MAX_FILE_SIZE", 1024)
    monkeypatch.setattr("app.pipeline.loader.SUPPORTED_EXTENSIONS", {".pdf", ".txt", ".docx"})


# ==========================================
# TESTS FOR validate_file
# ==========================================

def test_validate_file_success(tmp_path, mock_settings):
    """Test that a valid file passes validation and returns a Path object."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("Hello world", encoding="utf-8")

    result = validate_file(str(file_path))
    assert isinstance(result, Path)
    assert result == file_path


def test_validate_file_not_found(tmp_path):
    """Test that validating a non-existent file raises ParsingError."""
    fake_path = tmp_path / "non_existent.txt"

    with pytest.raises(ParsingError) as exc_info:
        validate_file(str(fake_path))
    assert "File does not exist" in str(exc_info.value)


def test_validate_file_unsupported_extension(tmp_path, mock_settings):
    """Test that a file with an unsupported extension raises UnsupportedFileError."""
    file_path = tmp_path / "test.exe"
    file_path.write_text("malicious content", encoding="utf-8")

    with pytest.raises(UnsupportedFileError) as exc_info:
        validate_file(str(file_path))
    assert "Unsupported extension" in str(exc_info.value)


def test_validate_file_empty(tmp_path, mock_settings):
    """Test that an empty file raises EmptyDocumentError."""
    file_path = tmp_path / "empty.txt"
    file_path.touch()  # 0 bytes

    with pytest.raises(EmptyDocumentError) as exc_info:
        validate_file(str(file_path))
    assert "Uploaded file is empty" in str(exc_info.value)


def test_validate_file_too_large(tmp_path, monkeypatch):
    """Test that a file exceeding MAX_FILE_SIZE raises FileTooLargeError."""
    monkeypatch.setattr("app.pipeline.loader.MAX_FILE_SIZE", 5)  # 5 bytes max
    monkeypatch.setattr("app.pipeline.loader.SUPPORTED_EXTENSIONS", {".txt"})

    file_path = tmp_path / "large.txt"
    file_path.write_text("This is definitely larger than 5 bytes", encoding="utf-8")

    with pytest.raises(FileTooLargeError) as exc_info:
        validate_file(str(file_path))
    assert "File exceeds the maximum allowed size" in str(exc_info.value)


# ==========================================
# TESTS FOR load_txt
# ==========================================

def test_load_txt_success(tmp_path):
    """Test successful loading and metadata updating for a TXT file."""
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Sample text content", encoding="utf-8")
    doc_id = "test-uuid-123"

    docs = load_txt(file_path, doc_id)

    assert len(docs) == 1
    assert docs[0].page_content == "Sample text content"
    assert docs[0].metadata["doc_id"] == doc_id
    assert docs[0].metadata["doc_name"] == "sample.txt"
    assert docs[0].metadata["file_type"] == "txt"
    assert docs[0].metadata["page"] is None


def test_load_txt_parsing_error(tmp_path):
    """Test that exceptions during text loading trigger a ParsingError wrapper."""
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Content", encoding="utf-8")

    with patch("app.pipeline.loader.TextLoader.load", side_effect=Exception("Disk read error")):
        with pytest.raises(ParsingError) as exc_info:
            load_txt(file_path, "uuid-123")
        assert "Failed to load TXT" in str(exc_info.value)


# ==========================================
# TESTS FOR load_pdf
# ==========================================

def test_load_pdf_success(tmp_path):
    """Test successful loading and metadata updating for a PDF file using mocked PyMuPDFLoader."""
    file_path = tmp_path / "sample.pdf"
    file_path.touch()
    doc_id = "test-pdf-uuid"

    mock_doc = Document(page_content="PDF Page 1", metadata={"page": 0})
    with patch("app.pipeline.loader.PyMuPDFLoader.load", return_value=[mock_doc]):
        docs = load_pdf(file_path, doc_id)

    assert len(docs) == 1
    assert docs[0].page_content == "PDF Page 1"
    assert docs[0].metadata["doc_id"] == doc_id
    assert docs[0].metadata["file_type"] == "pdf"
    assert docs[0].metadata["page"] == 1  # 0-indexed page + 1


def test_load_pdf_parsing_error(tmp_path):
    """Test that exceptions during PDF loading trigger a ParsingError wrapper."""
    file_path = tmp_path / "sample.pdf"
    file_path.touch()

    with patch("app.pipeline.loader.PyMuPDFLoader.load", side_effect=Exception("Corrupt PDF")):
        with pytest.raises(ParsingError) as exc_info:
            load_pdf(file_path, "uuid-123")
        assert "Failed to load PDF" in str(exc_info.value)


# ==========================================
# TESTS FOR load_docx
# ==========================================

def test_load_docx_success(tmp_path):
    """Test successful parsing of paragraphs and tables from a docx file."""
    file_path = tmp_path / "sample.docx"
    file_path.touch()
    doc_id = "test-docx-uuid"

    mock_paragraph = Mock()
    mock_paragraph.text = "Hello Paragraph"
    mock_paragraph.style.name = "Normal"
    mock_paragraph.element.tag = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"

    mock_cell = Mock()
    mock_cell.text = "Cell Data"
    mock_row = Mock()
    mock_row.cells = [mock_cell]
    mock_table = Mock()
    mock_table.rows = [mock_row]
    mock_table.element.tag = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tbl"

    mock_docx_instance = Mock()
    mock_docx_instance.element.body = [mock_paragraph.element, mock_table.element]

    with patch("app.pipeline.loader.DocxDocument", return_value=mock_docx_instance), \
         patch("app.pipeline.loader.Paragraph", return_value=mock_paragraph), \
         patch("app.pipeline.loader.Table", return_value=mock_table):
        docs = load_docx(file_path, doc_id)

    assert len(docs) == 2
    assert docs[0].page_content == "Hello Paragraph"
    assert docs[0].metadata["block_type"] == "paragraph"
    assert docs[0].metadata["style"] == "Normal"
    assert docs[1].page_content == "Cell Data"
    assert docs[1].metadata["block_type"] == "table"


def test_load_docx_parsing_error(tmp_path):
    """Test that exceptions during DOCX loading raise ParsingError."""
    file_path = tmp_path / "sample.docx"
    file_path.touch()

    with patch("app.pipeline.loader.DocxDocument", side_effect=Exception("Invalid docx zip")):
        with pytest.raises(ParsingError) as exc_info:
            load_docx(file_path, "uuid-123")
        assert "Failed to load DOCX" in str(exc_info.value)


# ==========================================
# TESTS FOR load_document (Orchestrator)
# ==========================================

def test_load_document_dispatch_txt(tmp_path, mock_settings):
    """Test the main `load_document` entrypoint correctly routes to `load_txt`."""
    file_path = tmp_path / "document.txt"
    file_path.write_text("Orchestration text", encoding="utf-8")

    mock_docs = [Document(page_content="Orchestration text")]
    with patch("app.pipeline.loader.load_txt", return_value=mock_docs):
        result = load_document(file_path)
    assert result == mock_docs


def test_load_document_empty_content_raises_error(tmp_path, mock_settings):
    """Test that if the loader returns empty or whitespace-only documents, EmptyDocumentError is raised."""
    file_path = tmp_path / "document.txt"
    file_path.write_text("   ", encoding="utf-8")

    mock_docs = [Document(page_content="   ")]
    with patch("app.pipeline.loader.load_txt", return_value=mock_docs):
        with pytest.raises(EmptyDocumentError) as exc_info:
            load_document(file_path)
        assert "No readable content found" in str(exc_info.value)