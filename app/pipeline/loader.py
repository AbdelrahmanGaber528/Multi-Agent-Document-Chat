from pathlib import Path
from uuid import uuid4
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from docx import Document as DocxDocument
from docx.text.paragraph import Paragraph
from docx.table import Table

from app.core.errors import (
    UnsupportedFileError,
    FileTooLargeError,
    ParsingError,
    EmptyDocumentError,
)
from app.core.config import get_settings

settings = get_settings()

MAX_FILE_SIZE = settings.max_upload_bytes

SUPPORTED_EXTENSIONS = settings.allowed_extensions



def validate_file(file_path: str) -> Path:
    
    path = Path(file_path)

    if not path.exists() or not path.is_file():
        raise ParsingError(
            "File does not exist.",
            details={"filename": path.name},
        )

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileError(
            f"Unsupported extension: {path.suffix}",
            details={"supported": list(SUPPORTED_EXTENSIONS)},
        )

    size = path.stat().st_size

    if size == 0:
        raise EmptyDocumentError("Uploaded file is empty.")

    if size > MAX_FILE_SIZE:
        raise FileTooLargeError(
            "File exceeds the maximum allowed size.",
            details={"size_bytes": size, "max_size_bytes": max_size},
        )

    return path



def load_pdf(path: Path, doc_id: str) -> list[Document]:
    
    try:
        loader = PyMuPDFLoader(str(path))
        documents = loader.load()

        for document in documents:
            document.metadata.update({
                "doc_id": doc_id,
                "doc_name": path.name,
                "file_type": "pdf",
                "page": document.metadata.get("page", 0) + 1,
            })

        return documents

    except Exception as exc:

        raise ParsingError(
            "Failed to load PDF.",
            details={"filename": path.name, "reason": str(exc)},
        ) from exc



def load_txt(path: Path, doc_id: str) -> list[Document]:
    
    try:
        loader = TextLoader(
            str(path),
            encoding="utf-8-sig",
            autodetect_encoding=False,
        )

        documents = loader.load()

        for document in documents:
            document.metadata.update({
                "doc_id": doc_id,
                "doc_name": path.name,
                "file_type": "txt",
                "page": None,
            })

        return documents

    except Exception as exc:
        raise ParsingError(
            "Failed to load TXT.",
            details={"filename": path.name, "reason": str(exc)},
        ) from exc



def load_docx(path: Path, doc_id: str) -> list[Document]:

    try:
        document = DocxDocument(str(path))
        blocks = []

        for element in document.element.body:

            if element.tag.endswith("}p"):
                paragraph = Paragraph(element, document)

                if paragraph.text.strip():
                    blocks.append(
                        Document(
                            page_content=paragraph.text,
                            metadata={
                                "doc_id": doc_id,
                                "doc_name": path.name,
                                "file_type": "docx",
                                "page": None,
                                "block_type": "paragraph",
                                "style": paragraph.style.name,
                            },
                        )
                    )

            elif element.tag.endswith("}tbl"):
                table = Table(element, document)

                rows = [
                    [cell.text.strip() for cell in row.cells]
                    for row in table.rows
                ]

                table_text = "\n".join(
                    " | ".join(row) for row in rows
                )

                if table_text.strip():
                    blocks.append(
                        Document(
                            page_content=table_text,
                            metadata={
                                "doc_id": doc_id,
                                "doc_name": path.name,
                                "file_type": "docx",
                                "page": None,
                                "block_type": "table",
                                "table_rows": rows,
                            },
                        )
                    )

        return blocks

    except Exception as exc:
        raise ParsingError(
            "Failed to load DOCX.",
            details={"filename": path.name, "reason": str(exc)},
        ) from exc



## Main loading method
def load_document(file_path: str | Path) -> list[Document]:
    
    path = validate_file(file_path)
    doc_id = str(uuid4())

    loaders = {
        ".pdf": load_pdf,
        ".docx": load_docx,
        ".txt": load_txt,
    }

    documents = loaders[path.suffix.lower()](path, doc_id)

    if not documents or not any(document.page_content.strip() for document in documents):
        
        raise EmptyDocumentError(
            "No readable content found.",
            details={"filename": path.name},
        )

    return documents