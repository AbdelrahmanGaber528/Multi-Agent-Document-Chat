import logging
import tempfile
from pathlib import Path
from uuid import uuid4

from app.core.config import get_settings
from app.core.contract import IngestResult
from app.core.errors import (
    IngestionError, EmptyDocumentError, UnsupportedFileError,
    FileTooLargeError, VectorStoreError,
)
from app.pipeline.vector_db import VectorDatabase
from app.pipeline.loader import load_document
from app.pipeline.parser import parse_documents
from app.pipeline.cleaner import clean_documents
from app.pipeline.chunker import chunk_documents

log = logging.getLogger(__name__)
settings = get_settings()

def ingest_file(filename: str, file_bytes: bytes) -> IngestResult:
    """Ingest uploaded bytes through loader -> parser -> cleaner -> chunker -> Chroma."""
    if not filename or not filename.strip():
        raise IngestionError("Filename cannot be empty.")
    if not file_bytes:
        raise EmptyDocumentError("Uploaded file is empty.")
    safe_name = Path(filename).name
    extension = Path(safe_name).suffix.lower()
    if extension not in settings.allowed_extensions:
        raise UnsupportedFileError(f"Unsupported file extension: {extension}")
    if len(file_bytes) > settings.max_upload_bytes:
        raise FileTooLargeError(
            f"File exceeds the {settings.max_upload_mb} MB limit.",
            details={"size_bytes": len(file_bytes), "max_size_bytes": settings.max_upload_bytes},
        )

    doc_id = str(uuid4())
    temp_path = None
    try:
        settings.ensure_dirs()
        # The existing loader consumes filesystem paths; materialize the upload temporarily.
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=extension, prefix=f"{doc_id}_",
            dir=settings.upload_dir, delete=False
        ) as temp_file:
            temp_file.write(file_bytes)
            temp_path = Path(temp_file.name)

        documents = load_document(temp_path, doc_id=doc_id)
        documents = parse_documents(documents)
        documents = clean_documents(documents)
        if not documents:
            raise EmptyDocumentError("No content remaining after cleaning.")

        for document in documents:
            document.metadata.update({
                "doc_id": doc_id,
                "doc_name": safe_name,
                "filename": safe_name,
                "source": safe_name,
            })

        chunks = chunk_documents(documents)
        if not chunks:
            raise EmptyDocumentError("Chunking produced no chunks.")

        for index, chunk in enumerate(chunks):
            chunk.metadata.update({
                "doc_id": doc_id,
                "doc_name": safe_name,
                "filename": safe_name,
                "chunk_index": index,
            })
            chunk.metadata.setdefault("chunk_id", f"{doc_id}_{index}")

        db = VectorDatabase(embedding_model=settings.embedding_model)
        inserted_ids = db.add_documents(chunks)
        log.info("Ingestion completed: %s, chunks=%d", safe_name, len(inserted_ids))
        return IngestResult(doc_id=doc_id, num_chunks=len(inserted_ids))
    except (EmptyDocumentError, UnsupportedFileError, FileTooLargeError, IngestionError, VectorStoreError):
        raise
    except Exception as exc:
        log.exception("Ingestion failed for %s", safe_name)
        raise IngestionError(
            f"Failed to ingest {safe_name}.", details={"reason": str(exc)}
        ) from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
