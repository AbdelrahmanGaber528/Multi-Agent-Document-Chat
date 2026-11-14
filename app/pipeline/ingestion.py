import logging
from uuid import uuid4
from pathlib import Path
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.errors import (
    IngestionError,
    EmptyDocumentError,
    UnsupportedFileError,
    VectorStoreError,
)
from app.pipeline.vector_db import VectorDatabase

from app.pipeline.loader import load_document
from app.pipeline.parser import parse_documents
from app.pipeline.cleaner import clean_documents
from app.pipeline.chunker import chunk_documents


log = logging.getLogger(__name__)

settings = get_settings()


def ingest_file(filename: str, file_bytes: bytes) -> dict:
    """
    Ingest an uploaded file into the vector database.

    Workflow:
        File bytes
        -> Loader
        -> Parser
        -> Cleaner
        -> Chunker
        -> VectorDatabase
    """

    if not filename:
        raise IngestionError("Filename cannot be empty.")

    if not file_bytes:
        raise EmptyDocumentError("Uploaded file is empty.")

    extension = Path(filename).suffix.lower()

    if extension not in settings.allowed_extensions:
        raise UnsupportedFileError(
            f"Unsupported file extension: {extension}"
        )

    if len(file_bytes) > settings.max_upload_bytes:
        raise IngestionError(
            f"File exceeds the {settings.max_upload_mb} MB limit."
        )

    doc_id = str(uuid4())

    try:
        log.info("Starting ingestion: %s", filename)

        # 1. Load file
        documents = load_file(
            filename=filename,
            file_bytes=file_bytes,
        )

        if not documents:
            raise EmptyDocumentError(
                "Loader returned no documents."
            )

        # 2. Parse documents
        documents = parse_documents(documents)

        # 3. Clean documents
        documents = clean_documents(documents)

        if not documents:
            raise EmptyDocumentError(
                "No content remaining after cleaning."
            )

        # 4. Attach source metadata
        for document in documents:
            document.metadata.update({
                "doc_id": doc_id,
                "filename": filename,
                "source": filename,
            })

        # 5. Split documents into chunks
        chunks = chunk_documents(documents)

        if not chunks:
            raise EmptyDocumentError(
                "Chunking produced no chunks."
            )

        # 6. Add stable chunk metadata
        for index, chunk in enumerate(chunks):
            chunk.metadata.update({
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": index,
            })

            if not chunk.metadata.get("chunk_id"):
                chunk.metadata["chunk_id"] = (
                    f"{doc_id}_{index}"
                )

        # 7. Store chunks in ChromaDB
        vector_db = VectorDatabase(
            embedding_model=settings.embedding_model,
        )

        inserted_ids = vector_db.add_documents(chunks)

        log.info(
            "Ingestion completed: %s, chunks=%d",
            filename,
            len(inserted_ids),
        )

        return {
            "status": "success",
            "doc_id": doc_id,
            "filename": filename,
            "total_chunks": len(inserted_ids),
            "chunk_ids": inserted_ids,
        }

    except (
        EmptyDocumentError,
        UnsupportedFileError,
        IngestionError,
        VectorStoreError,
    ):
        raise

    except Exception as exc:
        log.exception("Ingestion failed for %s", filename)

        raise IngestionError(
            f"Failed to ingest {filename}.",
            details={"reason": str(exc)},
        ) from exc