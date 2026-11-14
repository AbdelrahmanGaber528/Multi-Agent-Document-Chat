import json
import hashlib
from pathlib import Path
from typing import Any
from langchain_core.documents import Document
from langchain_chroma import Chroma
from app.pipeline.embeddings import get_embedding_model
from app.core.errors import VectorStoreError
from app.core.config import get_settings

settings = get_settings()

DEFAULT_PERSIST_DIRECTORY = settings.vector_db_dir
DEFAULT_COLLECTION_NAME = settings.vector_collection
EMBEDDING_MODEL = settings.embedding_model

def _stable_chunk_id(document: Document) -> str:
    """
    Generate a stable ID from document and chunk metadata.

    Prefer a chunk_id created by the chunker.
    """

    chunk_id = document.metadata.get("chunk_id")

    if chunk_id:
        return str(chunk_id)

    identity = "|".join([
        str(document.metadata.get("doc_id", "")),
        str(document.metadata.get("chunk_index", "")),
        document.page_content,
    ])

    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


# This function takes a metadata dict and cleans it so Chroma won’t reject it
def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Chroma metadata supports scalar values.
    Serialize nested structures to JSON strings.
    """

    sanitized = {}

    for key, value in metadata.items():

        if value is None:
            continue

        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value

        elif isinstance(value, (list, dict)):
            sanitized[key] = json.dumps(
                value,
                ensure_ascii=False,
            )

        else:
            sanitized[key] = str(value)

    return sanitized



class VectorDatabase:

    def __init__(
        self,
        persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
    ):

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            
            self.embedding_function = get_embedding_model(
                embedding_model
            )

            self.vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embedding_function,
                persist_directory=str(self.persist_directory),
                collection_metadata={
                    "hnsw:space": "cosine",
                },
            )

        except Exception as exc:
            raise VectorStoreError(
                "Failed to initialize ChromaDB.",
                details={"reason": str(exc)},
            ) from exc



    def add_documents(self, documents: list[Document]) -> list[str]:
        """Insert or update document chunks."""

        if not documents:
            raise VectorStoreError(
                "Cannot insert an empty document list."
            )

        try:
            prepared_documents = []
            ids = []

            for document in documents:

                prepared_documents.append(
                    Document(
                        page_content=document.page_content,
                        metadata=_sanitize_metadata(
                            document.metadata
                        ),
                    )
                )

                ids.append(_stable_chunk_id(document))

            self.vector_store.add_documents(
                documents=prepared_documents,
                ids=ids,
            )

            return ids

        except Exception as exc:
            raise VectorStoreError(
                "Failed to insert document chunks.",
                details={"reason": str(exc)},
            ) from exc



    def similarity_search(self, query: str, k: int = 5, doc_id: str | None = None) -> list[Document]:

        if not query.strip():
            raise VectorStoreError("Search query cannot be empty.")

        if k <= 0:
            raise VectorStoreError("k must be positive.")

        try:
            search_kwargs = {}

            if doc_id:
                search_kwargs["filter"] = {
                    "doc_id": doc_id,
                }

            return self.vector_store.similarity_search(
                query=query,
                k=k,
                **search_kwargs,
            )

        except Exception as exc:
            raise VectorStoreError(
                "Similarity search failed.",
                details={"reason": str(exc)},
            ) from exc



    def delete_document(self, doc_id: str) -> None:
        """Delete every chunk associated with a document."""

        try:
            self.vector_store.delete(
                where={"doc_id": doc_id}
            )

        except Exception as exc:
            raise VectorStoreError(
                "Failed to delete document chunks.",
                details={
                    "doc_id": doc_id,
                    "reason": str(exc),
                },
            ) from exc



    def get_document_chunks(self, doc_id: str) -> list[Document]:
        """Retrieve all chunks associated with a document."""

        try:
            results = self.vector_store.get(
                where={"doc_id": doc_id},
                include=["documents", "metadatas"],
            )

            return [
                Document(
                    page_content=content,
                    metadata=metadata or {},
                )
                for content, metadata in zip(
                    results["documents"] or [],
                    results["metadatas"] or [],
                )
            ]

        except Exception as exc:
            raise VectorStoreError(
                "Failed to retrieve document chunks.",
                details={
                    "doc_id": doc_id,
                    "reason": str(exc),
                },
            ) from exc