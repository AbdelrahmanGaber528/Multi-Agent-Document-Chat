from functools import lru_cache
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.errors import EmbeddingError
from app.core.config import get_settings

settings = get_settings()


DEFAULT_MODEL = settings.embedding_model


@lru_cache(maxsize=4)
def get_embedding_model(model_name: str = DEFAULT_MODEL) -> HuggingFaceEmbeddings:
    """Load and cache the embedding model."""
    candidates = [model_name]
    fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
    if model_name != fallback_model:
        candidates.append(fallback_model)

    last_exc = None
    # Pass 1: Try local cache first to avoid hanging on network retries
    for candidate in candidates:
        try:
            return HuggingFaceEmbeddings(
                model_name=candidate,
                model_kwargs={
                    "device": "cpu",
                    "local_files_only": True,
                },
                encode_kwargs={
                    "normalize_embeddings": True,
                    "batch_size": 32,
                },
            )
        except Exception as exc:
            last_exc = exc
            continue

    # Pass 2: Try online download if local cache was not available
    for candidate in candidates:
        try:
            return HuggingFaceEmbeddings(
                model_name=candidate,
                model_kwargs={
                    "device": "cpu",
                    "local_files_only": False,
                },
                encode_kwargs={
                    "normalize_embeddings": True,
                    "batch_size": 32,
                },
            )
        except Exception as exc:
            last_exc = exc
            continue

    raise EmbeddingError(
        "Failed to initialize embedding model.",
        details={
            "model": model_name,
            "reason": str(last_exc),
        },
    ) from last_exc



def embed_documents(documents: list[Document],model_name: str = DEFAULT_MODEL) -> list[list[float]]:
    """Generate embeddings for document chunks."""

    if not documents:
        raise EmbeddingError("Cannot embed an empty document list.")

    try:
        model = get_embedding_model(model_name)

        texts = [
            document.page_content
            for document in documents
        ]

        vectors = model.embed_documents(texts)

        return vectors

    except Exception as exc:
        raise EmbeddingError(
            "Failed to generate document embeddings.",
            details={
                "model": model_name,
                "reason": str(exc),
            },
        ) from exc




def embed_query(query: str,model_name: str = DEFAULT_MODEL) -> list[float]:
    """Generate an embedding for a search query."""

    if not query.strip():
        raise EmbeddingError("Query cannot be empty.")

    try:
        
        model = get_embedding_model(model_name)
        embedding_query = model.embed_query(query)
        
        return embedding_query

    except Exception as exc:
        raise EmbeddingError(
            "Failed to generate query embedding.",
            details={"reason": str(exc)},
        ) from exc