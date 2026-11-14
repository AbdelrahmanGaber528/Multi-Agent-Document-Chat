import json
from typing import Any
from langchain_core.tools import StructuredTool
from app.pipeline.vector_db import VectorDatabase
from app.core.config import get_settings


settings = get_settings()

TOP_K_SEMANTIC = settings.top_k_semantic



def build_metadata_filter_tool(vector_db: VectorDatabase) -> StructuredTool:

    def metadata_filter(query: str, filters: dict[str, Any], k: int = TOP_K_SEMANTIC ) -> str:
        """Search documents while applying metadata filters."""

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if not filters:
            raise ValueError("At least one metadata filter is required.")

        if k <= 0:
            raise ValueError("k must be positive.")

        documents = vector_db.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filters,
        )

        results = []

        for document in documents:
            results.append({
                "content": document.page_content,
                "metadata": document.metadata,
                "chunk_id": document.metadata.get("chunk_id"),
                "doc_id": document.metadata.get("doc_id"),
                "page": document.metadata.get("page"),
            })

        return json.dumps(
            results,
            ensure_ascii=False,
            default=str,
        )

    return StructuredTool.from_function(
        func=metadata_filter,
        name="metadata_filter",
        description=(
            "Retrieve semantically relevant chunks while restricting "
            "the search using document metadata. Use when the user "
            "specifies a document, page, author, date, or other metadata."
        ),
    )