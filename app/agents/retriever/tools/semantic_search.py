import json
from langchain_core.tools import StructuredTool
from app.pipeline.vector_db import VectorDatabase
from app.core.config import get_settings


settings = get_settings()
TOP_K_SEMANTIC = settings.top_k_semantic

def build_semantic_search_tool(vector_db: VectorDatabase) -> StructuredTool:

    def semantic_search(query: str, doc_id: str | None = None, k: int = TOP_K_SEMANTIC) -> str:
        """Retrieve document chunks using embedding similarity."""

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if k <= 0:
            raise ValueError("k must be positive.")

        documents = vector_db.similarity_search(
            query=query,
            k=k,
            doc_id=doc_id,
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

        return json.dumps(results, ensure_ascii=False, default=str)

    return StructuredTool.from_function(
        func=semantic_search,
        name="semantic_search",
        description=(
            "Search document chunks using semantic similarity. "
            "Use this to find content related in meaning to the query. "
            "Optionally restrict the search to one document ID."
        ),
    )