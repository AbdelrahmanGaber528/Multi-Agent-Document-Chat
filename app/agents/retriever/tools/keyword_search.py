import json
import re
from collections import Counter
from langchain_core.tools import StructuredTool
from app.pipeline.vector_db import VectorDatabase
from app.core.config import get_settings


settings = get_settings()
TOP_K_KEYWORD = settings.top_k_keyword



def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)




def build_keyword_search_tool(vector_db: VectorDatabase) -> StructuredTool:

    def keyword_search(query: str, doc_id: str | None = None, k: int = TOP_K_KEYWORD) -> str:
        """Find chunks containing important query terms."""

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if k <= 0:
            raise ValueError("k must be positive.")

        query_terms = set(_tokenize(query))

        if not query_terms:
            return "[]"

        where = {"doc_id": doc_id} if doc_id else None

        stored = vector_db.vector_store._collection.get(
            where=where,
            include=["documents", "metadatas"],
        )

        documents = stored.get("documents") or []
        metadatas = stored.get("metadatas") or []

        ranked = []

        for content, metadata in zip(documents, metadatas):
            content = content or ""
            metadata = metadata or {}

            content_terms = _tokenize(content)
            frequencies = Counter(content_terms)

            matched_terms = query_terms.intersection(frequencies)

            if not matched_terms:
                continue

            score = sum(
                frequencies[term] / max(len(content_terms), 1)
                for term in matched_terms
            )

            results = {
                "content": content,
                "metadata": metadata,
                "chunk_id": metadata.get("chunk_id"),
                "doc_id": metadata.get("doc_id"),
                "page": metadata.get("page"),
                "keyword_score": score,
                "matched_terms": sorted(matched_terms),
            }

            ranked.append(results)

        ranked.sort(
            key=lambda item: item["keyword_score"],
            reverse=True,
        )

        return json.dumps(
            ranked[:k],
            ensure_ascii=False,
            default=str,
        )

    return StructuredTool.from_function(
        func=keyword_search,
        name="keyword_search",
        description=(
            "Search document chunks using exact lexical term matching. "
            "Useful for names, identifiers, technical terms, and phrases "
            "that semantic search might miss."
        ),
    )