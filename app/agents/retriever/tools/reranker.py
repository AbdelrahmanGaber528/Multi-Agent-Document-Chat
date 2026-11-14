import json
import re
from langchain_core.tools import StructuredTool
from app.core.config import get_settings

settings = get_settings()

TOP_K = settings.top_k_reranked

def _lexical_score(query: str, content: str) -> float:
    query_terms = set(re.findall(r"\w+", query.lower()))
    content_terms = re.findall(r"\w+", content.lower())

    if not query_terms or not content_terms:
        return 0.0

    frequencies = {}

    for term in content_terms:
        frequencies[term] = frequencies.get(term, 0) + 1

    return sum(
        frequencies.get(term, 0) / len(content_terms)
        for term in query_terms
    )


def build_reranker_tool() -> StructuredTool:

    def reranker(query: str, candidates_json: str, top_k: int =TOP_K) -> str:
        """Rerank retrieved chunks by their relevance to the query."""

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        try:
            candidates = json.loads(candidates_json)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid candidates JSON.") from exc

        if not isinstance(candidates, list):
            raise ValueError("Candidates must be a JSON list.")

        if not candidates:
            return "[]"

        if settings.use_cross_encoder:
            from sentence_transformers import CrossEncoder

            if not hasattr(reranker, "_model"):
                reranker._model = CrossEncoder(
                    settings.cross_encoder_model
                )

            pairs = [
                [query, candidate.get("content", "")]
                for candidate in candidates
            ]

            scores = reranker._model.predict(pairs)

            for candidate, score in zip(candidates, scores):
                candidate["rerank_score"] = float(score)

        else:
            for candidate in candidates:
                candidate["rerank_score"] = _lexical_score(
                    query,
                    candidate.get("content", ""),
                )

        candidates.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return json.dumps(
            candidates[:top_k],
            ensure_ascii=False,
            default=str,
        )

    return StructuredTool.from_function(
        func=reranker,
        name="reranker",
        description=(
            "Reorder retrieved document chunks according to their "
            "relevance to the original question. Input candidates "
            "must be a JSON list returned by a search tool."
        ),
    )