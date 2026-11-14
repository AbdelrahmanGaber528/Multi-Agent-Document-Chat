import json
from langchain_core.tools import StructuredTool
from app.core.config import get_setting

settings = get_settings()

MAX_CHUNKS = settings.final_context_chunks
TOKEN_BUDGET = settings.context_token_budget


def build_context_selector_tool() -> StructuredTool:

    def context_selector(candidates_json: str, max_chunks: int = MAX_CHUNKS, token_budget: int = TOKEN_BUDGET) -> str:
        """Select unique evidence chunks within the context budget."""

        if max_chunks <= 0:
            raise ValueError("max_chunks must be positive.")

        if token_budget <= 0:
            raise ValueError("token_budget must be positive.")

        try:
            candidates = json.loads(candidates_json)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid candidates JSON.") from exc

        if not isinstance(candidates, list):
            raise ValueError("Candidates must be a JSON list.")

        selected = []
        seen_chunks = set()
        used_tokens = 0

        for candidate in candidates:
            if len(selected) >= max_chunks:
                break

            content = candidate.get("content", "").strip()

            if not content:
                continue

            chunk_id = candidate.get("chunk_id")

            if chunk_id:
                if chunk_id in seen_chunks:
                    continue
                seen_chunks.add(chunk_id)

            estimated_tokens = max(1, len(content) // 4)

            if used_tokens + estimated_tokens > token_budget:
                continue

            selected.append(candidate)
            used_tokens += estimated_tokens

        return json.dumps(
            {
                "chunks": selected,
                "selected_chunks": len(selected),
                "estimated_tokens": used_tokens,
                "token_budget": token_budget,
            },
            ensure_ascii=False,
            default=str,
        )


    return StructuredTool.from_function(
        func=context_selector,
        name="context_selector",
        description=(
            "Select the best unique evidence chunks from reranked "
            "candidates while respecting the maximum chunk count "
            "and context token budget."
        ),
    )