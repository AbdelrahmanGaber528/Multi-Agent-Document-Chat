from __future__ import annotations
from typing import Annotated, Any, TypedDict

from app.core.contract import AnalysisResult, EvidenceChunk, FinalAnswer


def new_gaps(missing_info: list[str], attempted: list[str]) -> list[str]:
    """Return stripped, deduplicated gaps not yet present in attempted queries."""
    seen = {q.strip().lower() for q in attempted if q.strip()}
    unattempted: list[str] = []

    for raw in missing_info:
        cleaned = raw.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key not in seen:
            seen.add(key)
            unattempted.append(cleaned)

    return unattempted


def append_unique_queries(
    existing: list[str],
    incoming: list[str],
) -> list[str]:
    """Append incoming queries not already recorded in existing (case-insensitive)."""
    return [*existing, *new_gaps(incoming, existing)]


def merge_chunks(
    existing: dict[str, EvidenceChunk],
    incoming: dict[str, EvidenceChunk],
) -> dict[str, EvidenceChunk]:
    """Merge chunks by chunk_id, keeping the higher-scored chunk on collision."""
    merged = dict(existing)
    for chunk_id, chunk in incoming.items():
        current = merged.get(chunk_id)
        if current is None or chunk.score > current.score:
            merged[chunk_id] = chunk
    return merged


class OrchestratorState(TypedDict):
    """Shared state passed across LangGraph orchestrator nodes."""

    # Immutable inputs
    question: str
    history: list[dict[str, Any]]
    max_rounds: int

    # Loop control & query tracking
    current_query: str
    rounds_used: int
    attempted_queries: Annotated[list[str], append_unique_queries]

    # Cumulative retrieval memory
    chunks_by_id: Annotated[dict[str, EvidenceChunk], merge_chunks]
    rewritten_queries: Annotated[list[str], append_unique_queries]

    # Node outputs & exit metadata
    analysis: AnalysisResult | None
    unresolved_gaps: list[str]
    final_answer: FinalAnswer | None


def create_initial_state(
    question: str,
    history: list[dict[str, Any]],
    max_rounds: int,
) -> OrchestratorState:
    """Validate inputs and construct the initial orchestrator state."""
    clean_question = question.strip()
    if not clean_question:
        raise ValueError("question must be a non-empty string.")
    if max_rounds < 1:
        raise ValueError(f"max_rounds must be >= 1, got {max_rounds}.")

    return OrchestratorState(
        question=clean_question,
        history=list(history),
        max_rounds=max_rounds,
        current_query=clean_question,
        rounds_used=0,
        attempted_queries=[],
        chunks_by_id={},
        rewritten_queries=[],
        analysis=None,
        unresolved_gaps=[],
        final_answer=None,
    )