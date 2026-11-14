from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from app.agents.analyst.agent import run_analyst
from app.agents.answer.agent import run_answer
from app.core.contract import EvidenceBundle
from app.orchestrator.state import OrchestratorState, merge_chunks, new_gaps

RetrieverFn = Callable[[str, list[dict[str, Any]]], EvidenceBundle]


def _build_cumulative_bundle(state: OrchestratorState) -> EvidenceBundle:
    """Reconstruct a deduplicated EvidenceBundle from accumulated state."""
    return EvidenceBundle(
        query=state["question"],
        rewritten_queries=list(state["rewritten_queries"]),
        chunks=list(state["chunks_by_id"].values()),
    )


def evidence_assessment(state: OrchestratorState) -> Literal["retriever", "answer"]:
    """Route to another retrieval round only when unattempted gaps and budget remain."""
    analysis = state["analysis"]
    if analysis is None or analysis.is_sufficient:
        return "answer"

    unattempted = new_gaps(analysis.missing_info, state["attempted_queries"])
    if unattempted and state["rounds_used"] < state["max_rounds"]:
        return "retriever"

    return "answer"


@dataclass(frozen=True)
class OrchestratorNodes:
    """Binds Part A retriever and Part B LLMs to LangGraph node callables."""

    retrieve_evidence: RetrieverFn
    analyst_llm: Any
    answer_llm: Any

    def retriever_node(self, state: OrchestratorState) -> dict[str, Any]:
        """Execute initial query or all unattempted missing-info queries for this round."""
        analysis = state["analysis"]
        queries = (
            [state["question"]]
            if analysis is None
            else new_gaps(analysis.missing_info, state["attempted_queries"])
        )

        incoming_chunks = {}
        rewritten: list[str] = []

        for query in queries:
            bundle = self.retrieve_evidence(query, state["history"])
            rewritten.extend(bundle.rewritten_queries)
            incoming_chunks = merge_chunks(
                incoming_chunks,
                {c.chunk_id: c for c in bundle.chunks},
            )

        return {
            "current_query": queries[-1],
            "rounds_used": state["rounds_used"] + 1,
            "attempted_queries": queries,
            "chunks_by_id": incoming_chunks,
            "rewritten_queries": rewritten,
        }

    def analyst_node(self, state: OrchestratorState) -> dict[str, Any]:
        """Run deterministic + LLM analysis over all accumulated evidence."""
        bundle = _build_cumulative_bundle(state)
        analysis = run_analyst(bundle, self.analyst_llm)
        return {"analysis": analysis}

    def answer_node(self, state: OrchestratorState) -> dict[str, Any]:
        """Synthesize the final cited answer and disclose any unresolved gaps."""
        analysis = state["analysis"]
        assert analysis is not None, "answer_node requires analyst_node to have run first."

        bundle = _build_cumulative_bundle(state)
        final_answer = run_answer(
            analysis=analysis,
            bundle=bundle,
            llm=self.answer_llm,
            rounds_used=state["rounds_used"],
        )
        return {
            "unresolved_gaps": list(analysis.missing_info),
            "final_answer": final_answer,
        }