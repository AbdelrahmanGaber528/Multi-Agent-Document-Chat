from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.core.config import MAX_ROUNDS
from app.core.contract import FinalAnswer
from app.orchestrator.nodes import (
    OrchestratorNodes,
    RetrieverFn,
    evidence_assessment,
)
from app.orchestrator.state import OrchestratorState, create_initial_state


def build_graph(
    retrieve_evidence: RetrieverFn,
    analyst_llm: BaseChatModel | Any,
    answer_llm: BaseChatModel | Any,
) -> CompiledStateGraph:
    """Build and compile the multi-round RAG orchestrator StateGraph."""
    nodes = OrchestratorNodes(
        retrieve_evidence=retrieve_evidence,
        analyst_llm=analyst_llm,
        answer_llm=answer_llm,
    )

    workflow = StateGraph(OrchestratorState)
    workflow.add_node("retriever", nodes.retriever_node)
    workflow.add_node("analyst", nodes.analyst_node)
    workflow.add_node("answer", nodes.answer_node)

    workflow.add_edge(START, "retriever")
    workflow.add_edge("retriever", "analyst")
    workflow.add_conditional_edges(
        "analyst",
        evidence_assessment,
        {
            "retriever": "retriever",
            "answer": "answer",
        },
    )
    workflow.add_edge("answer", END)

    return workflow.compile()


def invoke_graph(
    app: CompiledStateGraph,
    question: str,
    history: list[dict[str, Any]] | None = None,
    *,
    max_rounds: int = MAX_ROUNDS,
) -> FinalAnswer:
    """Invoke a pre-compiled orchestrator graph for a single user turn."""
    initial_state = create_initial_state(
        question=question,
        history=history or [],
        max_rounds=max_rounds,
    )
    final_state = app.invoke(
        initial_state,
        config={"recursion_limit": max(25, 2 * max_rounds + 5)},
    )

    final_answer = final_state["final_answer"]
    assert final_answer is not None, "answer_node must populate final_answer."
    return final_answer


def run(
    question: str,
    history: list[dict[str, Any]] | None = None,
    *,
    retrieve_evidence: RetrieverFn,
    analyst_llm: BaseChatModel | Any,
    answer_llm: BaseChatModel | Any,
    max_rounds: int = MAX_ROUNDS,
) -> FinalAnswer:
    """Convenience entrypoint that builds and invokes the graph with injected dependencies."""
    app = build_graph(
        retrieve_evidence=retrieve_evidence,
        analyst_llm=analyst_llm,
        answer_llm=answer_llm,
    )
    return invoke_graph(app, question, history, max_rounds=max_rounds)