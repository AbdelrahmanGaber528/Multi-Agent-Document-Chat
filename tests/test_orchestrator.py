from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from app.agents.analyst.agent import _AnalystSynthesis
from app.core.contract import EvidenceBundle, EvidenceChunk, Finding
from app.orchestrator.graph import run


def _chunk(
    chunk_id: str,
    text: str = "evidence",
    *,
    doc_id: str = "doc-1",
    doc_name: str = "report.pdf",
    page: int = 1,
    score: float = 0.8,
) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        doc_name=doc_name,
        page=page,
        text=text,
        score=score,
    )


class FakeAnalystLLM:
    def __init__(self, syntheses: list[_AnalystSynthesis]) -> None:
        self._syntheses = iter(syntheses)
        self.seen_chunk_ids: list[list[str]] = []

    def bind_tools(self, _tools: object) -> FakeAnalystLLM:
        return self

    def with_structured_output(self, _schema: object) -> FakeAnalystLLM:
        return self

    def invoke(self, messages: list[Any]) -> Any:
        # First call per round is the tool loop (2 messages); second call is synthesis (4 messages).
        if len(messages) == 2:
            return AIMessage(content="", tool_calls=[])
        return next(self._syntheses)


class FakeAnswerLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)

    def invoke(self, _messages: object) -> AIMessage:
        return AIMessage(content=next(self._responses))


def test_single_round_completes_when_evidence_is_sufficient() -> None:
    calls: list[tuple[str, list[dict[str, Any]]]] = []
    history = [{"role": "user", "content": "hi"}]

    def retriever(query: str, hist: list[dict[str, Any]]) -> EvidenceBundle:
        calls.append((query, hist))
        return EvidenceBundle(
            query=query,
            chunks=[_chunk("c1", "Revenue was $10M.", page=3)],
        )

    analyst_llm = FakeAnalystLLM(
        [
            _AnalystSynthesis(
                findings=[Finding(statement="Revenue was $10M.", chunk_ids=["c1"])],
                missing_info=[],
            )
        ]
    )
    answer_llm = FakeAnswerLLM(["Revenue reached $10M [c1]."])

    result = run(
        "What was revenue?",
        history,
        retrieve_evidence=retriever,
        analyst_llm=analyst_llm,
        answer_llm=answer_llm,
        max_rounds=3,
    )

    assert calls == [("What was revenue?", history)]
    assert result.rounds_used == 1
    assert result.insufficient_evidence is False
    assert result.missing_info == []
    assert result.text == "Revenue reached $10M (report.pdf, p. 3)."
    assert [c.chunk_id for c in result.citations] == ["c1"]


def test_multi_round_batches_gaps_and_merges_higher_scored_chunks() -> None:
    queries_called: list[str] = []

    def retriever(query: str, _hist: list[dict[str, Any]]) -> EvidenceBundle:
        queries_called.append(query)
        if query == "Compare Q3 and Q4":
            return EvidenceBundle(
                query=query,
                chunks=[_chunk("c1", "Q3 was $10M.", page=2, score=0.70)],
            )
        if query == "Q4 revenue":
            return EvidenceBundle(
                query=query,
                chunks=[
                    _chunk("c1", "Q3 was $10M.", page=2, score=0.95),
                    _chunk("c2", "Q4 was $12M.", page=4, score=0.90),
                ],
            )
        return EvidenceBundle(
            query=query,
            chunks=[_chunk("c3", "Q4 margin was 20%.", page=5, score=0.88)],
        )

    analyst_llm = FakeAnalystLLM(
        [
            _AnalystSynthesis(
                findings=[Finding(statement="Q3 revenue was $10M.", chunk_ids=["c1"])],
                missing_info=["Q4 revenue", "Q4 margin"],
            ),
            _AnalystSynthesis(
                findings=[
                    Finding(statement="Q3 revenue was $10M.", chunk_ids=["c1"]),
                    Finding(statement="Q4 revenue was $12M.", chunk_ids=["c2"]),
                    Finding(statement="Q4 margin was 20%.", chunk_ids=["c3"]),
                ],
                missing_info=[],
            ),
        ]
    )
    answer_llm = FakeAnswerLLM(["Q3 was $10M [c1], Q4 was $12M [c2] at 20% margin [c3]."])

    result = run(
        "Compare Q3 and Q4",
        retrieve_evidence=retriever,
        analyst_llm=analyst_llm,
        answer_llm=answer_llm,
        max_rounds=3,
    )

    assert queries_called == ["Compare Q3 and Q4", "Q4 revenue", "Q4 margin"]
    assert result.rounds_used == 2
    assert result.insufficient_evidence is False
    assert [c.chunk_id for c in result.citations] == ["c1", "c2", "c3"]


def test_stops_early_when_remaining_gaps_were_already_attempted() -> None:
    queries_called: list[str] = []

    def retriever(query: str, _hist: list[dict[str, Any]]) -> EvidenceBundle:
        queries_called.append(query)
        return EvidenceBundle(
            query=query,
            chunks=[_chunk("c1", "Only Q3 data exists.", page=1)],
        )

    analyst_llm = FakeAnalystLLM(
        [
            _AnalystSynthesis(
                findings=[Finding(statement="Only Q3 data is available.", chunk_ids=["c1"])],
                missing_info=["Q4 guidance"],
            ),
            _AnalystSynthesis(
                findings=[Finding(statement="Only Q3 data is available.", chunk_ids=["c1"])],
                missing_info=["Q4 guidance"],
            ),
        ]
    )
    answer_llm = FakeAnswerLLM(["Only Q3 data is reported [c1]."])

    result = run(
        "What is Q4 guidance?",
        retrieve_evidence=retriever,
        analyst_llm=analyst_llm,
        answer_llm=answer_llm,
        max_rounds=5,
    )

    # Round 1 searches the question; Round 2 searches "Q4 guidance"; Rounds 3-5 are skipped.
    assert queries_called == ["What is Q4 guidance?", "Q4 guidance"]
    assert result.rounds_used == 2
    assert result.insufficient_evidence is True
    assert result.missing_info == ["Q4 guidance"]
    assert "### Information Not Found" in result.text


def test_stops_at_max_rounds_and_discloses_unresolved_gaps() -> None:
    queries_called: list[str] = []

    def retriever(query: str, _hist: list[dict[str, Any]]) -> EvidenceBundle:
        queries_called.append(query)
        return EvidenceBundle(
            query=query,
            chunks=[_chunk("c1", "Partial context.", page=1)],
        )

    analyst_llm = FakeAnalystLLM(
        [
            _AnalystSynthesis(
                findings=[Finding(statement="Partial context found.", chunk_ids=["c1"])],
                missing_info=["Gap A"],
            ),
            _AnalystSynthesis(
                findings=[Finding(statement="Partial context found.", chunk_ids=["c1"])],
                missing_info=["Gap B"],
            ),
        ]
    )
    answer_llm = FakeAnswerLLM(["Partial context was found [c1]."])

    result = run(
        "Initial question",
        retrieve_evidence=retriever,
        analyst_llm=analyst_llm,
        answer_llm=answer_llm,
        max_rounds=2,
    )

    assert queries_called == ["Initial question", "Gap A"]
    assert result.rounds_used == 2
    assert result.insufficient_evidence is True
    assert result.missing_info == ["Gap B"]
    assert "### Information Not Found" in result.text