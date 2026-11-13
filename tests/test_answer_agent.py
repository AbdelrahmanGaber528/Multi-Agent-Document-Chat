from app.agents.answer.agent import run_answer
from app.agents.answer.tools.source_formatter import resolve_inline_citations
from app.core.contract import (
    AnalysisResult,
    Comparison,
    DocumentEntry,
    EvidenceBundle,
    EvidenceChunk,
    Finding,
)


class EmptyLLM:
    def invoke(self, _messages: object) -> object:
        return type("Response", (), {"content": ""})()


def _chunk(chunk_id: str = "c1") -> EvidenceChunk:
    return EvidenceChunk(chunk_id=chunk_id, doc_id="doc-1", doc_name="source.pdf", text="evidence")


def test_used_chunk_ids_includes_comparison_entry_ids() -> None:
    analysis = AnalysisResult(
        question="q",
        comparisons=[
            Comparison(
                topic="topic",
                entries=[
                    DocumentEntry(
                        doc_id="doc-1",
                        doc_name="source.pdf",
                        attributes={"Metric": "1"},
                        chunk_ids=["c1"],
                    )
                ],
            )
        ],
    )

    assert analysis.used_chunk_ids == ["c1"]


def test_invalid_multi_id_citation_marker_is_removed() -> None:
    resolved, cited_ids = resolve_inline_citations(
        "Result [c1, c2].",
        allowed_chunks=[_chunk()],
        bundle_chunk_ids={"c1", "c2"},
    )

    assert resolved == "Result."
    assert cited_ids == []


def test_missing_info_is_not_insufficient_before_max_rounds() -> None:
    result = run_answer(
        AnalysisResult(question="q", findings=[Finding(statement="Fact", chunk_ids=["c1"])]),
        EvidenceBundle(query="q", chunks=[_chunk()]),
        EmptyLLM(),
        missing_info_override=["More evidence"],
        max_rounds_reached=False,
    )

    assert result.insufficient_evidence is False
    assert "Information Not Found" not in result.text
