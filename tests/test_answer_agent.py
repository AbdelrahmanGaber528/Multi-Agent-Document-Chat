from app.agents.answer.agent import run_answer
from app.agents.answer.tools.citation_formatter import (
    build_citations,
    sanitize_analysis,
)
from app.agents.answer.tools.response_formatter import (
    append_missing_info_section,
    build_answer_prompt,
    extract_message_text,
    format_comparison_markdown,
    has_verified_content,
)
from app.agents.answer.tools.source_formatter import resolve_inline_citations
from app.core.contract import (
    AnalysisResult,
    Comparison,
    DocumentEntry,
    EvidenceBundle,
    EvidenceChunk,
    ExtractedTable,
    Finding,
)


class StaticLLM:
    def __init__(self, content: object) -> None:
        self._content = content

    def invoke(self, _messages: object) -> object:
        return type("Response", (), {"content": self._content})()


class FailingLLM:
    def invoke(self, _messages: object) -> object:
        raise RuntimeError("LLM provider timeout")


def _chunk(
    chunk_id: str = "c1",
    doc_id: str = "doc-1",
    doc_name: str = "source.pdf",
    page: int | None = None,
) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        doc_name=doc_name,
        page=page,
        text="evidence",
    )


def test_resolve_inline_citations_collapses_duplicates_and_preserves_markdown() -> None:
    c1 = _chunk("c1", doc_id="doc-1", doc_name="report.pdf", page=3)
    c2 = _chunk("c2", doc_id="doc-1", doc_name="report.pdf", page=3)
    c3 = _chunk("c3", doc_id="doc-2", doc_name="notes.pdf", page=7)

    text = (
        "Confirmed [c1][c2][c3] and ignored [c_bundle_only]. "
        "See [Report](https://example.com) and index [0]."
    )
    resolved, cited_ids = resolve_inline_citations(
        text,
        allowed_chunks=[c1, c2, c3],
        bundle_chunk_ids={"c1", "c2", "c3", "c_bundle_only"},
    )

    assert "(report.pdf, p. 3) (notes.pdf, p. 7)" in resolved
    assert "[c_bundle_only]" not in resolved
    assert "[Report](https://example.com)" in resolved
    assert "[0]" in resolved
    assert cited_ids == ["c1", "c2", "c3"]


def test_sanitize_analysis_drops_ungrounded_items_and_syncs_comparison_ids() -> None:
    analysis = AnalysisResult(
        question="q",
        findings=[
            Finding(statement="Grounded", chunk_ids=["c1", "c_bad"]),
            Finding(statement="Ungrounded", chunk_ids=["c_bad"]),
        ],
        comparisons=[
            Comparison(
                topic="Revenue",
                entries=[
                    DocumentEntry(
                        doc_id="doc-1",
                        doc_name="a.pdf",
                        attributes={"2025": "$10M"},
                        chunk_ids=["c1", "c_bad"],
                    ),
                    DocumentEntry(
                        doc_id="doc-2",
                        doc_name="b.pdf",
                        attributes={"2025": "$12M"},
                        chunk_ids=["c_bad"],
                    ),
                ],
                chunk_ids=["c1", "c_bad"],
            )
        ],
    )

    cleaned = sanitize_analysis(analysis, valid_chunk_ids={"c1"})

    assert len(cleaned.findings) == 1
    assert cleaned.findings[0].chunk_ids == ["c1"]
    assert len(cleaned.comparisons) == 1
    assert len(cleaned.comparisons[0].entries) == 1
    assert cleaned.comparisons[0].chunk_ids == ["c1"]


def test_has_verified_content_ignores_unreferenced_tables() -> None:
    analysis = AnalysisResult(
        question="q",
        tables=[
            ExtractedTable(
                chunk_id="c1",
                headers=["Col"],
                rows=[["Val"]],
            )
        ],
    )
    assert has_verified_content(analysis) is False


def test_format_comparison_markdown_escapes_pipes_and_newlines() -> None:
    comp = Comparison(
        topic="Metrics",
        entries=[
            DocumentEntry(
                doc_id="doc-1",
                doc_name="Doc | 1",
                attributes={"Note": "Line 1\nLine 2 | Extra"},
                chunk_ids=["c1"],
            )
        ],
        chunk_ids=["c1"],
    )
    md = format_comparison_markdown(comp)
    assert "\\|" in md
    assert "Line 1\nLine 2" not in md

def test_extract_message_text_handles_content_blocks() -> None:
    blocks = [
        {"type": "text", "text": "Part one. "},
        {"type": "text", "text": "Part two."},
    ]
    assert extract_message_text(blocks) == "Part one. Part two."


def test_append_missing_info_section_prevents_duplicate_heading() -> None:
    once = append_missing_info_section("Answer.", True, ["Q4 cash flow"])
    twice = append_missing_info_section(once, True, ["Q4 cash flow"])
    assert once.lower().count("### information not found") == 1
    assert twice.lower().count("### information not found") == 1


def test_run_answer_llm_exception_falls_back_with_grounded_citations() -> None:
    c1 = _chunk("c1", doc_name="filing.pdf", page=2)
    analysis = AnalysisResult(
        question="What was revenue?",
        findings=[Finding(statement="Revenue was $10M.", chunk_ids=["c1"])],
        missing_info=["Net income"],
    )

    result = run_answer(
        analysis=analysis,
        bundle=EvidenceBundle(query="What was revenue?", chunks=[c1]),
        llm=FailingLLM(),
        rounds_used=2,
        max_rounds_reached=True,
    )

    assert "Revenue was $10M." in result.text
    assert "(filing.pdf, p. 2)" in result.text
    assert "### Information Not Found" in result.text
    assert [c.chunk_id for c in result.citations] == ["c1"]
    assert result.insufficient_evidence is True
    assert result.rounds_used == 2

def test_run_answer_builds_prompt_without_unused_tables_and_resolves_blocks() -> None:
    c1 = _chunk("c1", doc_name="filing.pdf", page=5)
    c_unused = _chunk("c_unused", doc_name="appendix.pdf", page=9)
    analysis = AnalysisResult(
        question="q",
        findings=[Finding(statement="Metric is 42.", chunk_ids=["c1"])],
        tables=[
            ExtractedTable(
                chunk_id="c_unused",
                headers=["UnusedHeader"],
                rows=[["UnusedCell"]],
            )
        ],
    )

    prompt = build_answer_prompt(analysis)
    assert "[c1]" in prompt
    assert "UnusedHeader" not in prompt
    assert "UnusedCell" not in prompt

    llm = StaticLLM([{"type": "text", "text": "The metric is 42 [c1]."}])
    result = run_answer(
        analysis=analysis,
        bundle=EvidenceBundle(query="q", chunks=[c1, c_unused]),
        llm=llm,
    )

    assert result.text == "The metric is 42 (filing.pdf, p. 5)."
    assert [c.chunk_id for c in result.citations] == ["c1"]
    assert build_citations(["c1", "c1"], [c1, c_unused])[0].page == 5