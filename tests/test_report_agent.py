from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.agents.report.agent import ReportAgent, run_report
from app.agents.report.tools.cover import add_cover_page
from app.agents.report.tools.document import ReportDocument, make_new_document, save_document
from app.agents.report.tools.layout import add_page_break, add_spacer
from app.agents.report.tools.style import set_color_theme, set_font
from app.agents.report.tools.table import add_table
from app.agents.report.tools.text import add_bullet_list, add_heading, add_paragraph
from app.core.contract import (
    AnalysisResult,
    Citation,
    FinalAnswer,
    Finding,
    ReportRequest,
    ReportResult,
    ReportStyle,
)


@pytest.fixture
def sample_report_request(tmp_path: Path) -> ReportRequest:
    answer = FinalAnswer(
        text="Solaris Grid generated 45,000 MWh in 2025, surpassing Titan Power's 38,000 MWh.",
        citations=[
            Citation(chunk_id="c1", doc_id="d1", doc_name="Solaris_Grid_2025_Report.txt", page=2),
            Citation(chunk_id="c2", doc_id="d2", doc_name="Titan_Power_2025_Report.txt", page=1),
        ],
        insufficient_evidence=False,
        rounds_used=1,
    )
    analysis = AnalysisResult(
        question="Compare Solaris Grid and Titan Power generation.",
        findings=[
            Finding(statement="Solaris Grid generated 45,000 MWh.", chunk_ids=["c1"]),
            Finding(statement="Titan Power generated 38,000 MWh.", chunk_ids=["c2"]),
        ],
    )
    style = ReportStyle(
        title="Energy Generation Benchmark",
        subtitle="Solaris Grid vs Titan Power",
        primary_color="#1A237E",
        include_cover=True,
    )
    return ReportRequest(
        question="Compare Solaris Grid and Titan Power generation.",
        answer=answer,
        analysis=analysis,
        style=style,
    )


def test_report_tools_direct(tmp_path: Path):
    """Verify each design tool appends valid flowables and builds without errors."""
    output_pdf = str(tmp_path / "test_report.pdf")
    doc_ctx = ReportDocument()
    style = ReportStyle()
    make_new_document(doc_ctx, output_pdf, style)

    # 1. Cover page
    res_cover = add_cover_page(doc_ctx, "Test Report", "A subtitle")
    assert "Cover page created" in res_cover

    # 2. Heading
    res_h1 = add_heading(doc_ctx, "Executive Summary", level=1)
    assert "Heading level 1 added" in res_h1

    # 3. Paragraph
    res_p = add_paragraph(doc_ctx, "This is a <b>bold</b> and <i>italic</i> test paragraph.")
    assert "Paragraph added" in res_p

    # 4. Spacers & page breaks
    res_sp = add_spacer(doc_ctx, 10.0)
    assert "Spacer of 10.0mm added" in res_sp

    res_pb = add_page_break(doc_ctx)
    assert "Page break inserted" in res_pb

    # 5. Bullet list
    res_bl = add_bullet_list(doc_ctx, ["First key finding", "Second key finding"])
    assert "Bullet list with 2 items added" in res_bl

    # 6. Table
    res_tbl = add_table(
        doc_ctx,
        headers=["Company", "Generation (MWh)"],
        rows=[["Solaris Grid", "45,000"], ["Titan Power", "38,000"]],
        caption="2025 Production Comparison",
    )
    assert "Table added" in res_tbl

    # 7. Style updates
    res_font = set_font(doc_ctx, "heading1", "Times-Bold", 24)
    assert "Font for heading1 set to Times-Bold" in res_font

    res_theme = set_color_theme(doc_ctx, "#0D47A1", "#00838F")
    assert "Color theme updated" in res_theme

    # Build and verify PDF created on disk
    pages = save_document(doc_ctx)
    assert pages >= 1
    assert Path(output_pdf).exists()
    assert Path(output_pdf).stat().st_size > 0


def test_run_report_with_mock_llm(sample_report_request: ReportRequest):
    """Verify run_report agent loop executes and produces a valid ReportResult."""
    mock_llm = MagicMock()
    # Mock LLM returns empty tool calls so deterministic fallback builds the full report
    mock_ai_message = MagicMock()
    mock_ai_message.tool_calls = []
    mock_llm.bind_tools.return_value.invoke.return_value = mock_ai_message

    agent = ReportAgent(llm=mock_llm)
    result = agent.invoke(sample_report_request)

    assert isinstance(result, ReportResult)
    assert result.report_id is not None
    assert result.num_pages >= 1
    assert Path(result.path).exists()
    assert Path(result.path).stat().st_size > 0
