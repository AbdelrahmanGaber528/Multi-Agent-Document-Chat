import json
import logging
from datetime import date
from typing import Any, Optional
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage

from app.agents.report.tools import build_report_tools
from app.agents.report.tools.document import ReportDocument, make_new_document, save_document
from app.core.config import get_settings
from app.core.contract import ReportRequest, ReportResult, ReportStyle
from app.core.errors import ConfigurationError, ReportError

logger = logging.getLogger(__name__)

MAX_TOOL_TURNS = 25

_REPORT_AGENT_SYSTEM_PROMPT = """You are the Report Generator Agent in an enterprise document intelligence system.
Your job is to generate a comprehensive, visually appealing PDF report by calling formatting tools.

RULES & GUIDELINES:
1. DESIGN & STRUCTURE:
   - If `include_cover` is True in the request, start with `add_cover_page(title=..., subtitle=...)`.
   - Organize content into logical sections with `add_heading(text_content=..., level=1)`.
   - Use `add_heading(text_content=..., level=2)` for subsections.
   - Separate major sections with `add_spacer(height_mm=8.0)` or `add_page_break()` for clean paging.
2. CONTENT ACCURACY:
   - Use ONLY facts, conclusions, calculations, tables, comparisons, and citations provided in the prompt.
   - NEVER fabricate numbers, dates, or source document details.
3. TABULAR DATA:
   - For any comparisons, rankings, or metric analyses, format them nicely with `add_table(headers=..., rows=..., caption=...)`.
4. LISTS & FINDINGS:
   - Use `add_bullet_list(items=...)` to summarize key findings, takeaways, and bibliographic citations.
5. PROCEED AUTONOMOUSLY:
   - Issue multiple tool calls in each turn to efficiently construct the report story from top to bottom.
   - When all sections (Executive Summary, Detailed Findings, Analysis/Tables, Citations) are added, finish without making more tool calls."""


def _format_report_prompt(request: ReportRequest) -> str:
    today_str = date.today().strftime("%B %d, %Y")
    lines = [
        f"USER QUESTION / TOPIC: {request.question}",
        f"REPORT TITLE: {request.style.title}",
        f"SUBTITLE: {request.style.subtitle}",
        f"DATE: {today_str}",
        "",
        "=== SYNTHESIZED ANSWER ===",
        request.answer.text,
        "",
        "=== VERIFIED FINDINGS ===",
    ]

    for finding in request.analysis.findings:
        chunks_str = ", ".join(finding.chunk_ids) if finding.chunk_ids else "general"
        lines.append(f"- {finding.statement} (Citations: [{chunks_str}])")

    if request.analysis.calculations:
        lines.append("\n=== NUMERIC CALCULATIONS ===")
        for c in request.analysis.calculations:
            lines.append(f"- {c.description}: {c.expression} = {c.result}")

    if request.analysis.comparisons:
        lines.append("\n=== DOCUMENT COMPARISONS ===")
        for comp in request.analysis.comparisons:
            lines.append(f"Topic: {comp.topic}")
            lines.append(f"Criteria: {', '.join(comp.criteria)}")
            if comp.summary:
                lines.append(f"Summary: {comp.summary}")
            if comp.similarities:
                lines.append("Similarities: " + "; ".join(comp.similarities))
            if comp.differences:
                lines.append("Differences: " + "; ".join(comp.differences))
            for entry in comp.entries:
                lines.append(f"  • Doc: {entry.doc_name or entry.doc_id}: {entry.attributes}")

    if request.analysis.metric_analyses:
        lines.append("\n=== METRIC & TREND ANALYSES ===")
        for m in request.analysis.metric_analyses:
            lines.append(f"[{m.analysis_type.upper()}] {m.metric}: {m.summary}")
            if m.details:
                lines.append(f"  Details: {m.details}")

    if request.analysis.tables:
        lines.append("\n=== EXTRACTED DATA TABLES ===")
        for t in request.analysis.tables:
            caption = t.caption or f"Table for chunk {t.chunk_id}"
            lines.append(f"Table Caption: {caption}")
            lines.append(f"Headers: {t.headers}")
            lines.append(f"Rows ({len(t.rows)}): {t.rows[:10]}")

    if request.style.include_citations and request.answer.citations:
        lines.append("\n=== SOURCES & CITATIONS ===")
        for cit in request.answer.citations:
            page_text = f", Page {cit.page}" if cit.page else ""
            sec_text = f", Section: {cit.section}" if cit.section else ""
            lines.append(f"- [{cit.chunk_id}] Document: '{cit.doc_name}'{page_text}{sec_text}")

    return "\n".join(lines)


class ReportAgent:
    """Agent that plans and renders professional PDF reports using tool calling."""

    def __init__(self, llm: BaseChatModel):
        if llm is None:
            raise ConfigurationError("ReportAgent requires an initialized LLM.")
        self.llm = llm

    def invoke(self, request: ReportRequest) -> ReportResult:
        return run_report(request, self.llm)


def run_report(
    request: ReportRequest,
    llm: BaseChatModel,
    max_tool_turns: int = MAX_TOOL_TURNS,
) -> ReportResult:
    """Execute the Report Generator tool loop to create a styled PDF report."""
    settings = get_settings()
    report_id = str(uuid4())
    reports_dir = settings.data_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(reports_dir / f"{report_id}.pdf")

    doc_ctx = ReportDocument()
    make_new_document(doc_ctx, output_path, request.style)

    tools = build_report_tools(doc_ctx)
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    messages: list[BaseMessage] = [
        SystemMessage(content=_REPORT_AGENT_SYSTEM_PROMPT),
        HumanMessage(content=_format_report_prompt(request)),
    ]

    for turn in range(max_tool_turns):
        try:
            ai_msg = llm_with_tools.invoke(messages)
            messages.append(ai_msg)
        except Exception as exc:
            logger.warning("Report LLM invoke failed on turn %d: %s", turn, exc)
            break

        tool_calls = getattr(ai_msg, "tool_calls", None) or []
        if not tool_calls:
            logger.info("Report LLM finished generating formatting commands at turn %d.", turn)
            break

        for call in tool_calls:
            name, args, call_id = call["name"], call.get("args", {}), call["id"]
            tool = tool_map.get(name)
            if tool is None:
                messages.append(
                    ToolMessage(
                        tool_call_id=call_id,
                        content=f"Error: Unknown tool '{name}'. Available tools: {list(tool_map.keys())}",
                    )
                )
                continue

            try:
                res = tool.invoke(args)
                messages.append(ToolMessage(tool_call_id=call_id, content=str(res)))
            except Exception as exc:
                logger.warning("Tool call '%s' failed: %s", name, exc)
                messages.append(
                    ToolMessage(
                        tool_call_id=call_id,
                        content=f"Tool error: {exc}. Please adjust arguments and proceed.",
                    )
                )

    # If the LLM failed to invoke any tools (e.g. mock or error), provide deterministic fallback content
    if not doc_ctx.story:
        from app.agents.report.tools.cover import add_cover_page
        from app.agents.report.tools.layout import add_spacer
        from app.agents.report.tools.text import add_bullet_list, add_heading, add_paragraph

        if request.style.include_cover:
            add_cover_page(doc_ctx, request.style.title, request.style.subtitle)
        add_heading(doc_ctx, "Executive Summary", level=1)
        add_paragraph(doc_ctx, request.answer.text)
        add_spacer(doc_ctx, 6.0)

        if request.analysis.findings:
            add_heading(doc_ctx, "Key Verified Findings", level=2)
            add_bullet_list(doc_ctx, [f.statement for f in request.analysis.findings])

        if request.style.include_citations and request.answer.citations:
            add_heading(doc_ctx, "References & Sources", level=2)
            add_bullet_list(doc_ctx, [f"[{c.chunk_id}] {c.doc_name}" for c in request.answer.citations])

    num_pages = save_document(doc_ctx)
    logger.info("Report generated successfully: %s (%d pages)", output_path, num_pages)

    return ReportResult(
        report_id=report_id,
        filename=f"{report_id}.pdf",
        path=output_path,
        num_pages=num_pages,
    )
