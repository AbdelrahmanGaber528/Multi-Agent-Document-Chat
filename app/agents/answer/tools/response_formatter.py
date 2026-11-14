import json
from typing import Any

from app.core.contract import AnalysisResult, Comparison


def has_verified_content(analysis: AnalysisResult) -> bool:
    """Only substantive analytical outputs count—never raw pre-extracted tables alone."""
    return bool(
        analysis.findings
        or analysis.calculations
        or analysis.comparisons
        or analysis.metric_analyses
    )


def extract_message_text(content: Any) -> str:
    """Safely extract text whether LangChain returns a str or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content) if content is not None else ""


def _escape_md_cell(value: Any) -> str:
    """Escape pipe characters and newlines so Markdown table structure never breaks."""
    return str(value).replace("|", "\\|").replace("\r\n", " ").replace("\n", " ").strip()


def format_comparison_markdown(comp: Comparison) -> str:
    """Render a Comparison object into a clean, cited Markdown table."""
    if not comp.entries:
        return comp.summary or f"**{comp.topic}**"

    headers = ["Document", *comp.criteria, "Sources"]
    sep = ["---"] * len(headers)
    rows = [
        "| " + " | ".join(_escape_md_cell(h) for h in headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for entry in comp.entries:
        doc_label = _escape_md_cell(entry.doc_name or entry.doc_id)
        vals = [_escape_md_cell(entry.attributes.get(c) or "Not disclosed") for c in comp.criteria]
        markers = "".join(f"[{cid}]" for cid in entry.chunk_ids)
        rows.append("| " + " | ".join([doc_label, *vals, markers]) + " |")
    return "\n".join(rows)


def build_answer_prompt(analysis: AnalysisResult) -> str:
    """Build the UserMessage prompt for the Answer LLM from verified analysis results only."""
    used_ids = set(analysis.used_chunk_ids)
    lines = [f"USER QUESTION: {analysis.question}", ""]

    if used_ids:
        lines.append(f"VALID CHUNK IDS FOR INLINE CITATIONS: {', '.join(analysis.used_chunk_ids)}")
        lines.append("")

    if analysis.findings:
        lines.append("VERIFIED FINDINGS:")
        for f in analysis.findings:
            markers = "".join(f"[{cid}]" for cid in f.chunk_ids)
            lines.append(f"- {f.statement} {markers}".strip())
        lines.append("")

    if analysis.calculations:
        lines.append("VERIFIED CALCULATIONS:")
        for calc in analysis.calculations:
            markers = "".join(f"[{cid}]" for cid in calc.chunk_ids)
            lines.append(
                f"- {calc.description}: `{calc.expression}` = **{calc.result}** {markers}".strip()
            )
        lines.append("")

    if analysis.metric_analyses:
        lines.append("METRIC ANALYSES (RANKINGS / TRENDS / CORRELATIONS):")
        for ma in analysis.metric_analyses:
            markers = "".join(f"[{cid}]" for cid in ma.chunk_ids)
            lines.append(f"- [{ma.analysis_type.upper()} | {ma.metric}] {ma.summary} {markers}".strip())
            if ma.details:
                lines.append(f"  Details: {json.dumps(ma.details, default=str)}")
        lines.append("")

    if analysis.comparisons:
        lines.append("DOCUMENT COMPARISONS:")
        for comp in analysis.comparisons:
            lines.append(f"Topic: {comp.topic} | Criteria: {', '.join(comp.criteria)}")
            for entry in comp.entries:
                markers = "".join(f"[{cid}]" for cid in entry.chunk_ids)
                lines.append(
                    f"  - {entry.doc_name or entry.doc_id} {markers}: "
                    f"{json.dumps(entry.attributes, default=str)}"
                )
            if comp.similarities:
                lines.append(f"  Similarities: {'; '.join(comp.similarities)}")
            if comp.differences:
                lines.append(f"  Differences: {'; '.join(comp.differences)}")
            if comp.summary:
                lines.append(f"  Summary: {comp.summary}")
        lines.append("")

    referenced_tables = [t for t in analysis.tables if t.chunk_id in used_ids]
    if referenced_tables:
        lines.append("REFERENCED TABLES:")
        for tbl in referenced_tables:
            caption = f" ({tbl.caption})" if tbl.caption else ""
            lines.append(f"Table [{tbl.chunk_id}]{caption}:")
            lines.append(f"  Headers: {json.dumps(tbl.headers, default=str)}")
            for row in tbl.rows:
                lines.append(f"  Row: {json.dumps(row, default=str)}")
        lines.append("")

    return "\n".join(lines).strip()


def build_fallback_answer(analysis: AnalysisResult) -> str:
    """Deterministic non-empty Markdown answer if the LLM call fails or returns empty text."""
    if not has_verified_content(analysis):
        return "I could not find sufficient evidence in the uploaded documents to answer this question."

    parts: list[str] = []
    if analysis.findings:
        parts.append(
            "\n".join(
                f"- {f.statement} {''.join(f'[{cid}]' for cid in f.chunk_ids)}".strip()
                for f in analysis.findings
            )
        )
    if analysis.comparisons:
        parts.append("\n\n".join(format_comparison_markdown(comp) for comp in analysis.comparisons))
    if analysis.calculations:
        parts.append(
            "\n".join(
                f"- **{c.description}:** {c.result} (`{c.expression}`) {''.join(f'[{cid}]' for cid in c.chunk_ids)}".strip()
                for c in analysis.calculations
            )
        )
    if analysis.metric_analyses:
        parts.append(
            "\n".join(
                f"- {m.summary} {''.join(f'[{cid}]' for cid in m.chunk_ids)}".strip()
                for m in analysis.metric_analyses
            )
        )

    return "\n\n".join(p for p in parts if p) or "Analysis complete; see cited sources."


def append_missing_info_section(
    text: str,
    insufficient_evidence: bool,
    missing_info: list[str],
) -> str:
    """Deterministically append the missing-information disclosure when gaps remain."""
    if not insufficient_evidence or not missing_info:
        return text
    if "### information not found" in text.lower():
        return text
    bullets = "\n".join(f"- {item}" for item in missing_info)
    return (
        f"{text.rstrip()}\n\n"
        "### Information Not Found\n"
        f"The following details could not be found in the uploaded documents:\n{bullets}"
    )