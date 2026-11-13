import json
import logging
from typing import Any, Callable, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.agents.analyst.tools.calculator import calculate
from app.agents.analyst.tools.data_analysis import (
    DataAnalysisError,
    analyze_correlation,
    analyze_numbers,
    analyze_table_correlation,
    analyze_table_metric,
    analyze_table_trend,
    analyze_trend,
    find_table_with_columns,
)
from app.agents.analyst.tools.document_comparison import (
    bind_compare_documents,
    group_chunks_by_document,
    missing_to_info,
)
from app.agents.analyst.tools.table_extractor import extract_from_chunks
from app.core.contract import (
    AnalysisResult,
    Calculation,
    Comparison,
    EvidenceBundle,
    EvidenceChunk,
    ExtractedTable,
    Finding,
    MetricAnalysis,
)

logger = logging.getLogger(__name__)


MAX_TOOL_TURNS = 6

_ANALYST_SYSTEM_PROMPT = """You are the Analyst Agent in a multi-agent document analysis system.
Examine the retrieved evidence and answer the user's question using ONLY verified facts and the provided Python tools.

RULES:
1. NEVER do mental math, averages, percentages, rankings, trends, or correlations yourself. Always call a tool:
   - `calculate`: arithmetic expressions.
   - `analyze_table_metric` / `analyze_numbers`: rankings, averages, spread, best/worst.
   - `analyze_table_trend` / `analyze_trend`: changes over ordered periods (years, epochs, versions).
   - `analyze_table_correlation` / `analyze_correlation`: relationships between two numeric variables.
   - `compare_documents`: comparing 2 or more source documents across shared criteria.
2. NEVER invent `chunk_id` or `doc_id` values. Cite ONLY IDs present in the evidence.
3. If required facts, documents, or metrics are missing, list specific retrieval queries in `missing_info`. Do NOT re-request queries listed under ALREADY SEARCHED GAPS.
4. If a tool returns an error, fix the arguments and call it again."""


class _AnalystSynthesis(BaseModel): # The llm's output which consists of findings and missing info 

    model_config = ConfigDict(extra="forbid")

    findings: list[Finding] = Field(default_factory=list)
    missing_info: list[str] = Field(default_factory=list)


def _dedupe(items: list[str]) -> list[str]: # Remove duplicates while preserving order and stripping whitespace
    return list(dict.fromkeys(s.strip() for s in items if s.strip()))


def _to_json(output: Any) -> str: # Turns tool outputs into JSON bec the tool message content must be a string
    return json.dumps(output, default=lambda o: o.model_dump() if isinstance(o, BaseModel) else str(o))


def _format_evidence_prompt(  # Outputs a formatted prompt with the user question and retrieved evidence, including pre-extracted tables and their metadata
    bundle: EvidenceBundle,
    tables: list[ExtractedTable],
    attempted_queries: list[str],
) -> str:
    grouped = group_chunks_by_document(bundle.chunks)
    lines = [f"USER QUESTION: {bundle.query}", ""]

    if attempted_queries:
        lines.append("ALREADY SEARCHED GAPS (do NOT request these again):")
        lines.extend(f"- {q}" for q in attempted_queries)
        lines.append("")

    lines.append("RETRIEVED EVIDENCE BY DOCUMENT:")
    for doc_id, doc_chunks in grouped.items():
        lines.append(f"\n=== Document: {doc_chunks[0].doc_name} (doc_id: {doc_id}) ===")
        for c in doc_chunks:
            meta = f"type={c.chunk_type}"
            if c.page is not None:
                meta += f", page {c.page}"
            if c.section:
                meta += f", section '{c.section}'"
            # Avoid duplicating raw table markdown when pre-extracted JSON is shown below
            body = "[Table pre-extracted under PRE-EXTRACTED TABLES below]" if c.chunk_type == "table" else c.text
            lines.append(f"[{c.chunk_id} | {meta}]\n{body}")

    if tables:
        lines.append("\nPRE-EXTRACTED TABLES (reference by chunk_id in analyze_table_* tools):")
        for idx, t in enumerate(tables):
            lines.append(f"[table_index={idx}] {t.model_dump_json()}")

    return "\n".join(lines)


def _build_tools( # Turns functions into tools to be called by the llm
    get_chunks: Callable[[], list[EvidenceChunk]],
    tables: list[ExtractedTable],
) -> list[StructuredTool]:

    def _resolve_table(chunk_id: str, *required_columns: str) -> ExtractedTable:
        matching = [t for t in tables if t.chunk_id == chunk_id.strip()]
        if not matching:
            available = sorted({t.chunk_id for t in tables})
            raise DataAnalysisError(
                f"No extracted table found for chunk_id '{chunk_id}'. Available table chunks: {available}."
            )
        if len(matching) == 1:
            return matching[0]
        return find_table_with_columns(matching, *required_columns)

    def _table_metric(
        chunk_id: str,
        metric_column: str,
        label_column: Optional[str] = None,
        higher_is_better: bool = True,
    ) -> MetricAnalysis:
        """Rank the rows of a pre-extracted table by a numeric column and compute summary statistics.

        Args:
            chunk_id: ID of the table chunk (e.g. "c1").
            metric_column: Exact header of the numeric column to analyze, e.g. "Top-1 Accuracy".
            label_column: Optional header naming each row. Defaults to the first column.
            higher_is_better: True for accuracy/revenue, False for error rate/loss/latency.
        """
        cols = [metric_column] + ([label_column] if label_column else [])
        table = _resolve_table(chunk_id, *cols)
        return analyze_table_metric(table, metric_column, label_column, higher_is_better)

    def _table_trend(
        chunk_id: str,
        period_column: str,
        metric_column: str,
        higher_is_better: bool = True,
    ) -> MetricAnalysis:
        """Analyze how a table column changes across ordered periods (e.g. Year vs Top-1 Accuracy).

        Args:
            chunk_id: ID of the table chunk (e.g. "c4").
            period_column: Header of the period/time column, e.g. "Year".
            metric_column: Header of the numeric metric column, e.g. "Top-1 Accuracy".
            higher_is_better: True when rising values are favorable, False when falling is better.
        """
        table = _resolve_table(chunk_id, period_column, metric_column)
        return analyze_table_trend(table, period_column, metric_column, higher_is_better)

    def _table_correlation(
        chunk_id: str,
        x_column: str,
        y_column: str,
    ) -> MetricAnalysis:
        """Measure Pearson correlation between two numeric columns in a pre-extracted table.

        Args:
            chunk_id: ID of the table chunk (e.g. "c1").
            x_column: Header of the first numeric column, e.g. "Params (M)".
            y_column: Header of the second numeric column, e.g. "Top-1 Accuracy".
        """
        table = _resolve_table(chunk_id, x_column, y_column)
        return analyze_table_correlation(table, x_column, y_column)

    _table_metric.__name__ = "analyze_table_metric"
    _table_trend.__name__ = "analyze_table_trend"
    _table_correlation.__name__ = "analyze_table_correlation"

    funcs = [
        calculate,
        _table_metric,
        analyze_numbers,
        _table_trend,
        analyze_trend,
        _table_correlation,
        analyze_correlation,
        bind_compare_documents(get_chunks),
    ]
    return [StructuredTool.from_function(func=fn) for fn in funcs]


def run_analyst(
    bundle: EvidenceBundle,
    llm: BaseChatModel,
    attempted_queries: Optional[list[str]] = None,
    max_tool_turns: int = MAX_TOOL_TURNS,
) -> AnalysisResult:

    prior_gaps = _dedupe(attempted_queries or [])
    prior_gaps_lower = {g.lower() for g in prior_gaps}

    if not bundle.chunks:
        fallback = [bundle.query.strip()] if bundle.query.strip() and bundle.query.strip().lower() not in prior_gaps_lower else []
        return AnalysisResult(question=bundle.query, missing_info=fallback)

    valid_chunk_ids = {c.chunk_id for c in bundle.chunks}

    # Pre-extract all table chunks 
    tables: list[ExtractedTable] = extract_from_chunks(bundle.chunks)
    calculations: list[Calculation] = []
    metric_analyses: list[MetricAnalysis] = []
    comparisons: list[Comparison] = []

    # Bind tools using live chunk getter + table lookup by chunk_id
    tools = _build_tools(lambda: bundle.chunks, tables)
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    messages: list[BaseMessage] = [
        SystemMessage(content=_ANALYST_SYSTEM_PROMPT),
        HumanMessage(content=_format_evidence_prompt(bundle, tables, prior_gaps)),
    ]

    # Tool-calling loop
    for _ in range(max_tool_turns):
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        tool_calls = getattr(ai_msg, "tool_calls", None) or []
        if not tool_calls:
            break

        for call in tool_calls:
            name, args, call_id = call["name"], call.get("args", {}), call["id"]
            tool = tool_map.get(name)
            if tool is None:
                messages.append(
                    ToolMessage(
                        tool_call_id=call_id,
                        content=f"Tool error: Unknown tool '{name}'. Available: {list(tool_map)}.",
                    )
                )
                continue

            try:
                result = tool.invoke(args)
                if isinstance(result, Calculation):
                    result.chunk_ids = _dedupe([cid for cid in result.chunk_ids if cid in valid_chunk_ids])
                    calculations.append(result)
                elif isinstance(result, MetricAnalysis):
                    result.chunk_ids = _dedupe([cid for cid in result.chunk_ids if cid in valid_chunk_ids])
                    metric_analyses.append(result)
                elif isinstance(result, Comparison):
                    comparisons.append(result)

                messages.append(ToolMessage(tool_call_id=call_id, content=_to_json(result))) # We take all the tool outputs and append them to the messages for the llm to see in the next turn and decide its next step
            except (ValueError, ValidationError, KeyError, TypeError) as exc:
                messages.append(
                    ToolMessage(
                        tool_call_id=call_id,
                        content=f"Tool error: {exc}. Fix the arguments and retry.",
                    )
                )

    # Final structured synthesis (guarded against malformed LLM output)
    synthesizer = llm.with_structured_output(_AnalystSynthesis) # This function makes the llm output data in the given format to be compatible with the required output of the analyst agent
    messages.append(
        HumanMessage(
            content=(
                "Synthesize your final findings with exact chunk_ids and list any remaining "
                "missing_info needed to answer the question completely."
            )
        )
    )
    try:
        raw_synthesis = synthesizer.invoke(messages)
        synthesis = raw_synthesis if isinstance(raw_synthesis, _AnalystSynthesis) else _AnalystSynthesis()
    except Exception as exc:
        logger.warning("Analyst synthesis failed; falling back to tool outputs only: %s", exc)
        synthesis = _AnalystSynthesis()

    # Remove any findings that reference chunk_ids not present in the retrieved evidence
    grounded_findings: list[Finding] = []
    for f in synthesis.findings:
        verified_cids = _dedupe([cid for cid in f.chunk_ids if cid in valid_chunk_ids])
        if len(verified_cids) < len(f.chunk_ids):
            dropped = [cid for cid in f.chunk_ids if cid not in valid_chunk_ids]
            logger.warning("Dropped ungrounded chunk_ids %s from finding: %s", dropped, f.statement)
        if verified_cids:
            grounded_findings.append(Finding(statement=f.statement.strip(), chunk_ids=verified_cids))
        else:
            logger.warning("Dropped finding with no valid citations: %s", f.statement)

    # Merge comparison gaps + LLM gaps and filter out already-attempted queries
    comparison_gaps = [gap for comp in comparisons for gap in missing_to_info(comp)]
    all_missing = [
        gap
        for gap in _dedupe(comparison_gaps + synthesis.missing_info)
        if gap.lower() not in prior_gaps_lower
    ]

    return AnalysisResult(
        question=bundle.query,
        findings=grounded_findings,
        calculations=calculations,
        tables=tables,
        comparisons=comparisons,
        metric_analyses=metric_analyses,
        missing_info=all_missing,
    )