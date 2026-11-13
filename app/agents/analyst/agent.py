import json
from typing import Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.agents.analyst.tools.calculator import calculate
from app.agents.analyst.tools.data_analysis import (
    analyze_correlation,
    analyze_numbers,
    analyze_table_correlation,
    analyze_table_metric,
    analyze_table_trend,
    analyze_trend,
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
3. If required facts, documents, or metrics are missing, list specific retrieval queries in `missing_info`.
4. If a tool returns an error, fix the arguments and call it again."""


class _AnalystSynthesis(BaseModel):# The llm's output which consists of findings and missing info 

    model_config = ConfigDict(extra="forbid")
    findings: list[Finding] = Field(default_factory=list)
    missing_info: list[str] = Field(default_factory=list)


def _dedupe(items: list[str]) -> list[str]: # Remove duplicates while preserving order and stripping whitespace
    return list(dict.fromkeys(s.strip() for s in items if s.strip()))


def _to_json(output: Any) -> str: # Turns tool outputs into JSON bec the tool message content must be a string
    return json.dumps(output, default=lambda o: o.model_dump() if isinstance(o, BaseModel) else str(o))


def _format_evidence_prompt(bundle: EvidenceBundle, tables: list[ExtractedTable]) -> str: # Outputs a formatted prompt with the user question and retrieved evidence, including pre-extracted tables and their metadata
    grouped = group_chunks_by_document(bundle.chunks)
    lines = [f"USER QUESTION: {bundle.query}", "", "RETRIEVED EVIDENCE BY DOCUMENT:"]

    for doc_id, doc_chunks in grouped.items():
        lines.append(f"\n=== Document: {doc_chunks[0].doc_name} (doc_id: {doc_id}) ===")
        for c in doc_chunks:
            meta = f"type={c.chunk_type}"
            if c.page is not None:
                meta += f", page {c.page}"
            if c.section:
                meta += f", section '{c.section}'"
            lines.append(f"[{c.chunk_id} | {meta}]\n{c.text}")

    if tables:
        lines.append("\nPRE-EXTRACTED TABLES (pass directly to analyze_table_* tools):")
        lines.extend(t.model_dump_json() for t in tables)

    return "\n".join(lines)


def _build_tools(get_chunks: Callable[[], list[EvidenceChunk]]) -> list[StructuredTool]: #Turns the functions into tools
    funcs = [
        calculate,
        analyze_table_metric,
        analyze_numbers,
        analyze_table_trend,
        analyze_trend,
        analyze_table_correlation,
        analyze_correlation,
        bind_compare_documents(get_chunks),
    ]
    return [StructuredTool.from_function(func=fn) for fn in funcs]


def run_analyst(
    bundle: EvidenceBundle,
    llm: BaseChatModel,
    max_tool_turns: int = MAX_TOOL_TURNS,
) -> AnalysisResult:
    """Run the Analyst Agent on an EvidenceBundle and return a validated AnalysisResult."""
    if not bundle.chunks:
        return AnalysisResult(
            question=bundle.query,
            missing_info=[bundle.query.strip()] if bundle.query.strip() else ["No evidence retrieved."],
        )

    valid_chunk_ids = {c.chunk_id for c in bundle.chunks}

    # 1. Pre-extract all table chunks deterministically
    tables: list[ExtractedTable] = extract_from_chunks(bundle.chunks)
    calculations: list[Calculation] = []
    metric_analyses: list[MetricAnalysis] = []
    comparisons: list[Comparison] = []

    # 2. Bind tools using a live chunk getter so multi-round bundles stay current
    tools = _build_tools(lambda: bundle.chunks)
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    messages: list[BaseMessage] = [
        SystemMessage(content=_ANALYST_SYSTEM_PROMPT),
        HumanMessage(content=_format_evidence_prompt(bundle, tables)),
    ]

    # 3. Tool-calling loop
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
                    calculations.append(result)
                elif isinstance(result, MetricAnalysis):
                    metric_analyses.append(result)
                elif isinstance(result, Comparison):
                    comparisons.append(result)

                messages.append(ToolMessage(tool_call_id=call_id, content=_to_json(result)))
            except (ValueError, ValidationError) as exc:
                messages.append(
                    ToolMessage(
                        tool_call_id=call_id,
                        content=f"Tool error: {exc}. Fix the arguments and retry.",
                    )
                )

    # 4. Final structured synthesis for qualitative Findings + missing_info
    synthesizer = llm.with_structured_output(_AnalystSynthesis)
    messages.append(
        HumanMessage(
            content=(
                "Synthesize your final findings with exact chunk_ids and list any remaining "
                "missing_info needed to answer the question completely."
            )
        )
    )
    synthesis: _AnalystSynthesis = synthesizer.invoke(messages)

    # 5. Strip ungrounded chunk_ids from LLM-authored findings
    grounded_findings: list[Finding] = []
    for f in synthesis.findings:
        verified_cids = _dedupe([cid for cid in f.chunk_ids if cid in valid_chunk_ids])
        if verified_cids:
            grounded_findings.append(Finding(statement=f.statement.strip(), chunk_ids=verified_cids))

    # 6. Combine comparison gaps + LLM-identified gaps for the feedback loop
    comparison_gaps = [gap for comp in comparisons for gap in missing_to_info(comp)]
    all_missing = _dedupe(comparison_gaps + synthesis.missing_info)

    return AnalysisResult(
        question=bundle.query,
        findings=grounded_findings,
        calculations=calculations,
        tables=tables,
        comparisons=comparisons,
        metric_analyses=metric_analyses,
        missing_info=all_missing,
    )