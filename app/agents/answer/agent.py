import logging
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.answer.tools.citation_formatter import build_citations, sanitize_analysis
from app.agents.answer.tools.response_formatter import (
    append_missing_info_section,
    build_answer_prompt,
    build_fallback_answer,
    extract_message_text,
    has_verified_content,
)
from app.agents.answer.tools.source_formatter import resolve_inline_citations
from app.core.contract import AnalysisResult, EvidenceBundle, FinalAnswer

logger = logging.getLogger(__name__)

_ANSWER_SYSTEM_PROMPT = """You are the Answer Agent in a multi-agent document analysis system.
Write a clear, accurate, well-structured final response to the user's question using ONLY the verified results provided.

RULES:
1. Answer in English.
2. Ground every claim strictly in the provided findings, calculations, comparisons, and metric analyses. Do NOT recalculate numbers or add outside knowledge.
3. Format multi-item comparisons, rankings, and tabular data as clean Markdown tables.
4. Cite sources inline using ONLY exact single-ID brackets from VALID CHUNK IDS—one ID per bracket (e.g., [c1] or [c1][c3], never [c1, c3]). NEVER write document names or page numbers yourself—they are formatted automatically.
5. Do NOT write an "Information Not Found" or missing-data section; that disclosure is appended automatically."""


def run_answer(
    analysis: AnalysisResult,
    bundle: EvidenceBundle,
    llm: BaseChatModel,
    rounds_used: int = 1,
    missing_info_override: Optional[list[str]] = None,
    max_rounds_reached: bool = True,
) -> FinalAnswer:
    """Synthesize a verified AnalysisResult and EvidenceBundle into a cited FinalAnswer."""
    valid_chunk_ids = {c.chunk_id for c in bundle.chunks}
    clean_analysis = sanitize_analysis(analysis, valid_chunk_ids)

    raw_missing = missing_info_override if missing_info_override is not None else clean_analysis.missing_info
    effective_missing = list(dict.fromkeys(s.strip() for s in raw_missing if s.strip()))

    verified = has_verified_content(clean_analysis)
    insufficient_evidence = not verified or (max_rounds_reached and bool(effective_missing))

    if not verified:
        text = append_missing_info_section(
            build_fallback_answer(clean_analysis),
            insufficient_evidence=True,
            missing_info=effective_missing,
        )
        return FinalAnswer(
            text=text,
            citations=[],
            insufficient_evidence=True,
            missing_info=effective_missing,
            rounds_used=max(1, rounds_used),
        )

    prompt = build_answer_prompt(clean_analysis)
    messages = [
        SystemMessage(content=_ANSWER_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    try:
        response = llm.invoke(messages)
        raw_text = extract_message_text(response.content).strip()
        if not raw_text:
            logger.warning("Answer LLM returned empty content; using deterministic fallback.")
            raw_text = build_fallback_answer(clean_analysis)
    except Exception:
        logger.warning("Answer LLM invocation failed; using deterministic fallback.", exc_info=True)
        raw_text = build_fallback_answer(clean_analysis)

    # Restrict inline citations strictly to chunks the Analyst actually relied on
    allowed_ids = set(clean_analysis.used_chunk_ids)
    cited_pool = [c for c in bundle.chunks if c.chunk_id in allowed_ids]

    resolved_text, inline_cids = resolve_inline_citations(
        raw_text,
        allowed_chunks=cited_pool,
        bundle_chunk_ids=valid_chunk_ids,
    )
    final_text = append_missing_info_section(
        resolved_text,
        insufficient_evidence,
        effective_missing,
    )

    all_cited_ids = inline_cids or clean_analysis.used_chunk_ids
    citations = build_citations(all_cited_ids, cited_pool)

    return FinalAnswer(
        text=final_text,
        citations=citations,
        insufficient_evidence=insufficient_evidence,
        missing_info=effective_missing,
        rounds_used=max(1, rounds_used),
    )