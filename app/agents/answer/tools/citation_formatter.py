import logging

from app.core.contract import AnalysisResult, Citation, EvidenceChunk

logger = logging.getLogger(__name__)


def sanitize_analysis(analysis: AnalysisResult, valid_chunk_ids: set[str]) -> AnalysisResult:
    """Return a copy of AnalysisResult with unknown chunk_ids stripped and ungrounded items dropped."""

    def _clean(cids: list[str], label: str) -> list[str]:
        kept = list(dict.fromkeys(cid for cid in cids if cid in valid_chunk_ids))
        dropped = [cid for cid in cids if cid not in valid_chunk_ids]
        if dropped:
            logger.warning("Answer Agent dropped unknown chunk_ids %s from %s", dropped, label)
        return kept

    cleaned_findings = []
    for f in analysis.findings:
        kept_cids = _clean(f.chunk_ids, f"finding '{f.statement[:40]}'")
        if not kept_cids:
            logger.warning("Dropped ungrounded finding with no valid chunk_ids: '%s'", f.statement[:60])
            continue
        cleaned_findings.append(f.model_copy(update={"chunk_ids": kept_cids}))

    cleaned_calcs = []
    for c in analysis.calculations:
        kept_cids = _clean(c.chunk_ids, f"calculation '{c.description[:40]}'")
        if not kept_cids:
            logger.warning("Dropped ungrounded calculation with no valid chunk_ids: '%s'", c.description[:60])
            continue
        cleaned_calcs.append(c.model_copy(update={"chunk_ids": kept_cids}))

    cleaned_metrics = []
    for m in analysis.metric_analyses:
        kept_cids = _clean(m.chunk_ids, f"metric '{m.metric}'")
        if not kept_cids:
            logger.warning("Dropped ungrounded metric analysis with no valid chunk_ids: '%s'", m.metric)
            continue
        cleaned_metrics.append(m.model_copy(update={"chunk_ids": kept_cids}))

    cleaned_comps = []
    for comp in analysis.comparisons:
        cleaned_entries = []
        all_entry_cids: list[str] = []
        for e in comp.entries:
            kept_entry_cids = _clean(e.chunk_ids, f"comparison entry '{e.doc_id}'")
            if not kept_entry_cids:
                logger.warning("Dropped ungrounded comparison entry '%s' in '%s'", e.doc_id, comp.topic)
                continue
            cleaned_entries.append(e.model_copy(update={"chunk_ids": kept_entry_cids}))
            all_entry_cids.extend(kept_entry_cids)

        kept_comp_cids = _clean(comp.chunk_ids, f"comparison '{comp.topic}'")
        merged_comp_cids = list(dict.fromkeys(kept_comp_cids + all_entry_cids))
        if not cleaned_entries and not merged_comp_cids:
            logger.warning("Dropped ungrounded comparison with no valid entries or chunk_ids: '%s'", comp.topic)
            continue
        cleaned_comps.append(
            comp.model_copy(
                update={
                    "entries": cleaned_entries,
                    "chunk_ids": merged_comp_cids,
                }
            )
        )

    return analysis.model_copy(
        update={
            "findings": cleaned_findings,
            "calculations": cleaned_calcs,
            "metric_analyses": cleaned_metrics,
            "comparisons": cleaned_comps,
        }
    )


def build_citations(chunk_ids: list[str], chunks: list[EvidenceChunk]) -> list[Citation]:
    """Build deduplicated Citation objects strictly from chunks present in the allowed pool."""
    chunk_lookup = {c.chunk_id: c for c in chunks}
    citations: list[Citation] = []
    seen: set[str] = set()

    for cid in chunk_ids:
        if cid in seen or cid not in chunk_lookup:
            continue
        seen.add(cid)
        chunk = chunk_lookup[cid]
        citations.append(
            Citation(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                doc_name=chunk.doc_name,
                page=chunk.page,
                section=chunk.section,
            )
        )
    return citations