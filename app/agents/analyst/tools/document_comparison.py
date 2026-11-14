from collections import defaultdict
import re
from typing import Any, Callable, Optional

from app.agents.analyst.tools.table_extractor import parse_number
from app.core.contract import Comparison, DocumentEntry, EvidenceChunk, MissingTarget
from app.core.errors import DocumentComparisonError

_MISSING_RE = re.compile(
    r"^(n/?a|none|null|unknown"
    r"|not (specified|reported|provided|available|applicable|mentioned|stated|disclosed)"
    r"|no (data|information))"
    r"(?:\s*\([^)]*\))?"
    r"(?:\s+(in|by|for|from|on)\b.*)?$"
)


def group_chunks_by_document(chunks: list[EvidenceChunk]) -> dict[str, list[EvidenceChunk]]:
    """Group retrieved chunks by doc_id while preserving retrieval order."""
    grouped: dict[str, list[EvidenceChunk]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.doc_id].append(chunk)
    return dict(grouped)


def list_document_sources(chunks: list[EvidenceChunk]) -> list[dict[str, Any]]:
    """Return a deduplicated summary of source documents keyed by doc_id."""
    grouped = group_chunks_by_document(chunks)
    return [
        {
            "doc_id": doc_id,
            "doc_name": doc_chunks[0].doc_name,
            "chunk_ids": [c.chunk_id for c in doc_chunks],
            "chunk_count": len(doc_chunks),
        }
        for doc_id, doc_chunks in grouped.items()
    ]


def missing_targets(comparison: Comparison) -> list[MissingTarget]:
    """Return typed gap targets with doc_id (for metadata filtering) and contextual query."""
    if not comparison.missing_by_doc:
        return []

    name_by_id = {entry.doc_id: entry.doc_name for entry in comparison.entries}
    name_counts: dict[str, int] = defaultdict(int)
    for name in name_by_id.values():
        name_counts[name] += 1

    targets: list[MissingTarget] = []
    for doc_id, criteria in comparison.missing_by_doc.items():
        doc_name = name_by_id.get(doc_id, doc_id)
        label = f"{doc_name} ({doc_id})" if name_counts[doc_name] > 1 else doc_name
        for criterion in criteria:
            targets.append(
                MissingTarget(
                    doc_id=doc_id,
                    doc_name=doc_name,
                    criterion=criterion,
                    query=f"{criterion} of {label} ({comparison.topic})",
                )
            )
    return targets


def missing_to_info(comparison: Comparison) -> list[str]:

    return [target.query for target in missing_targets(comparison)]


def _dedupe(items: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for item in items:
        cleaned = item.strip()
        if cleaned:
            seen.setdefault(cleaned, None)
    return list(seen)


def _dedupe_ci(items: list[str]) -> list[str]:
    """Deduplicate strings case-insensitively while preserving first-seen casing."""
    seen: dict[str, str] = {}
    for item in items:
        cleaned = item.strip()
        if cleaned:
            seen.setdefault(cleaned.lower(), cleaned)
    return list(seen.values())


def _is_missing(value: Optional[str]) -> bool:
    if value is None:
        return True
    cleaned = value.strip().lower().rstrip(".")
    return cleaned == "" or cleaned in {"-", "\u2013", "\u2014"} or bool(_MISSING_RE.match(cleaned))


def _comparable_number(raw: str) -> Optional[tuple[float, bool, str]]:
    """Parse a number along with its unit decoration (% and currency), or None if unsafe."""
    s = raw.strip()
    if "\u00b1" in s or "+/-" in s:
        return None
    val = parse_number(s)
    if val is None:
        return None
    has_pct = s.rstrip("*\u2020\u2021 ").endswith("%")
    no_sign = s.lstrip("+-(").strip()
    currency = no_sign[0] if no_sign and no_sign[0] in "$\u20ac\u00a3\u00a5" else ""
    return (val, has_pct, currency)


def compare_documents(
    topic: str,
    criteria: list[str],
    documents: list[DocumentEntry],
    chunks: list[EvidenceChunk],
    similarities: Optional[list[str]] = None,
    differences: Optional[list[str]] = None,
    summary: Optional[str] = None,
) -> Comparison:

    if not topic or not topic.strip():
        raise DocumentComparisonError("topic cannot be empty.")

    clean_criteria = _dedupe_ci(criteria)
    if not clean_criteria:
        raise DocumentComparisonError("criteria must contain at least one comparison dimension.")

    if not chunks:
        raise DocumentComparisonError("No evidence chunks provided to verify document citations.")

    chunk_by_id = {c.chunk_id: c for c in chunks}
    doc_name_by_id: dict[str, str] = {}
    for c in chunks:
        doc_name_by_id.setdefault(c.doc_id, c.doc_name)

    if len(doc_name_by_id) < 2:
        only_id, only_name = next(iter(doc_name_by_id.items()))
        raise DocumentComparisonError(
            f"Retrieved evidence only covers 1 document ('{only_name}', doc_id='{only_id}'). "
            "Cannot compare documents yet—record the missing source in missing_info instead."
        )

    if len(documents) < 2:
        raise DocumentComparisonError(
            "Comparing documents requires at least 2 source entries. "
            "If evidence only covers one document, record the missing source in missing_info."
        )

    expected_by_lower = {crit.lower(): crit for crit in clean_criteria}
    entries: list[DocumentEntry] = []
    seen_doc_ids: set[str] = set()
    all_chunk_ids: list[str] = []
    missing_by_doc: dict[str, list[str]] = {}

    for entry in documents:
        doc_id = entry.doc_id.strip()
        if doc_id not in doc_name_by_id:
            raise DocumentComparisonError(
                f"Unknown doc_id '{doc_id}'. Available doc_ids: {sorted(doc_name_by_id)}."
            )
        if doc_id in seen_doc_ids:
            raise DocumentComparisonError(
                f"Duplicate doc_id '{doc_id}'. Merge evidence for the same document into one entry."
            )
        seen_doc_ids.add(doc_id)

        canonical_name = doc_name_by_id[doc_id]
        chunk_ids = _dedupe(entry.chunk_ids)
        if not chunk_ids:
            raise DocumentComparisonError(
                f"Document '{canonical_name}' ({doc_id}) must cite at least one chunk_id."
            )

        for cid in chunk_ids:
            chunk = chunk_by_id.get(cid)
            if chunk is None:
                raise DocumentComparisonError(
                    f"Unknown chunk_id '{cid}' for '{canonical_name}' ({doc_id}). Never invent citations."
                )
            if chunk.doc_id != doc_id:
                raise DocumentComparisonError(
                    f"Chunk '{cid}' belongs to '{chunk.doc_id}', not '{doc_id}'."
                )

        all_chunk_ids.extend(chunk_ids)

        # Require exact 1-to-1 key parity so neither typos nor forgotten keys become false N/As
        attr_lookup = {k.strip().lower(): v for k, v in entry.attributes.items()}
        missing_keys = [crit for crit in clean_criteria if crit.lower() not in attr_lookup]
        extra_keys = [k for k in entry.attributes if k.strip().lower() not in expected_by_lower]
        if missing_keys or extra_keys:
            problems = []
            if missing_keys:
                problems.append(f"missing keys {missing_keys} (write 'N/A' if unreported)")
            if extra_keys:
                problems.append(f"unexpected keys {extra_keys}")
            raise DocumentComparisonError(
                f"Attribute keys for '{canonical_name}' ({'; '.join(problems)}) "
                f"must match criteria {clean_criteria}."
            )

        aligned_attrs: dict[str, str] = {}
        doc_missing: list[str] = []
        for crit in clean_criteria:
            raw_val = attr_lookup[crit.lower()]
            if _is_missing(raw_val):
                aligned_attrs[crit] = "N/A"
                doc_missing.append(crit)
            else:
                aligned_attrs[crit] = raw_val.strip()

        if doc_missing:
            missing_by_doc[doc_id] = doc_missing

        entries.append(
            DocumentEntry(
                doc_id=doc_id,
                doc_name=canonical_name,
                attributes=aligned_attrs,
                chunk_ids=chunk_ids,
            )
        )

    detected_similarities = _dedupe(list(similarities or []))
    detected_differences = _dedupe(list(differences or []))

    for crit in clean_criteria:
        present = [
            (e.doc_name, e.attributes[crit])
            for e in entries
            if not _is_missing(e.attributes[crit])
        ]
        if len(present) < 2:
            continue

        parsed = [_comparable_number(val) for _, val in present]
        same_unit_numbers = (
            all(p is not None for p in parsed)
            and len({(p[1], p[2]) for p in parsed if p is not None}) == 1
        )

        if same_unit_numbers:
            nums = [p[0] for p in parsed if p is not None]
            if len(set(nums)) == 1:
                note = f"All sources that report {crit} agree: {present[0][1]}."
                if note not in detected_similarities:
                    detected_similarities.append(note)
            else:
                breakdown = "; ".join(f"{name}: {val}" for name, val in present)
                note = f"{crit} differs across sources ({breakdown})."
                if note not in detected_differences:
                    detected_differences.append(note)
        elif len({val.lower() for _, val in present}) == 1:
            note = f"All sources that report {crit} agree: {present[0][1]}."
            if note not in detected_similarities:
                detected_similarities.append(note)

    doc_labels = [e.doc_name for e in entries]
    final_summary = (summary or "").strip()
    if not final_summary:
        final_summary = (
            f"Compared {len(entries)} documents ({', '.join(doc_labels)}) "
            f"across {len(clean_criteria)} criteria: {', '.join(clean_criteria)}."
        )
    if missing_by_doc:
        name_map = {e.doc_id: e.doc_name for e in entries}
        gaps = "; ".join(
            f"{name_map[did]} missing {', '.join(cols)}" for did, cols in missing_by_doc.items()
        )
        final_summary += f" Incomplete fields: {gaps}."

    return Comparison(
        topic=topic.strip(),
        criteria=clean_criteria,
        entries=entries,
        similarities=detected_similarities,
        differences=detected_differences,
        missing_by_doc=missing_by_doc,
        summary=final_summary,
        chunk_ids=_dedupe(all_chunk_ids),
    )


def bind_compare_documents(
    get_chunks: Callable[[], list[EvidenceChunk]],
) -> Callable[..., Comparison]:

    def _compare_documents(
        topic: str,
        criteria: list[str],
        documents: list[DocumentEntry],
        similarities: Optional[list[str]] = None,
        differences: Optional[list[str]] = None,
        summary: Optional[str] = None,
    ) -> Comparison:
        """Organize and compare information across two or more source documents.

        Use this tool whenever the question compares multiple papers, reports,
        models, methodologies, or approaches across different documents. Every
        entry must cite real doc_ids and chunk_ids from the retrieved evidence.

        Args:
            topic: What is being compared, e.g. "CNN Architectures Across Three Papers".
            criteria: Ordered comparison dimensions, e.g.
                ["Methodology", "Accuracy", "Computational Complexity", "Limitations"].
            documents: At least 2 DocumentEntry items, each holding:
                - doc_id: Source document ID from the retrieved chunks.
                - attributes: Mapping that MUST include every key in `criteria`
                  (write "N/A" explicitly if a document does not report a criterion).
                - chunk_ids: Non-empty list of chunk IDs from that document supporting
                  the entry.
            similarities: Optional list of key commonalities identified across sources.
            differences: Optional list of qualitative contrasts identified across sources.
            summary: Optional concise synthesis of how the documents compare.
        """
        return compare_documents(
            topic=topic,
            criteria=criteria,
            documents=documents,
            chunks=get_chunks(),
            similarities=similarities,
            differences=differences,
            summary=summary,
        )

    _compare_documents.__name__ = "compare_documents"
    return _compare_documents