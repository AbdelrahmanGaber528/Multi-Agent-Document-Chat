import logging
import re

from app.core.contract import EvidenceChunk

logger = logging.getLogger(__name__)

# Matches single [chunk_id] brackets not followed by '(' (ignores Markdown links)
_SINGLE_BRACKET_PATTERN = re.compile(r"\[([^\[\]\n]+)\](?!\()")
_MULTI_ID_BRACKET_PATTERN = re.compile(r"\[([^\[\]\n,]+(?:\s*,\s*[^\[\]\n,]+)+)\](?!\()")


def format_source_label(chunk: EvidenceChunk) -> str:
    parts = [chunk.doc_name]
    if chunk.page is not None:
        parts.append(f"p. {chunk.page}")
    if chunk.section:
        parts.append(f"section '{chunk.section}'")
    return ", ".join(parts)


def resolve_inline_citations(
    text: str,
    allowed_chunks: list[EvidenceChunk],
    bundle_chunk_ids: set[str],
) -> tuple[str, list[str]]:
    """Replace [chunk_id] markers matching allowed_chunks with plain (label) citations.

    Markers referencing bundle chunks not used by the Analyst are stripped cleanly;
    unrelated prose brackets (e.g., [1], [Note]) are preserved.
    """
    allowed_lookup = {c.chunk_id: c for c in allowed_chunks}
    referenced_ids: list[str] = []

    def _strip_multi_id_marker(match: re.Match[str]) -> str:
        ids = [part.strip() for part in match.group(1).split(",")]
        if any(cid in bundle_chunk_ids for cid in ids):
            logger.warning("Stripped invalid multi-ID inline citation marker '%s'", match.group(0))
            return ""
        return match.group(0)

    def _replace(match: re.Match[str]) -> str:
        cid = match.group(1).strip()
        chunk = allowed_lookup.get(cid)
        if chunk is not None:
            if cid not in referenced_ids:
                referenced_ids.append(cid)
            return f"({format_source_label(chunk)})"
        if cid in bundle_chunk_ids:
            logger.warning("Stripped unverified inline citation marker '[%s]'", cid)
            return ""
        return match.group(0)

    resolved_text = _MULTI_ID_BRACKET_PATTERN.sub(_strip_multi_id_marker, text)
    resolved_text = _SINGLE_BRACKET_PATTERN.sub(_replace, resolved_text)
    # Collapse adjacent duplicate labels like "(doc.pdf, p. 2) (doc.pdf, p. 2)"
    resolved_text = re.sub(r"(\([^()]+\))(?:\s*\1)+", r"\1", resolved_text)
    # Ensure adjacent distinct labels are separated by a space: "(A)(B)" -> "(A) (B)"
    resolved_text = resolved_text.replace(")(", ") (")
    # Clean up whitespace before punctuation if an unverified marker was stripped
    resolved_text = re.sub(r"\s+([.,;:!?])", r"\1", resolved_text)
    return resolved_text.strip(), referenced_ids