import re
from typing import Optional
from app.core.contract import EvidenceChunk, ExtractedTable

MAX_CAPTION_LENGTH = 120

_SEPARATOR_RE = re.compile(r"^\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)*\|?$")
_CELL_SPLIT_RE = re.compile(r"(?<!\\)\|")  # split on | but not on an escaped \|
_MD_EMPHASIS_RE = re.compile(r"\*{1,2}(.+?)\*{1,2}")
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_MISSING = {"", "-", "\u2013", "\u2014", "n/a", "na", "nan", "none", "null", "?"}



def _clean_cell(cell: str) -> str:
    """Strip whitespace, Markdown bold/italic markers, <br> tags and escaped pipes."""
    cleaned = _BR_RE.sub(" ", cell.strip()).replace("\\|", "|")
    return _MD_EMPHASIS_RE.sub(r"\1", cleaned).strip()


def _split_row(line: str) -> list[str]:
    """Split a pipe-delimited row into cleaned cell values."""
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|") and not stripped.endswith("\\|"):
        stripped = stripped[:-1]
    return [_clean_cell(cell) for cell in _CELL_SPLIT_RE.split(stripped)]


def _is_separator(line: str) -> bool:
    return "|" in line and bool(_SEPARATOR_RE.match(line.strip()))


def _find_caption(lines: list[str], start: int) -> Optional[str]:
    """Use the nearest non-blank line above the table as caption if it looks like one."""
    k = start - 1
    while k >= 0 and not lines[k]:
        k -= 1
    if k < 0 or "|" in lines[k]:
        return None
    candidate = lines[k].strip("#* ").strip()
    if candidate and len(candidate) <= MAX_CAPTION_LENGTH:
        return candidate
    return None


def _build_table(
    chunk_id: str,
    caption: Optional[str],
    header: list[str],
    body: list[list[str]],
) -> ExtractedTable:
    """Make every row as wide as the widest row, so no cell is silently dropped."""
    width = max([len(header)] + [len(r) for r in body])
    headers = header + [f"col_{k + 1}" for k in range(len(header), width)]
    rows = [r + [""] * (width - len(r)) for r in body]
    return ExtractedTable(chunk_id=chunk_id, caption=caption, headers=headers, rows=rows)



def extract_tables(
    text: str,
    chunk_id: str,
    caption: Optional[str] = None,
) -> list[ExtractedTable]:
    """Identify and extract all structured tables from one evidence chunk.

    Use this tool whenever a chunk contains tabular data (accuracy, precision,
    recall, F1-score, model parameters, experimental or financial figures)
    BEFORE comparing models or computing statistics. Do not read numbers out of
    the raw table text yourself.

    Handles standard Markdown tables (header, `| --- |` separator, rows) and
    pipe-delimited tables whose separator row was lost in PDF extraction (the
    first row is then treated as the header). Several tables in one chunk are
    returned as separate results; a blank line ends a table.

    Args:
        text: Full text of the evidence chunk (Markdown for chunk_type == "table").
        chunk_id: ID of the chunk the text came from, kept for citations.
        caption: Optional title for the tables. If omitted, a short line directly
            above each table is used when it looks like a caption.

    Returns:
        A list of ExtractedTable (headers plus rows of strings), in reading order.
        The list is EMPTY when the text contains no table: that is a normal
        outcome, so treat the chunk as plain text.

    Raises:
        TableExtractorError: If chunk_id or text is empty.
    """
    if not chunk_id or not chunk_id.strip():
        raise TableExtractorError("chunk_id is required.")
    if not text or not text.strip():
        raise TableExtractorError("Chunk text is empty.")

    chunk_id = chunk_id.strip()
    lines = [line.strip() for line in text.strip().splitlines()]  # blank lines kept as boundaries
    n = len(lines)
    tables: list[ExtractedTable] = []
    i = 0

    while i < n:
        line = lines[i]
        if "|" not in line or _is_separator(line):
            i += 1
            continue

        # Case 1: standard Markdown table (header, separator, rows)
        if i + 1 < n and _is_separator(lines[i + 1]):
            header = _split_row(line)
            j = i + 2
            body: list[list[str]] = []
            while j < n and "|" in lines[j]:
                body.append(_split_row(lines[j]))
                j += 1
            if any(header) and body:
                cap = caption if caption is not None else _find_caption(lines, i)
                tables.append(_build_table(chunk_id, cap, header, body))
            i = j
            continue

        # Case 2: pipe-delimited block with no separator row
        block: list[list[str]] = []
        j = i
        while j < n and "|" in lines[j] and not _is_separator(lines[j]):
            row = _split_row(lines[j])
            if len(row) < 2:
                break
            block.append(row)
            j += 1

        if len(block) >= 2:
            cap = caption if caption is not None else _find_caption(lines, i)
            tables.append(_build_table(chunk_id, cap, block[0], block[1:]))
            i = j
        else:
            i += 1

    return tables


def extract_from_chunks(chunks: list[EvidenceChunk]) -> list[ExtractedTable]:
    """Extract tables from every chunk with chunk_type == "table" in a bundle."""
    tables: list[ExtractedTable] = []
    for chunk in chunks:
        if chunk.chunk_type == "table" and chunk.text.strip():
            tables.extend(extract_tables(chunk.text, chunk.chunk_id))
    return tables


def parse_number(cell: str) -> Optional[float]:
    """Convert a table cell to a float, or None if it is not a number."""
    if cell is None:
        return None
    s = cell.strip().replace("\u2212", "-")
    if s.lower() in _MISSING:
        return None

    negative = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    s = re.split(r"\u00b1|\+/-", s)[0].strip()

    sign = ""
    if s and s[0] in "+-":
        sign, s = s[0], s[1:].strip()
    s = sign + s.lstrip("$\u20ac\u00a3\u00a5").rstrip("%*\u2020\u2021 ").strip()
    # -------------------------------------------------------------

    s = _THOUSANDS_RE.sub("", s)

    if not _NUMBER_RE.fullmatch(s):
        return None
    value = float(s)
    return -abs(value) if negative else value

def _unique_headers(headers: list[str]) -> list[str]:
    """Give empty or duplicate headers distinct names so columns can be looked up."""
    seen: dict[str, int] = {}
    names: list[str] = []
    for k, header in enumerate(headers, start=1):
        name = header or f"col_{k}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        names.append(name)
    return names


def to_records(table: ExtractedTable) -> list[dict[str, str]]:
    """Return each row as {column name: cell text}, e.g. {"Model": "BERT", "F1": "91.2"}."""
    names = _unique_headers(table.headers)
    return [dict(zip(names, row)) for row in table.rows]


def get_column(
    table: ExtractedTable,
    column: str,
    numeric: bool = False,
) -> list[str] | list[Optional[float]]:

    names = _unique_headers(table.headers)
    lookup = {name.lower(): idx for idx, name in enumerate(names)}
    idx = lookup.get(column.strip().lower())
    if idx is None:
        raise TableExtractorError(f"Column '{column}' not found. Available columns: {names}")

    values = [row[idx] if idx < len(row) else "" for row in table.rows]
    return [parse_number(v) for v in values] if numeric else values