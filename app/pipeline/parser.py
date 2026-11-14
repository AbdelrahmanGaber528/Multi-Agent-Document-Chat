import re
from langchain_core.documents import Document
from app.core.errors import ParsingError


HEADING_PATTERN = re.compile(
    r"^(?:"
    r"\d+(?:\.\d+)*\.?\s+.+"
    r"|chapter\s+\d+.*"
    r"|section\s+\d+.*"
    r")$",
    re.IGNORECASE,
)


def detect_heading(document: Document) -> bool:
    
    style = str(document.metadata.get("style", "")).lower()

    if style.startswith("heading") or style == "title":
        return True

    text = document.page_content.strip()

    if not text or len(text) > 150:
        return False

    return bool(HEADING_PATTERN.match(text))


def parse_documents(documents: list[Document]) -> list[Document]:

    parsed = []
    current_section = "Introduction"

    try:
        for index, document in enumerate(documents):

            text = document.page_content.strip()

            if not text:
                continue

            metadata = dict(document.metadata)

            metadata.setdefault("doc_id", None)
            metadata.setdefault("doc_name", "unknown")
            metadata.setdefault("page", None)

            block_type = metadata.get("block_type")

            if block_type == "table":
                metadata["chunk_type"] = "table"

            elif detect_heading(document):
                current_section = text
                metadata["chunk_type"] = "heading"

            else:
                metadata["chunk_type"] = "text"

            metadata["section"] = current_section
            metadata["block_index"] = index

            parsed.append(
                Document(
                    page_content=text,
                    metadata=metadata,
                )
            )

        if not parsed:
            raise ParsingError("No content available after parsing.")

        return parsed

    except Exception as exc:
        raise ParsingError(
            "Document parsing failed.",
            details={"reason": str(exc)},
        ) from exc