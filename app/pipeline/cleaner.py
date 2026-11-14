import re
import unicodedata
from langchain_core.documents import Document
from app.core.errors import ParsingError


def clean_text(text: str) -> str:
    
    if not isinstance(text, str):
        raise TypeError("Document content must be a string.")

    # Normalize Unicode representations.
    text = unicodedata.normalize("NFC", text)

    # Normalize line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove null characters.
    text = text.replace("\x00", "")

    # Remove horizontal whitespace while preserving newlines.
    text = re.sub(r"[^\S\n]+", " ", text)

    # Remove excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()



def clean_documents(documents: list[Document]) -> list[Document]:

    cleaned = []

    try:
        for document in documents:

            content = clean_text(document.page_content)

            if not content:
                continue

            cleaned.append(
                Document(
                    page_content=content,
                    metadata=dict(document.metadata),
                )
            )

        return cleaned

    except Exception as exc:
        raise ParsingError(
            "Document cleaning failed.",
            details={"reason": str(exc)},
        ) from exc