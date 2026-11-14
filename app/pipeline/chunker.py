from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.errors import ChunkingError
from app.core.config import get_settings


settings = get_settings()

DEFAULT_CHUNK_SIZE = settings.chunk_size_tokens
DEFAULT_CHUNK_OVERLAP = settings.chunk_overlap_tokens



def chunk_documents(documents: list[Document],chunk_size: int = DEFAULT_CHUNK_SIZE,chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[Document]:

    if chunk_size <= 0:
        raise ChunkingError("Chunk size must be positive.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ChunkingError(
            "Chunk overlap must be non-negative and smaller than chunk size."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
        length_function=len,
        add_start_index=True,
    )


    chunks = []
    chunk_counter = 0
    current_section = {}

    try:
        for document in documents:

            metadata = dict(document.metadata)

            doc_id = metadata.get("doc_id")
            chunk_type = metadata.get("chunk_type", "text")

            # Remember headings for the following content.
            if chunk_type == "heading":
                current_section[doc_id] = document.page_content
                continue

            section = (
                metadata.get("section")
                or current_section.get(doc_id)
                or "Unknown"
            )

            metadata["section"] = section

            # Tables remain intact.
            if chunk_type == "table":
                pieces = [document]

            else:
                pieces = splitter.split_documents([document])

            for piece in pieces:

                if not piece.page_content.strip():
                    continue

                piece_metadata = dict(piece.metadata)

                piece_metadata.update({
                    "chunk_id": f"{doc_id}_{chunk_counter}",
                    "chunk_index": chunk_counter,
                    "section": section,
                    "chunk_type": chunk_type,
                })

                chunks.append(
                    Document(
                        page_content=piece.page_content,
                        metadata=piece_metadata,
                    )
                )

                chunk_counter += 1

        if not chunks:
            raise ChunkingError("No chunks were generated.")

        return chunks



    except Exception as exc:
        raise ChunkingError(
            "Failed to split documents into chunks.",
            details={"reason": str(exc)},
        ) from exc