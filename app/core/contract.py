from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class EvidenceChunk(BaseModel):
    """One retrieved piece of a document: a text passage or a whole table."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(..., min_length=1, description="Unique ID of the chunk")
    doc_id: str = Field(..., min_length=1, description="ID of the source document")
    doc_name: str = Field(..., min_length=1, description="Original file name, used in citations")
    page: Optional[int] = Field(default=None, ge=1, description="1-based page number, if known")
    section: Optional[str] = Field(default=None, description="Section or heading, if known")
    text: str = Field(..., description="Chunk content. Markdown when chunk_type == 'table'")
    chunk_type: Literal["text", "table"] = "text"
    score: float = Field(default=0.0, description="Reranker relevance score (higher is better)")


class EvidenceBundle(BaseModel): 
    """What the Retriever hands to the Analyst."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., description="The user's original question")
    rewritten_queries: list[str] = Field(default_factory=list)
    chunks: list[EvidenceChunk] = Field(default_factory=list)


class IngestResult(BaseModel):
    """What the ingestion facade returns after processing an uploaded file."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(..., min_length=1)
    num_chunks: int = Field(..., ge=0)