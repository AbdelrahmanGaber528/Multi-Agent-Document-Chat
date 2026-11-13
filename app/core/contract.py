from typing import Any, Literal, Optional, Union

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


# ── Analyst output models ──────────────────────────────────────────
# These are passed to the Answer Agent and the orchestrator.


class Finding(BaseModel):
    """One fact or conclusion backed by specific evidence chunks."""

    model_config = ConfigDict(extra="forbid")

    statement: str = Field(..., min_length=1)
    chunk_ids: list[str] = Field(default_factory=list, description="Chunks supporting this statement")


class Calculation(BaseModel):
    """A numeric result produced by the calculator tool (never by the LLM itself)."""

    model_config = ConfigDict(extra="forbid")

    description: str = Field(..., description="What was calculated, in plain words")
    expression: str = Field(..., description="Expression passed to the calculator")
    result: Union[float, int, str]
    chunk_ids: list[str] = Field(default_factory=list, description="Chunks the inputs came from")


class ExtractedTable(BaseModel):
    """A table parsed from a chunk with chunk_type == 'table'."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    caption: Optional[str] = Field(default=None, description="Short title or heading above the table")
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)


class MetricAnalysis(BaseModel):
    """Output of a data-analysis tool: ranking, trend, or correlation."""

    model_config = ConfigDict(extra="forbid")

    analysis_type: Literal["ranking", "trend", "correlation"] = Field(
        ..., description="Kind of analysis performed"
    )
    metric: str = Field(..., min_length=1, description="Name of the metric analysed")
    summary: str = Field(..., description="One-paragraph human-readable summary")
    chunk_ids: list[str] = Field(
        default_factory=list, description="Evidence chunks the analysis drew on"
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured results (rankings, statistics, trend data, etc.)",
    )




class DocumentEntry(BaseModel):
    """One document's data in a multi-document comparison."""

    model_config = ConfigDict(extra="forbid", coerce_numbers_to_str=True)

    doc_id: str = Field(..., min_length=1, description="Source document ID from EvidenceChunk")
    doc_name: str = Field(default="", description="Display file name; resolved from chunks")
    attributes: dict[str, Optional[str]] = Field(default_factory=dict)
    chunk_ids: list[str] = Field(..., min_length=1, description="Chunk IDs supporting this document")


class MissingTarget(BaseModel):
    """An information gap identified during document comparison."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(..., min_length=1)
    doc_name: str = Field(..., min_length=1)
    criterion: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)


class Comparison(BaseModel):
    """Result of comparing information across documents."""

    model_config = ConfigDict(extra="forbid")

    topic: str = Field(..., min_length=1)
    criteria: list[str] = Field(default_factory=list)
    entries: list[DocumentEntry] = Field(default_factory=list)
    similarities: list[str] = Field(default_factory=list)
    differences: list[str] = Field(default_factory=list)
    missing_by_doc: dict[str, list[str]] = Field(default_factory=dict)
    summary: str = ""
    chunk_ids: list[str] = Field(default_factory=list)



class AnalysisResult(BaseModel):
    """Output of the Analyst Agent."""

    model_config = ConfigDict(extra="forbid")

    question: str
    findings: list[Finding] = Field(default_factory=list)
    calculations: list[Calculation] = Field(default_factory=list)
    tables: list[ExtractedTable] = Field(default_factory=list)
    comparisons: list[Comparison] = Field(default_factory=list)
    metric_analyses: list[MetricAnalysis] = Field(
        default_factory=list,
        description="Rankings, trends, and correlations produced by data-analysis tools",
    )
    missing_info: list[str] = Field(
        default_factory=list,
        description="Gaps that trigger the feedback loop back to the Retriever",
    )

    @property
    def is_sufficient(self) -> bool:
        """True when the Analyst has nothing left to ask the Retriever for."""
        return len(self.missing_info) == 0

    @property
    def used_chunk_ids(self) -> list[str]:
        """Every chunk_id referenced anywhere in the analysis, de-duplicated, in order."""
        seen: dict[str, None] = {}
        for group in (self.findings, self.calculations, self.comparisons, self.metric_analyses):
            for item in group:
                for cid in item.chunk_ids:
                    seen.setdefault(cid, None)
                if isinstance(item, Comparison):
                    for entry in item.entries:
                        for cid in entry.chunk_ids:
                            seen.setdefault(cid, None)
        return list(seen)


class Citation(BaseModel):
    """A source reference shown to the user. Built only from chunks in the evidence."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    doc_id: str
    doc_name: str
    page: Optional[int] = Field(default=None, ge=1)
    section: Optional[str] = None


class FinalAnswer(BaseModel):
    """Output of the Answer Agent and of orchestrator.run()."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., description="Final response; may contain Markdown tables")
    citations: list[Citation] = Field(default_factory=list)
    insufficient_evidence: bool = Field(
        default=False,
        description="True if max rounds was reached while information was still missing",
    )
    missing_info: list[str] = Field(
        default_factory=list,
        description="What could not be found, shown to the user when insufficient_evidence is True",
    )
    rounds_used: int = Field(default=1, ge=1)