from typing import Any, Optional


class AppError(Exception):
    """Base class for all expected, user-facing errors."""

    status_code: int = 500
    code: str = "app_error"

    def __init__(self, message: str, *, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"error": self.code, "message": self.message, "details": self.details}

    def __str__(self) -> str:
        return self.message



class ConfigurationError(AppError):
    status_code = 500
    code = "configuration_error"



class IngestionError(AppError):
    """Parent of everything that can go wrong while preparing a document."""

    status_code = 422
    code = "ingestion_error"


class UnsupportedFileError(IngestionError):
    status_code = 415
    code = "unsupported_file_type"


class FileTooLargeError(IngestionError):
    status_code = 413
    code = "file_too_large"


class ParsingError(IngestionError):
    """File is corrupt, encrypted, or text/tables could not be extracted."""

    code = "parsing_error"


class EmptyDocumentError(IngestionError):
    """Nothing usable came out (e.g. scanned PDF with no OCR)."""

    code = "empty_document"


class ChunkingError(IngestionError):
    code = "chunking_error"


class EmbeddingError(AppError):
    status_code = 502
    code = "embedding_error"


class VectorStoreError(AppError):
    status_code = 500
    code = "vector_store_error"


class DocumentNotFoundError(AppError):
    status_code = 404
    code = "document_not_found"



# ── Retrieval (agents/retriever/) ─────────────────────────────────────
class RetrievalError(AppError):
    status_code = 500
    code = "retrieval_error"


class QueryRewriteError(RetrievalError):
    """LLM rewrite failed. The retriever should fall back to the raw query."""

    code = "query_rewrite_error"


class EmptyIndexError(RetrievalError):
    """Search was attempted before any document was ingested."""

    status_code = 409
    code = "empty_index"



# ── Analysis (agents/analyst/) ────────────────────────────────────────
class AnalysisError(AppError, ValueError):
    """Parent of every error raised by the Analyst tools."""

    status_code = 422
    code = "analysis_error"


class TableExtractorError(AnalysisError):
    """Invalid input to the table tools (empty text, missing chunk_id, unknown column)."""

    code = "table_extractor_error"


class DocumentComparisonError(AnalysisError):
    """Invalid, ungrounded, or insufficient inputs for a multi-document comparison."""

    code = "document_comparison_error"


class DataAnalysisError(AnalysisError):
    """Input data is empty, mismatched, too small, or contains no valid numeric values."""

    code = "data_analysis_error"


class CalculatorError(AnalysisError):
    """The expression is invalid, unsafe, or has no finite numeric result."""

    code = "calculator_error"
    
    
    
    
# ── Answer (agents/answer/) ───────────────────────────────────────────
class AnswerError(AppError):
    """Parent of every error raised while writing the final answer."""
 
    status_code = 500
    code = "answer_error"
 
 
class AnswerGenerationError(AnswerError):
    """The LLM call failed or returned an empty / unusable answer."""
 
    status_code = 502
    code = "answer_generation_error"
 
 
class CitationError(AnswerError, ValueError):
    """A citation refers to a chunk_id that is not in the evidence, or lacks doc/page info."""
 
    status_code = 422
    code = "citation_error"
 
 
class SourceFormatterError(AnswerError, ValueError):
    """The list of evidence sources could not be built (empty or malformed input)."""
 
    status_code = 422
    code = "source_formatter_error"
 
 
class ResponseFormatterError(AnswerError, ValueError):
    """The answer could not be organized into text or tables (bad table shape, empty content)."""
 
    status_code = 422
    code = "response_formatter_error"