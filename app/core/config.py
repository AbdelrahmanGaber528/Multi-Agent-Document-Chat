from functools import lru_cache # lru : least recently used , cache for function's return values .
from pathlib import Path
from typing import Optional

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    
    app_name: str = "Multi-Agent Document Chat"
    log_level: str = "INFO"


    data_dir: Path = PROJECT_ROOT / "data"
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"
    vector_db_dir: Path = PROJECT_ROOT / "data" / "vector_db"


    # ── Upload limits
    allowed_extensions: tuple[str, ...] = (".pdf", ".docx", ".txt")
    max_upload_mb: int = 50

    
    # ── LLM
    llm_provider: str = "openrouter"
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "openrouter/free"
    llm_temperature: float = 0.0
    llm_api_key : str = None
    llm_timeout_s: int = Field(default=60, ge=1)
    llm_max_retries: int = Field(default=3, ge=0) 

    # ── Embeddings
    embedding_provider: str = "sentence-transformers"
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2" 
    embedding_batch_size: int = Field(default=64, ge=1)
    embedding_api_key: str = None

    # ── Chunking 
    chunk_size_tokens: int = Field(default=500, ge=100, le=1000)
    chunk_overlap_tokens: int = Field(default=80, ge=0)

    # ── Vector DB
    vector_db_backend: str = "chroma"
    vector_collection: str = "documents"
    vector_db_url: Optional[str] = None  # set when the DB runs in docker-compose


    # ── Retrieval
    top_k_semantic: int = Field(default=20, ge=1)
    top_k_keyword: int = Field(default=20, ge=1)
    top_k_reranked: int = Field(default=10, ge=1)
    final_context_chunks: int = Field(default=8, ge=1)  # hard cap sent to Analyst
    context_token_budget: int = Field(default=6000, ge=500)
    rrf_k: int = Field(default=60, ge=1)
    use_cross_encoder: bool = False
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_query_variants: int = Field(default=3, ge=1, le=5)


    # ── Orchestration 
    max_rounds: int = Field(default=3, ge=1, le=10)


    # ── Validators
    
    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, v: str) -> str:
        return v.upper()

   
    @field_validator("allowed_extensions")
    @classmethod
    def _normalize_extensions(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(e.lower() if e.startswith(".") else f".{e.lower()}" for e in v)


    @model_validator(mode="after")
    def _check_chunking(self) -> "Settings":
        
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
        
        if self.final_context_chunks > self.top_k_reranked:
            raise ValueError("final_context_chunks must be <= top_k_reranked")
        
        return self



    # ── Helpers
    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def ensure_dirs(self) -> None:
        """Create data folders. Call once at startup (main.py)."""
        for p in (self.data_dir, self.upload_dir, self.vector_db_dir):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Cached singleton."""
    return Settings()