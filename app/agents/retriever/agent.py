import json
from typing import Any
from langchain_core.language_models.chat_models import BaseChatModel
from app.core.config import get_settings
from app.core.contract import EvidenceBundle, EvidenceChunk
from app.core.errors import ConfigurationError
from app.pipeline.vector_db import VectorDatabase
from .tools import build_retriever_tools

settings = get_settings()

class RetrieverAgent:
    """Runs the configured retrieval tools and returns a validated EvidenceBundle."""

    def __init__(self, llm: BaseChatModel, vector_db: VectorDatabase | None = None):
        
        if llm is None:
            raise ConfigurationError("Retriever Agent requires an LLM.")
        
        self.llm = llm
        self.vector_db = vector_db or VectorDatabase(embedding_model=settings.embedding_model)
        self.tools = {tool.name: tool for tool in build_retriever_tools(llm, self.vector_db)}


    def invoke(self, query: str) -> EvidenceBundle:
        
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")
        query = query.strip()

        try:
            rewritten = str(self.tools["query_rewriter"].invoke({"query": query})).strip() or query
            
        except Exception:
            rewritten = query

        candidates: dict[str, dict[str, Any]] = {}
        
        
        for tool_name, tool_query in (("semantic_search", rewritten), ("keyword_search", rewritten)):
            
            raw = self.tools[tool_name].invoke({"query": tool_query})
            
            
            try:
                
                rows = json.loads(raw) if isinstance(raw, str) else raw
                
            except json.JSONDecodeError:
                rows = []
                
            if isinstance(rows, list):
                for row in rows:
                    metadata = row.get("metadata") or {}
                    chunk_id = row.get("chunk_id") or metadata.get("chunk_id")
                    if not chunk_id:
                        continue
                    
                    old = candidates.get(str(chunk_id), {})
                    old_metadata = old.get("metadata") or {}

                    candidates[str(chunk_id)] = {
                        **old,
                        **row,
                        "chunk_id": str(chunk_id),
                        "metadata": {**old_metadata, **metadata},
                    }

        candidate_list = list(candidates.values())
        
        
        if candidate_list:
            
            reranked_raw = self.tools["reranker"].invoke({
                "query": query, "candidates_json": json.dumps(candidate_list, ensure_ascii=False)
            })
            try:
                candidate_list = json.loads(reranked_raw)
            except (TypeError, json.JSONDecodeError):
                pass
            selected_raw = self.tools["context_selector"].invoke({
                "candidates_json": json.dumps(candidate_list, ensure_ascii=False)
            })
            try:
                selected = json.loads(selected_raw).get("chunks", [])
            except (TypeError, json.JSONDecodeError, AttributeError):
                selected = candidate_list[:settings.final_context_chunks]
                
        else:
            selected = []

        chunks = []
        
        for row in selected:
            
            metadata = row.get("metadata") or {}
            text = row.get("content") or ""
            chunk_id = row.get("chunk_id") or metadata.get("chunk_id")
            doc_id = row.get("doc_id") or metadata.get("doc_id")
            doc_name = metadata.get("doc_name") or metadata.get("filename") or metadata.get("source")
            
            if not (text.strip() and chunk_id and doc_id and doc_name):
                continue
            
            page = metadata.get("page")
            
            try:
                page = int(page) if page is not None else None
                if page is not None and page < 1:
                    page = None
                    
            except (TypeError, ValueError):
                page = None
            chunk_type = metadata.get("chunk_type", "text")
            
            if chunk_type not in ("text", "table"):
                chunk_type = "text"
                
            chunks.append(EvidenceChunk(
                chunk_id=str(chunk_id), doc_id=str(doc_id), doc_name=str(doc_name),
                page=page, section=metadata.get("section"), text=text,
                chunk_type=chunk_type, score=float(row.get("rerank_score", row.get("keyword_score", 0.0)) or 0.0),
            ))

        return EvidenceBundle(query=query, rewritten_queries=[rewritten], chunks=chunks)