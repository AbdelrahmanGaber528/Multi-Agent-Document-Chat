from langchain_core.language_models.chat_models import BaseChatModel
from app.pipeline.vector_db import VectorDatabase
from .query_rewriter import build_query_rewriter_tool
from .semantic_search import build_semantic_search_tool
from .keyword_search import build_keyword_search_tool
from .metadata_filter import build_metadata_filter_tool
from .reranker import build_reranker_tool
from .context_selector import build_context_selector_tool


def build_retriever_tools(llm: BaseChatModel, vector_db: VectorDatabase):
    return [
        build_query_rewriter_tool(llm),
        build_semantic_search_tool(vector_db),
        build_keyword_search_tool(vector_db),
        build_metadata_filter_tool(vector_db),
        build_reranker_tool(),
        build_context_selector_tool(),
    ]