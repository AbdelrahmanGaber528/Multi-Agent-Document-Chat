from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from app.core.config import get_settings
from app.core.errors import ConfigurationError
from app.pipeline.vector_db import VectorDatabase
from .tools import build_retriever_tools

settings = get_settings()


RETRIEVER_SYSTEM_PROMPT = """
You are the Retriever Agent in a Multi-Agent Document Chat system.

Your responsibility is to retrieve relevant evidence from uploaded
documents and prepare it for the Analyst Agent.

You must NOT answer the user's question.

AVAILABLE TOOLS:

1. query_rewriter
   Rewrites the original question into a clearer retrieval query.

2. semantic_search
   Retrieves chunks based on semantic similarity.

3. keyword_search
   Retrieves chunks based on exact lexical matching.

4. metadata_filter
   Retrieves chunks using metadata constraints.

5. reranker
   Reorders retrieved chunks according to their relevance.

6. context_selector
   Selects the final evidence within the context budget.

RETRIEVAL WORKFLOW:

1. Understand the original user question.
2. Call query_rewriter.
3. Use the rewritten query with semantic_search.
4. Use keyword_search to find exact terms and identifiers.
5. If the user specifies metadata constraints, use metadata_filter.
6. Combine the retrieved candidates.
7. Pass the combined candidates to reranker.
8. Pass the reranked candidates to context_selector.
9. Return the selected evidence to the Analyst Agent.

IMPORTANT RULES:

- Do not answer the user's question.
- Never invent document content or citations.
- Preserve chunk_id, doc_id, page, and metadata.
- Do not modify citation metadata.
- Do not invent document IDs or metadata filters.
- Do not use metadata_filter when no metadata constraint is needed.
- Do not skip reranking or context selection.
- Avoid unnecessary repeated tool calls.
- If no evidence is found, clearly report that.
- Keep the original question available for relevance evaluation.

FINAL OUTPUT:

Return a concise structured evidence report containing:

- original_query
- rewritten_query
- evidence: the selected chunks, preserving their metadata
- evidence_count
- retrieval_status

The report is intended for the Analyst Agent, not the end user.
"""



class RetrieverAgent:

    def __init__(self,llm: BaseChatModel,vector_db: VectorDatabase | None = None):
        
        if llm is None:
            raise ConfigurationError(
                "Retriever Agent requires an LLM."
            )

        self.llm = llm

        self.vector_db = vector_db or VectorDatabase(
            embedding_model=settings.embedding_model,
        )

        self.tools = build_retriever_tools(
            llm=self.llm,
            vector_db=self.vector_db,
        )

        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=RETRIEVER_SYSTEM_PROMPT,
        )


    def invoke(self, query: str) -> dict:
        """Run the Retriever Agent for a user query."""

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        return self.agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": query.strip(),
                }
            ]
        })