from langchain_core.tools import StructuredTool
from langchain_core.language_models.chat_models import BaseChatModel


def build_query_rewriter_tool(llm: BaseChatModel) -> StructuredTool:

    def query_rewriter(query: str) -> str:
        """Rewrite a user question into a clear retrieval query."""

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        prompt = f"""
                    You are a query rewriting component in a RAG system.

                    Rewrite the user's question into a concise, information-rich
                    search query that preserves its original intent.

                    Rules:
                    - Do not answer the question.
                    - Preserve important names, dates, entities, and technical terms.
                    - Do not introduce information that was not in the question.
                    - Return only the rewritten query.

                    User question:
                    {query}
                """

        response = llm.invoke(prompt)
        rewritten = response.content

        return str(rewritten).strip()

    return StructuredTool.from_function(
        func=query_rewriter,
        name="query_rewriter",
        description=(
            "Rewrite a user's question into a clearer search query "
            "before retrieving document chunks. Never answer the question."
        ),
    )