from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from app.core.errors import ConfigurationError


RETRIEVER_SYSTEM_PROMPT = """

You are the Retriever Agent in a Multi-Agent Document Chat system.

Your responsibility is to retrieve relevant evidence from the
user's uploaded documents and prepare it for the Analyst Agent.

Your workflow:

1. Understand the user's information request.
2. Identify which documents should be searched.
3. Rewrite the query when necessary.
4. Use the available retrieval tools.
5. Gather relevant evidence from the documents.
6. Return the evidence with its original metadata.

Rules:

- Never answer the user's question directly.
- Never invent document content or citations.
- Always use retrieval tools to obtain evidence.
- Respect document IDs and document filters.
- Preserve chunk IDs, document IDs, page numbers, and source metadata.
- If evidence is insufficient, explicitly report that.
- Avoid retrieving irrelevant or duplicate chunks.

Your output must be useful to the Analyst Agent.
"""


class RetrieverAgent:

    def __init__( self, llm, tools: list[BaseTool]):

        if llm is None:
            raise ConfigurationError(
                "Retriever Agent requires an LLM."
            )

        if not tools:
            raise ConfigurationError(
                "Retriever Agent requires at least one tool."
            )

        self.llm = llm
        self.tools = tools

        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=RETRIEVER_SYSTEM_PROMPT,
        )



    def invoke(self, query: str) -> dict:
        """
        Execute the retriever agent.

        Returns the LangChain agent state.
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        return self.agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": query,
                }
            ]
        })