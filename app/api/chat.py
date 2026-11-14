import logging
from functools import lru_cache

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.agents.retriever import RetrieverAgent
from app.pipeline.vector_db import VectorDatabase

log = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

settings = get_settings()


@lru_cache(maxsize=1)
def _components():
    """Initialize reusable chat components."""

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
    )

    vector_db = VectorDatabase(
        embedding_model=settings.embedding_model,
    )

    retriever = RetrieverAgent(
        llm=llm,
        vector_db=vector_db,
    )

    return retriever


def _run_chat(question: str):
    retriever = _components()

    retrieval_result = retriever.invoke(question)

    return {
        "question": question,
        "retrieval": retrieval_result,
    }


@router.post("")
async def chat(body: dict):
    """Retrieve document evidence for a chat question."""

    question = body.get("question", "").strip()

    if not question:
        raise ValueError("Question cannot be blank.")

    try:
        result = await run_in_threadpool(
            _run_chat,
            question,
        )

        return result

    except Exception as exc:
        log.exception("Chat request failed")
        raise RuntimeError(
            f"Chat processing failed: {exc}"
        ) from exc