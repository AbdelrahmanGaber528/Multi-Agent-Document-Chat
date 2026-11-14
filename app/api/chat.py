from __future__ import annotations

import logging
from functools import lru_cache
from uuid import uuid4

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from app.agents.evaluator import Evaluator, EvaluationError
from app.agents.generator import Generator, GenerationError
from app.cache.answer_cache import (
    cache_answer,
    current_corpus_version,
    get_cached_answer,
)
from app.core.contract import ChatRequest, ChatResponse
from app.core.errors import ChatUnavailable, InvalidRequest, NoKnowledgeLoaded
from app.knowledge.retriever import (
    EmptyKnowledgeBaseError,
    KnowledgeRetriever,
    RetrievalError,
)
from app.memory.hooks import build_memory_hook
from app.workflow.evaluator_generator import run_loop

log = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@lru_cache(maxsize=1)
def _components() -> tuple[KnowledgeRetriever, Generator, Evaluator]:
    return KnowledgeRetriever(), Generator(), Evaluator()


def _run_chat(
    body: ChatRequest,
    corpus_version: str,
    request_id: str,
) -> ChatResponse:
    try:
        retriever, generator, evaluator = _components()
    except Exception as exc:
        log.error("Chat components failed to initialise: %s", exc, exc_info=True)
        raise ChatUnavailable(
            "Chat models are not configured. Check the server log."
        ) from exc

    try:
        retrieval = retriever.retrieve(body.question)
    except EmptyKnowledgeBaseError as exc:
        raise NoKnowledgeLoaded(
            "The knowledge base is empty. Ingest a source before chatting."
        ) from exc
    except (RetrievalError, ValueError) as exc:
        raise ChatUnavailable(f"Knowledge retrieval failed: {exc}") from exc

    documents = [chunk.document for chunk in retrieval.chunks]
    result = run_loop(
        body.question,
        documents,
        generator,
        evaluator,
        on_event=build_memory_hook(body.session_id, request_id),
    )
    response = ChatResponse(
        answer=result.answer,
        validated=result.validated,
        iterations_used=result.iterations_used,
        insufficient_knowledge=result.insufficient_knowledge,
        sources=result.sources,
        evaluator_feedback=result.evaluator_feedback,
        cache_hit=False,
    )

    if result.validated:
        cache_answer(
            body.question,
            response.model_dump(),
            corpus_version,
        )
    return response


@router.post("", response_model=ChatResponse)
async def chat(body: ChatRequest) -> ChatResponse:
    question = body.question.strip()
    if not question:
        raise InvalidRequest("Question cannot be blank.")

    request = body.model_copy(update={"question": question})
    corpus_version = current_corpus_version()

    cached = get_cached_answer(question, corpus_version)
    if cached is not None:
        try:
            response = ChatResponse.model_validate(cached)
        except Exception:
            log.warning("Ignoring malformed cached response", exc_info=True)
        else:
            return response.model_copy(update={"cache_hit": True})

    request_id = uuid4().hex
    try:
        return await run_in_threadpool(_run_chat, request, corpus_version, request_id)
    except (GenerationError, EvaluationError) as exc:
        raise ChatUnavailable(f"Chat model failed: {exc}") from exc