import logging
from functools import lru_cache

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.core.contract import ChatRequest, ChatResponse

log = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@lru_cache(maxsize=1)
def get_workflow():
    from app.orchestrator.graph import build_graph

    return build_graph()


def _run_chat(body: ChatRequest) -> ChatResponse:
    try:
        workflow = get_workflow()

        initial_state = {
            "question": body.question.strip(),
            "session_id": body.session_id,
        }

        result = workflow.invoke(initial_state)

        return ChatResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
        )

    except Exception as exc:
        log.exception("Chat execution failed")
        raise HTTPException(
            status_code=500,
            detail="Chat processing failed.",
        ) from exc


@router.post("", response_model=ChatResponse)
async def chat(body: ChatRequest):
    question = body.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be blank.",
        )

    request = body.model_copy(
        update={"question": question}
    )

    return await run_in_threadpool(_run_chat, request)