import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.core.contract import ChatRequest, ChatResponse
from app.orchestrator.graph import build_graph, invoke_graph

log = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

def _run_chat(body: ChatRequest, request: Request) -> ChatResponse:
    try:
        workflow = getattr(request.app.state, "workflow", None)
        if workflow is None:
            raise RuntimeError("Chat workflow has not been initialized.")
        settings = get_settings()
        result = invoke_graph(
            workflow, body.question, body.history, max_rounds=settings.max_rounds
        )
        return ChatResponse(
            answer=result.text, citations=result.citations,
            insufficient_evidence=result.insufficient_evidence,
            missing_info=result.missing_info, rounds_used=result.rounds_used,
        )
    except Exception as exc:
        log.exception("Chat execution failed")
        raise HTTPException(status_code=500, detail="Chat processing failed.") from exc

@router.post("", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be blank.")
    clean_body = body.model_copy(update={"question": question})
    return await run_in_threadpool(_run_chat, clean_body, request)
