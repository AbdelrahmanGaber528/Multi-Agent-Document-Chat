"""Voice chat endpoint: transcribe a recording, then run the existing chat workflow."""

import json
import logging
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError

from app.api.chat import _run_chat
from app.core.config import get_settings
from app.core.contract import ChatRequest, ChatResponse
from app.services.transcription import Transcriber, TranscriptionError

log = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


class VoiceChatResponse(ChatResponse):
    transcript: str


def _transcribe(transcriber: Transcriber, audio: bytes) -> str:
    return transcriber.transcribe(audio)


@router.post("/voice", response_model=VoiceChatResponse)
async def voice_chat(
    request: Request,
    audio: UploadFile = File(...),
    history: str = Form("[]"),
) -> VoiceChatResponse:
    if audio.content_type not in {"audio/wav", "audio/x-wav", "audio/wave"}:
        raise HTTPException(status_code=415, detail="Audio must be a WAV recording.")

    settings = get_settings()
    content = await audio.read(settings.max_audio_upload_bytes + 1)
    if not content:
        raise HTTPException(status_code=400, detail="Audio recording is empty.")
    if len(content) > settings.max_audio_upload_bytes:
        raise HTTPException(status_code=413, detail="Audio recording is too large.")

    try:
        parsed_history: Any = json.loads(history)
        if not isinstance(parsed_history, list) or any(
            not isinstance(item, dict) for item in parsed_history
        ):
            raise ValueError("History must be a JSON array of message objects.")
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid chat history.") from exc

    transcriber = getattr(request.app.state, "transcriber", None)
    if transcriber is None:
        transcriber = Transcriber(settings.whisper_model)
        request.app.state.transcriber = transcriber

    try:
        transcript = await run_in_threadpool(_transcribe, transcriber, content)
    except TranscriptionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        chat_request = ChatRequest(question=transcript, history=parsed_history)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail="Invalid chat history.") from exc

    result = await run_in_threadpool(_run_chat, chat_request, request)
    return VoiceChatResponse(transcript=transcript, **result.model_dump())
