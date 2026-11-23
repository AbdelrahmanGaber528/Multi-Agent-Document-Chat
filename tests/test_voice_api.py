from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import voice
from app.core.contract import ChatResponse
from app.services.transcription import TranscriptionError


class FakeTranscriber:
    def __init__(self, transcript="What is the summary?"):
        self.transcript = transcript

    def transcribe(self, _audio):
        if isinstance(self.transcript, Exception):
            raise self.transcript
        return self.transcript


def make_client(monkeypatch, transcriber=None):
    app = FastAPI()
    app.include_router(voice.router)
    app.state.transcriber = transcriber or FakeTranscriber()
    monkeypatch.setattr(
        voice,
        "get_settings",
        lambda: SimpleNamespace(max_audio_upload_bytes=100, whisper_model="base.en"),
    )
    monkeypatch.setattr(
        voice,
        "_run_chat",
        lambda body, request: ChatResponse(answer=f"Answer to: {body.question}"),
    )
    return TestClient(app)


def test_voice_endpoint_transcribes_and_returns_chat_answer(monkeypatch):
    client = make_client(monkeypatch)
    response = client.post(
        "/chat/voice",
        files={"audio": ("recording.wav", b"wav", "audio/wav")},
        data={"history": "[]"},
    )

    assert response.status_code == 200
    assert response.json()["transcript"] == "What is the summary?"
    assert response.json()["answer"] == "Answer to: What is the summary?"


def test_voice_endpoint_rejects_unsupported_audio(monkeypatch):
    response = make_client(monkeypatch).post(
        "/chat/voice",
        files={"audio": ("recording.mp3", b"audio", "audio/mpeg")},
    )
    assert response.status_code == 415


def test_voice_endpoint_rejects_empty_audio(monkeypatch):
    response = make_client(monkeypatch).post(
        "/chat/voice",
        files={"audio": ("recording.wav", b"", "audio/wav")},
    )
    assert response.status_code == 400


def test_voice_endpoint_enforces_audio_size_limit(monkeypatch):
    response = make_client(monkeypatch).post(
        "/chat/voice",
        files={"audio": ("recording.wav", b"x" * 101, "audio/wav")},
    )
    assert response.status_code == 413


def test_voice_endpoint_rejects_invalid_history(monkeypatch):
    response = make_client(monkeypatch).post(
        "/chat/voice",
        files={"audio": ("recording.wav", b"wav", "audio/wav")},
        data={"history": "not-json"},
    )
    assert response.status_code == 400


def test_voice_endpoint_returns_transcription_error(monkeypatch):
    client = make_client(monkeypatch, FakeTranscriber(TranscriptionError("No speech")))
    response = client.post(
        "/chat/voice",
        files={"audio": ("recording.wav", b"wav", "audio/wav")},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "No speech"
