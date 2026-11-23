from types import SimpleNamespace

import pytest

from app.services.transcription import Transcriber, TranscriptionError


class FakeModel:
    def __init__(self, segments):
        self.segments = segments

    def transcribe(self, audio, **kwargs):
        assert audio.read() == b"wav data"
        assert kwargs.get("language") is None
        return iter(self.segments), SimpleNamespace()


def test_transcriber_returns_joined_speech(monkeypatch):
    transcriber = Transcriber()
    monkeypatch.setattr(
        transcriber,
        "_get_model",
        lambda: FakeModel([SimpleNamespace(text=" Hello "), SimpleNamespace(text="world.")]),
    )

    assert transcriber.transcribe(b"wav data") == "Hello world."


def test_transcriber_rejects_empty_audio():
    with pytest.raises(TranscriptionError, match="empty"):
        Transcriber().transcribe(b"")


def test_transcriber_reports_no_speech(monkeypatch):
    transcriber = Transcriber()
    monkeypatch.setattr(transcriber, "_get_model", lambda: FakeModel([]))

    with pytest.raises(TranscriptionError, match="No speech"):
        transcriber.transcribe(b"wav data")


def test_transcriber_wraps_model_errors(monkeypatch):
    transcriber = Transcriber()

    def fail():
        raise RuntimeError("model failed")

    monkeypatch.setattr(transcriber, "_get_model", fail)
    with pytest.raises(TranscriptionError, match="Could not transcribe"):
        transcriber.transcribe(b"wav data")
