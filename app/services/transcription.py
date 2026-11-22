"""Local speech-to-text using faster-whisper."""

from __future__ import annotations

import logging
from io import BytesIO
from threading import Lock

from faster_whisper import WhisperModel

log = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """Raised when local speech transcription cannot produce a transcript."""


class Transcriber:
    """Lazily load and reuse a CPU multilingual Whisper model."""

    def __init__(self, model_name: str = "base") -> None:
        self.model_name = model_name
        self._model: WhisperModel | None = None
        self._model_lock = Lock()

    def _get_model(self) -> WhisperModel:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    log.info("Loading local speech model %s", self.model_name)
                    self._model = WhisperModel(
                        self.model_name, device="cpu", compute_type="int8"
                    )
        return self._model

    def transcribe(self, audio_bytes: bytes, language: str | None = None) -> str:
        if not audio_bytes:
            raise TranscriptionError("Audio recording is empty.")
        try:
            options = {"beam_size": 5, "vad_filter": True}
            if language is not None:
                options["language"] = language
            segments, _info = self._get_model().transcribe(
                BytesIO(audio_bytes), **options
            )
            transcript = " ".join(segment.text.strip() for segment in segments).strip()
        except Exception as exc:
            log.exception("Local audio transcription failed")
            raise TranscriptionError("Could not transcribe this recording.") from exc

        if not transcript:
            raise TranscriptionError("No speech was detected in the recording.")
        return transcript
