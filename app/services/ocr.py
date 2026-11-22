"""Local image text recognition with RapidOCR service."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from threading import Lock

from PIL import Image, UnidentifiedImageError

from app.core.config import get_settings
from app.core.errors import EmptyDocumentError, ParsingError

log = logging.getLogger(__name__)
_engine = None
_engine_lock = Lock()


def _get_engine():
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                try:
                    from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

                    settings = get_settings()
                    settings.ocr_model_dir.mkdir(parents=True, exist_ok=True)
                    _engine = RapidOCR(
                        params={
                            "Global.model_root_dir": str(settings.ocr_model_dir),
                            "Rec.ocr_version": getattr(OCRVersion, "PPOCRV5", "PP-OCRv5"),
                            "Rec.lang_type": getattr(LangRec, "ARABIC", "arabic"),
                            "Rec.model_type": getattr(ModelType, "MOBILE", "mobile"),
                        }
                    )
                except Exception as exc:
                    log.exception("Could not initialize RapidOCR")
                    raise ParsingError(
                        "OCR could not be initialized.", details={"reason": str(exc)}
                    ) from exc
    return _engine


def recognized_lines(result) -> list[str]:
    """Return non-empty OCR lines in the order reported by RapidOCR."""
    if result is None:
        return []
    return [line.strip() for line in (getattr(result, "txts", None) or []) if line.strip()]


def extract_image_text(image_path: str | Path) -> str:
    """Validate a raster image and return OCR text from its detected lines."""
    image_path = Path(image_path)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(image_path) as image:
                image.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ParsingError(
            "Uploaded image is invalid or unsafe.",
            details={"filename": image_path.name},
        ) from exc

    try:
        result = _get_engine()(str(image_path))
        text = "\n".join(recognized_lines(result))
    except ParsingError:
        raise
    except Exception as exc:
        log.exception("OCR failed for %s", image_path.name)
        raise ParsingError(
            "Failed to extract text from image.",
            details={"filename": image_path.name, "reason": str(exc)},
        ) from exc

    if not text.strip():
        raise EmptyDocumentError(
            "No readable text found in image.", details={"filename": image_path.name}
        )
    return text
