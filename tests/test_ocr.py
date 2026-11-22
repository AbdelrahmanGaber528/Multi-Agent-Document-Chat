from types import SimpleNamespace

import pytest
from PIL import Image

from app.core.errors import EmptyDocumentError, ParsingError
from app.pipeline import loader, ocr
from app.pipeline.cleaner import clean_documents
from app.pipeline.chunker import chunk_documents
from app.pipeline.parser import parse_documents


class FakeOCREngine:
    def __init__(self, result):
        self.result = result
        self.image_path = None

    def __call__(self, image_path):
        self.image_path = image_path
        return self.result


def _write_image(path):
    Image.new("RGB", (32, 32), "white").save(path)


def test_ocr_joins_detected_lines_in_engine_order(tmp_path, monkeypatch):
    image_path = tmp_path / "bilingual.png"
    _write_image(image_path)
    engine = FakeOCREngine(SimpleNamespace(txts=("English title", "مرحبا بالعالم", " ")))
    monkeypatch.setattr(ocr, "_get_engine", lambda: engine)

    assert ocr.extract_image_text(image_path) == "English title\nمرحبا بالعالم"
    assert engine.image_path == str(image_path)


def test_ocr_empty_result_raises_empty_document(tmp_path, monkeypatch):
    image_path = tmp_path / "blank.jpg"
    _write_image(image_path)
    monkeypatch.setattr(ocr, "_get_engine", lambda: FakeOCREngine(None))

    with pytest.raises(EmptyDocumentError, match="No readable text"):
        ocr.extract_image_text(image_path)


def test_ocr_rejects_invalid_image(tmp_path):
    image_path = tmp_path / "broken.png"
    image_path.write_bytes(b"not an image")

    with pytest.raises(ParsingError, match="invalid or unsafe"):
        ocr.extract_image_text(image_path)


def test_image_loader_text_flows_through_document_pipeline(tmp_path, monkeypatch):
    image_path = tmp_path / "notes.webp"
    _write_image(image_path)
    monkeypatch.setattr(loader, "SUPPORTED_EXTENSIONS", {".webp"})
    monkeypatch.setattr(loader, "extract_image_text", lambda _path: "English line\nسطر عربي")

    docs = loader.load_document(image_path, doc_id="image-doc")
    assert docs[0].metadata["file_type"] == "image"
    assert docs[0].metadata["page"] == 1

    chunks = chunk_documents(clean_documents(parse_documents(docs)))
    assert "English line" in chunks[0].page_content
    assert "سطر عربي" in chunks[0].page_content
    assert chunks[0].metadata["doc_id"] == "image-doc"
