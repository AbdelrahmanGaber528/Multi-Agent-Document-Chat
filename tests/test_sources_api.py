from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import sources
from app.core.contract import IngestResult


def test_image_upload_is_accepted_and_ingested(monkeypatch):
    app = FastAPI()
    app.include_router(sources.router)
    seen = {}

    def fake_ingest(filename, content):
        seen["filename"] = filename
        seen["content"] = content
        return IngestResult(doc_id="image-doc", num_chunks=2)

    monkeypatch.setattr("app.pipeline.ingestion.ingest_file", fake_ingest)

    response = TestClient(app).post(
        "/resources/upload",
        files={"file": ("receipt.png", b"image bytes", "image/png")},
    )

    assert response.status_code == 200
    assert response.json() == {"doc_id": "image-doc", "num_chunks": 2}
    assert seen == {"filename": "receipt.png", "content": b"image bytes"}


def test_unsupported_image_extension_is_rejected():
    app = FastAPI()
    app.include_router(sources.router)
    response = TestClient(app).post(
        "/resources/upload",
        files={"file": ("graphic.svg", b"<svg/>", "image/svg+xml")},
    )
    assert response.status_code == 415
