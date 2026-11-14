import logging
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from app.core.config import get_settings
from app.pipeline.vector_db import VectorDatabase

log = logging.getLogger(__name__)

router = APIRouter(prefix="/resources", tags=["resources"])

settings = get_settings()


def get_vector_db():
    return VectorDatabase(
        embedding_model=settings.embedding_model,
    )


@router.post("/upload")
async def upload_resource(file: UploadFile = File(...)):
    
    if not file.filename:
        raise HTTPException(400, "Filename is required.")

    extension = "." + file.filename.rsplit(".", 1)[-1].lower()

    if extension not in settings.allowed_extensions:
        raise HTTPException(415, "Unsupported file type.")

    content = await file.read()

    if not content:
        raise HTTPException(400, "Uploaded file is empty.")

    if len(content) > settings.max_upload_bytes:
        raise HTTPException(413, "File is too large.")

    from app.pipeline.ingestion import ingest_file

    result = await run_in_threadpool(
        ingest_file,
        file.filename,
        content,
    )

    return result


@router.get("")
def list_resources():
    db = get_vector_db()

    try:
        stored = db.vector_store.get(
            include=["metadatas"]
        )

        resources = {}

        for metadata in stored.get("metadatas") or []:
            metadata = metadata or {}
            doc_id = metadata.get("doc_id")

            if not doc_id:
                continue

            if doc_id not in resources:
                resources[doc_id] = {
                    "doc_id": doc_id,
                    "filename": metadata.get("filename"),
                    "chunks": 0,
                }

            resources[doc_id]["chunks"] += 1

        return {
            "resources": list(resources.values()),
            "total": len(resources),
        }

    except Exception as exc:
        log.exception("Failed to list resources")
        raise HTTPException(
            500,
            "Failed to retrieve resources.",
        ) from exc


@router.delete("/{resource_id}")
def delete_resource(resource_id: str):
    
    db = get_vector_db()

    chunks = db.get_document_chunks(resource_id)

    if not chunks:
        raise HTTPException(
            404,
            "Resource not found.",
        )

    db.delete_document(resource_id)

    return {
        "message": "Resource deleted successfully.",
        "resource_id": resource_id,
    }