import logging

from fastapi import APIRouter, File, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.core.contract import (
    IngestResult,
    IngestUrlRequest,
    IngestWikipediaRequest,
    SourceListResponse,
)
from app.core.config import settings
from app.core.errors import (
    FileTooLarge,
    IngestionFailed,
    InvalidRequest,
    InvalidURL,
    SourceNotFound,
)
from app.ingestion.pipeline import ingest_file, ingest_url, ingest_wikipedia
from app.knowledge.vector_store import delete_source, list_sources, source_exists

log = logging.getLogger(__name__)

router = APIRouter(prefix="/sources", tags=["sources"])


def _checked(result: IngestResult) -> IngestResult:
    """Convert failed ingestion results into HTTP errors."""

    if result.status == "failed":
        raise IngestionFailed(result.message or "Ingestion failed")

    return result


@router.post("/files", response_model=IngestResult)
async def upload_file(file: UploadFile = File(...)) -> IngestResult:

    if not file.filename:
        raise InvalidRequest("Filename is missing")

    filename = file.filename

    data = await file.read()

    if len(data) > settings.max_upload_bytes:
        raise FileTooLarge(
            f"File exceeds the {settings.max_upload_mb} MB limit"
        )

    if not data:
        raise InvalidRequest("Uploaded file is empty")

    log.info("Upload received: %s (%d bytes)", filename, len(data))

    result = await run_in_threadpool(
        ingest_file,
        filename,
        data,
    )

    return _checked(result)



@router.post("/url", response_model=IngestResult)
def add_url(body: IngestUrlRequest) -> IngestResult:

    url = body.url.strip()

    if not url:
        raise InvalidURL("URL is empty")

    log.info("URL ingestion requested")

    return _checked(ingest_url(url))



@router.post("/wikipedia", response_model=IngestResult)
def add_wikipedia(body: IngestWikipediaRequest) -> IngestResult:

    page = body.page.strip()

    if not page:
        raise InvalidRequest(
            "Wikipedia page title or URL is empty"
        )

    log.info(
        "Wikipedia ingestion requested: %s (%s)",
        page,
        body.lang,
    )

    return _checked(ingest_wikipedia(page, body.lang))



@router.get("", response_model=SourceListResponse)
def get_sources() -> SourceListResponse:

    return SourceListResponse(
        sources=list_sources()
    )



@router.delete("/{source_id}", response_model=SourceListResponse)
def remove_source(source_id: str) -> SourceListResponse:

    if not source_exists(source_id):
        raise SourceNotFound(f"No source with id {source_id}")

    deleted = delete_source(source_id)

    if not deleted:
        raise IngestionFailed("Failed to delete source")

    return SourceListResponse(
        sources=list_sources()
    )