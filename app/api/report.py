import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from app.agents.report.agent import run_report
from app.core.config import get_settings
from app.core.contract import ReportRequest, ReportResult
from app.core.errors import AppError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/report", tags=["report"])


def _generate_report(body: ReportRequest, request: Request) -> ReportResult:
    llm = getattr(request.app.state, "llm", None)
    if llm is None:
        raise RuntimeError("LLM is not initialized on the application state.")
    return run_report(body, llm)


@router.post("/generate", response_model=ReportResult)
async def generate_report_endpoint(body: ReportRequest, request: Request) -> ReportResult:
    """Generate a professional PDF report from analysis and answer findings."""
    try:
        return await run_in_threadpool(_generate_report, body, request)
    except AppError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.to_dict()) from exc
    except Exception as exc:
        logger.exception("Report generation endpoint failed")
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {exc}") from exc


@router.get("/{report_id}")
async def download_report_endpoint(report_id: str):
    """Download a generated PDF report by its report_id."""
    settings = get_settings()
    pdf_path = settings.data_dir / "reports" / f"{report_id}.pdf"

    if not pdf_path.exists() or not pdf_path.is_file():
        raise HTTPException(status_code=404, detail="Requested report PDF was not found.")

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=f"Report_{report_id}.pdf",
    )
