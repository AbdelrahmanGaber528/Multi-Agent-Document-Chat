from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.platypus import Flowable, SimpleDocTemplate

from app.core.contract import ReportStyle
from app.core.errors import ReportRenderError


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and draw total page count and footer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        # Don't draw footer on cover page (page 1) if multi-page and cover exists
        if self._pageNumber == 1 and page_count > 1:
            return

        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColorRGB(0.4, 0.4, 0.4)
        
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(self._pagesize[0] - 20 * mm, 12 * mm, footer_text)
        
        # Subtle horizontal rule above footer
        self.setStrokeColorRGB(0.85, 0.85, 0.85)
        self.setLineWidth(0.5)
        self.line(20 * mm, 16 * mm, self._pagesize[0] - 20 * mm, 16 * mm)
        
        self.restoreState()


@dataclass
class ReportDocument:
    """Mutable build context shared across all formatting tools."""

    story: list[Flowable] = field(default_factory=list)
    style: Optional[ReportStyle] = None
    output_path: Optional[str] = None
    doc: Optional[SimpleDocTemplate] = None


def make_new_document(doc_ctx: ReportDocument, output_path: str, style: ReportStyle) -> str:
    """Initialize the PDF document canvas with page size and margins."""
    doc_ctx.output_path = output_path
    doc_ctx.style = style
    doc_ctx.story = []

    page_size = A4 if style.page_size == "A4" else letter

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    doc_ctx.doc = SimpleDocTemplate(
        output_path,
        pagesize=page_size,
        leftMargin=style.left_margin_mm * mm,
        rightMargin=style.right_margin_mm * mm,
        topMargin=style.top_margin_mm * mm,
        bottomMargin=style.bottom_margin_mm * mm,
    )
    return output_path


def save_document(doc_ctx: ReportDocument) -> int:
    """Finalize and compile the PDF story to disk, returning total page count."""
    if doc_ctx.doc is None or not doc_ctx.output_path:
        raise ReportRenderError("Document has not been initialized.")

    try:
        doc_ctx.doc.build(doc_ctx.story, canvasmaker=NumberedCanvas)
    except Exception as exc:
        raise ReportRenderError(
            f"Failed to compile PDF document: {exc}",
            details={"output_path": doc_ctx.output_path, "reason": str(exc)},
        ) from exc

    # Determine page count by reading generated PDF
    try:
        import fitz  # PyMuPDF is in requirements
        doc = fitz.open(doc_ctx.output_path)
        count = len(doc)
        doc.close()
        return max(1, count)
    except Exception:
        # Fallback approximation if PyMuPDF inspect fails
        return max(1, len([f for f in doc_ctx.story if getattr(f, "__class__", None).__name__ == "PageBreak"]) + 1)
