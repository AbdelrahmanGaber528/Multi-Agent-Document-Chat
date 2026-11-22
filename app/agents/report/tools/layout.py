from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Spacer

from app.agents.report.tools.document import ReportDocument


def add_page_break(doc_ctx: ReportDocument) -> str:
    """Insert a page break in the document."""
    doc_ctx.story.append(PageBreak())
    return "Page break inserted successfully."


def add_spacer(doc_ctx: ReportDocument, height_mm: float) -> str:
    """Add vertical whitespace in millimetres.

    Args:
        doc_ctx: Document context.
        height_mm: Height of the spacer in millimetres (1.0 to 100.0).
    """
    valid_height = max(1.0, min(float(height_mm), 100.0))
    doc_ctx.story.append(Spacer(1, valid_height * mm))
    return f"Spacer of {valid_height:.1f}mm added."
