from datetime import date
from typing import Optional

from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, Spacer, Table, TableStyle

from app.agents.report.tools.document import ReportDocument
from app.agents.report.tools.text import _hex_to_color


def add_cover_page(
    doc_ctx: ReportDocument,
    title: str,
    subtitle: Optional[str] = None,
    report_date: Optional[str] = None,
    accent_color_hex: Optional[str] = None,
) -> str:
    """Create a full professional cover page with brand banner, title, metadata, and page break.

    Args:
        doc_ctx: Document context.
        title: Main report title.
        subtitle: Optional subtitle or short description.
        report_date: Optional date string (defaults to today's date).
        accent_color_hex: Optional banner background color override.
    """
    style = doc_ctx.style
    primary_color = _hex_to_color(accent_color_hex, style.primary_color if style else "#1A237E")
    body_color = _hex_to_color(None, style.body_color if style else "#212121")

    # Space at top of cover page
    doc_ctx.story.append(Spacer(1, 40 * mm))

    # Top brand bar
    bar_table = Table([[""]], colWidths=[170 * mm], rowHeights=[6 * mm])
    bar_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), primary_color),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
        ])
    )
    doc_ctx.story.append(bar_table)
    doc_ctx.story.append(Spacer(1, 15 * mm))

    # Title
    title_style = ParagraphStyle(
        name=f"CoverTitle_{len(doc_ctx.story)}",
        fontName=style.heading1_font if style else "Helvetica-Bold",
        fontSize=28,
        leading=34,
        textColor=primary_color,
        spaceAfter=12,
    )
    clean_title = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    doc_ctx.story.append(Paragraph(clean_title, title_style))

    # Subtitle
    if subtitle:
        sub_style = ParagraphStyle(
            name=f"CoverSub_{len(doc_ctx.story)}",
            fontName=style.body_font if style else "Helvetica",
            fontSize=14,
            leading=18,
            textColor=HexColor("#546E7A"),
            spaceAfter=30,
        )
        clean_sub = subtitle.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        doc_ctx.story.append(Paragraph(clean_sub, sub_style))
    else:
        doc_ctx.story.append(Spacer(1, 20 * mm))

    # Decorative divider
    divider = Table([[""]], colWidths=[170 * mm], rowHeights=[1 * mm])
    divider.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), HexColor("#E0E0E0")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
        ])
    )
    doc_ctx.story.append(divider)
    doc_ctx.story.append(Spacer(1, 40 * mm))

    # Metadata block
    effective_date = report_date or date.today().strftime("%B %d, %Y")
    meta_style = ParagraphStyle(
        name=f"CoverMeta_{len(doc_ctx.story)}",
        fontName=style.body_font if style else "Helvetica",
        fontSize=10,
        leading=14,
        textColor=body_color,
    )
    doc_ctx.story.append(Paragraph(f"<b>Generated on:</b> {effective_date}", meta_style))
    doc_ctx.story.append(Paragraph("<b>System:</b> Multi-Agent Document Chat Analysis", meta_style))

    # Break to next page for body content
    doc_ctx.story.append(PageBreak())
    return f"Cover page created for '{title}'."
