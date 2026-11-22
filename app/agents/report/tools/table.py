from typing import Optional

from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

from app.agents.report.tools.document import ReportDocument
from app.agents.report.tools.text import _hex_to_color


def add_table(
    doc_ctx: ReportDocument,
    headers: list[str],
    rows: list[list[str]],
    col_widths: Optional[list[float]] = None,
    header_bg_hex: Optional[str] = None,
    stripe_rows: bool = True,
    caption: Optional[str] = None,
) -> str:
    """Render a styled data table with headers, borders, and alternating row colors.

    Args:
        doc_ctx: Document context.
        headers: Column header strings.
        rows: 2D list of row values (strings).
        col_widths: Optional list of column widths in mm.
        header_bg_hex: Background color for header row.
        stripe_rows: If True, alternating rows have a light grey background.
        caption: Optional table title/caption displayed above the table.
    """
    if not headers and not rows:
        return "Cannot create empty table."

    style = doc_ctx.style
    primary_color = _hex_to_color(header_bg_hex, style.table_header_bg if style else "#E3F2FD")
    header_text_color = _hex_to_color(style.primary_color if style else "#1A237E")
    body_text_color = _hex_to_color(style.body_color if style else "#212121")
    font_name = style.body_font if style else "Helvetica"
    font_size = max(8, (style.body_size - 1) if style else 10)

    if caption:
        caption_style = ParagraphStyle(
            name=f"TableCaption_{len(doc_ctx.story)}",
            fontName=f"{font_name}-Bold",
            fontSize=font_size + 1,
            leading=(font_size + 1) * 1.3,
            textColor=header_text_color,
            spaceAfter=4,
            keepWithNext=True,
        )
        doc_ctx.story.append(Paragraph(caption, caption_style))

    header_cell_style = ParagraphStyle(
        name=f"TH_{len(doc_ctx.story)}",
        fontName=f"{font_name}-Bold",
        fontSize=font_size,
        leading=font_size * 1.2,
        textColor=header_text_color,
    )
    body_cell_style = ParagraphStyle(
        name=f"TD_{len(doc_ctx.story)}",
        fontName=font_name,
        fontSize=font_size,
        leading=font_size * 1.2,
        textColor=body_text_color,
    )

    table_data = []
    if headers:
        header_cells = [
            Paragraph(h.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), header_cell_style)
            for h in headers
        ]
        table_data.append(header_cells)

    for row in rows:
        row_cells = [
            Paragraph(str(val).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), body_cell_style)
            for val in row
        ]
        table_data.append(row_cells)

    widths = [w * mm for w in col_widths] if col_widths else None

    table_style_commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, 0), primary_color),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, 0), 1.2, header_text_color),
        ("LINEBELOW", (0, -1), (-1, -1), 1.0, HexColor("#B0BEC5")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, HexColor("#ECEFF1")),
        ("BOX", (0, 0), (-1, -1), 0.5, HexColor("#CFD8DC")),
    ]

    if stripe_rows:
        num_rows = len(table_data)
        for r in range(1, num_rows):
            if r % 2 == 1:
                table_style_commands.append(("BACKGROUND", (0, r), (-1, r), HexColor("#F8F9FA")))

    t = Table(table_data, colWidths=widths, style=TableStyle(table_style_commands))
    doc_ctx.story.append(t)
    doc_ctx.story.append(Spacer(1, 6 * mm))
    return f"Table added with {len(headers)} columns and {len(rows)} data rows."
