from typing import Literal, Optional

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, Spacer

from app.agents.report.tools.document import ReportDocument


def _hex_to_color(hex_str: Optional[str], default_hex: str = "#212121") -> HexColor:
    target = hex_str.strip() if hex_str and hex_str.strip() else default_hex
    if not target.startswith("#"):
        target = f"#{target}"
    try:
        return HexColor(target)
    except Exception:
        return HexColor(default_hex)


def add_heading(
    doc_ctx: ReportDocument,
    text: str,
    level: int = 1,
    color_hex: Optional[str] = None,
) -> str:
    """Add a section heading.

    Args:
        doc_ctx: Document context.
        text: Heading text.
        level: Heading level (1 for main section, 2 for subsection, 3 for sub-subsection).
        color_hex: Optional hex color override (e.g. '#1A237E').
    """
    style = doc_ctx.style
    if level == 1:
        font_name = style.heading1_font if style else "Helvetica-Bold"
        font_size = style.heading1_size if style else 20
        default_color = style.primary_color if style else "#1A237E"
        space_before = 12
        space_after = 6
    elif level == 2:
        font_name = style.heading2_font if style else "Helvetica-Bold"
        font_size = style.heading2_size if style else 14
        default_color = style.accent_color if style else "#0D47A1"
        space_before = 10
        space_after = 4
    else:
        font_name = style.heading2_font if style else "Helvetica-Bold"
        font_size = max(10, (style.heading2_size - 2) if style else 12)
        default_color = style.body_color if style else "#212121"
        space_before = 8
        space_after = 3

    text_color = _hex_to_color(color_hex, default_color)

    p_style = ParagraphStyle(
        name=f"Heading_{level}_{len(doc_ctx.story)}",
        fontName=font_name,
        fontSize=font_size,
        leading=font_size * 1.3,
        textColor=text_color,
        spaceBefore=space_before,
        spaceAfter=space_after,
        keepWithNext=True,
    )

    clean_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    doc_ctx.story.append(Paragraph(clean_text, p_style))
    return f"Heading level {level} added: '{text}'."


def add_paragraph(
    doc_ctx: ReportDocument,
    text: str,
    font: Optional[str] = None,
    size: Optional[int] = None,
    color_hex: Optional[str] = None,
    bold: bool = False,
    italic: bool = False,
    align: Literal["left", "center", "right", "justify"] = "left",
) -> str:
    """Add a formatted body paragraph.

    Args:
        doc_ctx: Document context.
        text: Paragraph text (supports inline <b> and <i> tags).
        font: Font family (e.g. 'Helvetica' or 'Times-Roman').
        size: Font size in pt.
        color_hex: Hex text color code.
        bold: Whether to bold the entire paragraph.
        italic: Whether to italicize the entire paragraph.
        align: Text alignment ('left', 'center', 'right', 'justify').
    """
    style = doc_ctx.style
    base_font = font or (style.body_font if style else "Helvetica")
    base_size = size or (style.body_size if style else 11)
    base_color = _hex_to_color(color_hex, style.body_color if style else "#212121")
    line_spacing = style.line_spacing if style else 1.4

    alignment_map = {
        "left": TA_LEFT,
        "center": TA_CENTER,
        "right": TA_RIGHT,
        "justify": TA_JUSTIFY,
    }

    if bold and italic:
        font_name = f"{base_font}-BoldOblique" if "Helvetica" in base_font else f"{base_font}-BoldItalic"
    elif bold:
        font_name = f"{base_font}-Bold"
    elif italic:
        font_name = f"{base_font}-Oblique" if "Helvetica" in base_font else f"{base_font}-Italic"
    else:
        font_name = base_font

    p_style = ParagraphStyle(
        name=f"Para_{len(doc_ctx.story)}",
        fontName=font_name,
        fontSize=base_size,
        leading=base_size * line_spacing,
        textColor=base_color,
        alignment=alignment_map.get(align, TA_LEFT),
        spaceAfter=6,
    )

    clean_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    clean_text = clean_text.replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")
    clean_text = clean_text.replace("&lt;i&gt;", "<i>").replace("&lt;/i&gt;", "</i>")

    doc_ctx.story.append(Paragraph(clean_text, p_style))
    return f"Paragraph added ({len(text)} characters)."


def add_bullet_list(
    doc_ctx: ReportDocument,
    items: list[str],
    bullet_char: str = "•",
    indent_mm: float = 6.0,
) -> str:
    """Add a bulleted list of items.

    Args:
        doc_ctx: Document context.
        items: List of string bullet points.
        bullet_char: Symbol or bullet character (e.g. '•' or '-').
        indent_mm: Left indent in millimetres.
    """
    if not items:
        return "No items provided for bullet list."

    style = doc_ctx.style
    font_name = style.body_font if style else "Helvetica"
    font_size = style.body_size if style else 11
    text_color = _hex_to_color(None, style.body_color if style else "#212121")
    line_spacing = style.line_spacing if style else 1.4

    p_style = ParagraphStyle(
        name=f"Bullet_{len(doc_ctx.story)}",
        fontName=font_name,
        fontSize=font_size,
        leading=font_size * line_spacing,
        textColor=text_color,
        leftIndent=indent_mm * mm,
        firstLineIndent=-4 * mm,
        spaceAfter=3,
    )

    for item in items:
        clean_item = item.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        doc_ctx.story.append(Paragraph(f"{bullet_char}&nbsp;&nbsp;{clean_item}", p_style))

    doc_ctx.story.append(Spacer(1, 4 * mm))
    return f"Bullet list with {len(items)} items added."
