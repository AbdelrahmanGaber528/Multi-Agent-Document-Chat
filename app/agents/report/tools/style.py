from typing import Literal

from app.agents.report.tools.document import ReportDocument


def set_font(
    doc_ctx: ReportDocument,
    element: Literal["heading1", "heading2", "body"],
    font_name: str,
    size: int,
) -> str:
    """Dynamically customize font family and size for subsequent elements.

    Args:
        doc_ctx: Document context.
        element: Which document element to change ('heading1', 'heading2', or 'body').
        font_name: Standard PDF font ('Helvetica', 'Helvetica-Bold', 'Times-Roman', 'Times-Bold', 'Courier').
        size: Font point size (6 to 72).
    """
    if not doc_ctx.style:
        return "Style not initialized."

    valid_size = max(6, min(size, 72))

    if element == "heading1":
        doc_ctx.style.heading1_font = font_name
        doc_ctx.style.heading1_size = valid_size
    elif element == "heading2":
        doc_ctx.style.heading2_font = font_name
        doc_ctx.style.heading2_size = valid_size
    elif element == "body":
        doc_ctx.style.body_font = font_name
        doc_ctx.style.body_size = valid_size

    return f"Font for {element} set to {font_name} ({valid_size}pt)."


def set_color_theme(
    doc_ctx: ReportDocument,
    primary_hex: str,
    accent_hex: str,
) -> str:
    """Update primary and accent brand colors for headings and banners.

    Args:
        doc_ctx: Document context.
        primary_hex: Hex code for primary color (e.g. '#1A237E').
        accent_hex: Hex code for accent color (e.g. '#0D47A1').
    """
    if not doc_ctx.style:
        return "Style not initialized."

    p = primary_hex if primary_hex.startswith("#") else f"#{primary_hex}"
    a = accent_hex if accent_hex.startswith("#") else f"#{accent_hex}"

    doc_ctx.style.primary_color = p
    doc_ctx.style.accent_color = a
    return f"Color theme updated to primary={p}, accent={a}."
