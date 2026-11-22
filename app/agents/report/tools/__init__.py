from typing import Literal, Optional

from langchain_core.tools import StructuredTool

from app.agents.report.tools import cover, layout, style, table, text
from app.agents.report.tools.document import ReportDocument


def build_report_tools(doc_ctx: ReportDocument) -> list[StructuredTool]:
    """Wrap ReportLab tools into LangChain StructuredTools bound to doc_ctx."""

    def add_page_break() -> str:
        """Insert a page break to move content to the next page."""
        return layout.add_page_break(doc_ctx)

    def add_spacer(height_mm: float) -> str:
        """Add vertical whitespace between sections.

        Args:
            height_mm: Height of whitespace in millimetres (typically 5 to 25 mm).
        """
        return layout.add_spacer(doc_ctx, height_mm)

    def add_heading(text_content: str, level: int = 1, color_hex: Optional[str] = None) -> str:
        """Add a section heading.

        Args:
            text_content: The text of the heading.
            level: Heading level (1 for main sections, 2 for subsections, 3 for minor headings).
            color_hex: Optional hex color override (e.g. '#1A237E').
        """
        return text.add_heading(doc_ctx, text_content, level, color_hex)

    def add_paragraph(
        text_content: str,
        font: Optional[str] = None,
        size: Optional[int] = None,
        color_hex: Optional[str] = None,
        bold: bool = False,
        italic: bool = False,
        align: Literal["left", "center", "right", "justify"] = "left",
    ) -> str:
        """Add a body paragraph of text.

        Args:
            text_content: Content of the paragraph.
            font: Optional font name ('Helvetica', 'Times-Roman', 'Courier').
            size: Font size in points (e.g. 10, 11, 12).
            color_hex: Optional hex color (e.g. '#212121').
            bold: True to make entire paragraph bold.
            italic: True to italicize paragraph.
            align: Text alignment ('left', 'center', 'right', 'justify').
        """
        return text.add_paragraph(doc_ctx, text_content, font, size, color_hex, bold, italic, align)

    def add_bullet_list(items: list[str], bullet_char: str = "•", indent_mm: float = 6.0) -> str:
        """Add a bulleted list of items.

        Args:
            items: List of strings for each bullet item.
            bullet_char: Character to use for the bullet (e.g. '•', '-').
            indent_mm: Indentation from left margin in millimetres.
        """
        return text.add_bullet_list(doc_ctx, items, bullet_char, indent_mm)

    def add_table(
        headers: list[str],
        rows: list[list[str]],
        col_widths: Optional[list[float]] = None,
        header_bg_hex: Optional[str] = None,
        stripe_rows: bool = True,
        caption: Optional[str] = None,
    ) -> str:
        """Render a styled table with borders, header styling, and alternating row colors.

        Args:
            headers: List of column header names.
            rows: 2D list of row cells (strings).
            col_widths: Column widths in millimetres (e.g. [40.0, 70.0, 60.0]).
            header_bg_hex: Background color for header row (e.g. '#E3F2FD').
            stripe_rows: Whether to alternate row background colors.
            caption: Optional title displayed immediately above the table.
        """
        return table.add_table(doc_ctx, headers, rows, col_widths, header_bg_hex, stripe_rows, caption)

    def set_font(
        element: Literal["heading1", "heading2", "body"],
        font_name: str,
        size: int,
    ) -> str:
        """Change font family and size for subsequent document elements.

        Args:
            element: Which element type to customize ('heading1', 'heading2', or 'body').
            font_name: Standard font name ('Helvetica', 'Helvetica-Bold', 'Times-Roman', 'Courier').
            size: Size in pt.
        """
        return style.set_font(doc_ctx, element, font_name, size)

    def set_color_theme(primary_hex: str, accent_hex: str) -> str:
        """Set report color theme palette.

        Args:
            primary_hex: Hex code for primary brand color (e.g. '#1A237E').
            accent_hex: Hex code for accent color (e.g. '#0D47A1').
        """
        return style.set_color_theme(doc_ctx, primary_hex, accent_hex)

    def add_cover_page(
        title: str,
        subtitle: Optional[str] = None,
        report_date: Optional[str] = None,
        accent_color_hex: Optional[str] = None,
    ) -> str:
        """Add a professional title page with brand banner and metadata.

        Args:
            title: Main title of the report.
            subtitle: Optional subtitle.
            report_date: Optional date string.
            accent_color_hex: Optional hex color for the top banner bar.
        """
        return cover.add_cover_page(doc_ctx, title, subtitle, report_date, accent_color_hex)

    funcs = [
        add_cover_page,
        add_heading,
        add_paragraph,
        add_bullet_list,
        add_table,
        add_spacer,
        add_page_break,
        set_font,
        set_color_theme,
    ]
    return [StructuredTool.from_function(func=fn) for fn in funcs]
