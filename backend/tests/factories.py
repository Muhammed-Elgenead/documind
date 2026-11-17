"""Build small test documents in memory."""

import io

from docx import Document as DocxDocument
from docx.enum.text import WD_BREAK
from fpdf import FPDF


def make_pdf(pages: list[str]) -> bytes:
    pdf = FPDF()
    pdf.set_font("Helvetica", size=11)
    for text in pages:
        pdf.add_page()
        if text:
            pdf.multi_cell(0, 6, text)
    return bytes(pdf.output())


def make_docx(pages: list[str]) -> bytes:
    """One paragraph per page, separated by hard page breaks."""
    doc = DocxDocument()
    for i, text in enumerate(pages):
        para = doc.add_paragraph(text)
        if i < len(pages) - 1:
            para.add_run().add_break(WD_BREAK.PAGE)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def page_text(page: int, words: int = 300) -> str:
    """Distinct, searchable text for a page: 'page7 word0 ... '."""
    return " ".join(f"page{page}w{i}" for i in range(words))
