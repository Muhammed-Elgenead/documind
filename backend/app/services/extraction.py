"""Extract text from uploaded files, page by page.

Page numbers are 1-based and are what citations will point at:
- PDF: real pages.
- DOCX: Word has no fixed pages, so we split on explicit page breaks and on the
  "last rendered page break" markers Word saves. Close to what the user saw in Word,
  but not guaranteed.
- Markdown: no pages; everything is page 1.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from docx import Document as DocxDocument
from docx.oxml.ns import qn
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = frozenset({".pdf", ".docx", ".md", ".markdown"})


class UnsupportedFileType(ValueError):
    pass


@dataclass(frozen=True)
class PageText:
    page: int
    text: str


def clean_text(text: str) -> str:
    # Postgres TEXT cannot store NUL; PDFs sometimes contain them.
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(path: Path) -> list[PageText]:
    """Return one PageText per page (including empty pages, so page numbers stay true)."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        raw = _pdf_pages(path)
    elif suffix == ".docx":
        raw = _docx_pages(path)
    elif suffix in {".md", ".markdown"}:
        raw = [path.read_text(encoding="utf-8", errors="replace")]
    else:
        raise UnsupportedFileType(f"Unsupported file type: {suffix or '(none)'}")
    return [PageText(page=i, text=clean_text(t)) for i, t in enumerate(raw, start=1)]


def _pdf_pages(path: Path) -> list[str]:
    reader = PdfReader(path)
    return [page.extract_text() or "" for page in reader.pages]


_W_P = qn("w:p")
_W_T = qn("w:t")
_W_TAB = qn("w:tab")
_W_BR = qn("w:br")
_W_TYPE = qn("w:type")
_W_LAST_RENDERED_BREAK = qn("w:lastRenderedPageBreak")
_W_PAGE_BREAK_BEFORE = qn("w:pageBreakBefore")


def _docx_pages(path: Path) -> list[str]:
    body = DocxDocument(str(path)).element.body
    pages: list[list[str]] = [[]]
    # After a hard break, Word usually also writes a rendered-break marker at the
    # start of the next page; ignore it so we don't count the same break twice.
    just_hard_broke = False

    def new_page(hard: bool) -> None:
        nonlocal just_hard_broke
        if not hard and just_hard_broke:
            just_hard_broke = False
            return
        pages.append([])
        just_hard_broke = hard

    for para in body.iter(_W_P):
        ppr = para.pPr
        if ppr is not None and ppr.find(_W_PAGE_BREAK_BEFORE) is not None and any(pages[-1]):
            new_page(hard=True)
        for el in para.iter(_W_T, _W_TAB, _W_BR, _W_LAST_RENDERED_BREAK):
            # Skip content of paragraphs nested inside this one (e.g. text boxes);
            # body.iter(_W_P) visits those separately.
            if _nearest_paragraph(el) is not para:
                continue
            if el.tag == _W_T:
                if el.text:
                    pages[-1].append(el.text)
                    just_hard_broke = False
            elif el.tag == _W_TAB:
                pages[-1].append("\t")
            elif el.tag == _W_BR:
                if el.get(_W_TYPE) == "page":
                    new_page(hard=True)
                else:
                    pages[-1].append("\n")
            elif el.tag == _W_LAST_RENDERED_BREAK and any(pages[-1]):
                new_page(hard=False)
        pages[-1].append("\n")

    return ["".join(parts) for parts in pages]


def _nearest_paragraph(el):
    node = el.getparent()
    while node is not None and node.tag != _W_P:
        node = node.getparent()
    return node
