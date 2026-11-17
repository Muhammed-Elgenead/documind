import pytest

from app.services.extraction import UnsupportedFileType, clean_text, extract_pages
from tests.factories import make_docx, make_pdf


def test_pdf_pages_keep_their_numbers(tmp_path):
    path = tmp_path / "a.pdf"
    path.write_bytes(make_pdf(["Alpha page one.", "", "Gamma page three."]))
    pages = extract_pages(path)
    assert [p.page for p in pages] == [1, 2, 3]
    assert "Alpha" in pages[0].text
    assert pages[1].text == ""
    assert "Gamma" in pages[2].text


def test_docx_splits_on_page_breaks(tmp_path):
    path = tmp_path / "a.docx"
    path.write_bytes(make_docx(["First page text.", "Second page text.", "Third."]))
    pages = extract_pages(path)
    assert [p.text for p in pages] == ["First page text.", "Second page text.", "Third."]


def test_docx_tables_are_included(tmp_path):
    import io

    from docx import Document as DocxDocument

    doc = DocxDocument()
    doc.add_paragraph("Before table.")
    table = doc.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Cell A"
    table.cell(0, 1).text = "Cell B"
    buf = io.BytesIO()
    doc.save(buf)
    path = tmp_path / "t.docx"
    path.write_bytes(buf.getvalue())

    (page,) = extract_pages(path)
    assert "Before table." in page.text
    assert "Cell A" in page.text and "Cell B" in page.text


def test_markdown_is_single_page(tmp_path):
    path = tmp_path / "notes.MD"
    path.write_text("# Title\n\nSome body.\n", encoding="utf-8")
    pages = extract_pages(path)
    assert len(pages) == 1
    assert pages[0].page == 1
    assert pages[0].text == "# Title\n\nSome body."


def test_unsupported_type_raises(tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("hi")
    with pytest.raises(UnsupportedFileType):
        extract_pages(path)


def test_clean_text_strips_nul_and_extra_blank_lines():
    assert clean_text("a\x00b\r\n\r\n\r\n\r\nc  \n") == "ab\n\nc"
