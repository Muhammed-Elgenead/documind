import uuid
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document, DocumentStatus
from app.db.session import get_session
from app.main import app
from app.services.documents import create_document, fail_interrupted_documents
from tests.conftest import TEST_USER_ID
from tests.factories import make_docx, make_pdf, page_text

# ---------- Validation (no database needed) ----------


@pytest.fixture
def no_db_session():
    session = AsyncMock(spec=AsyncSession)

    async def _get_session():
        yield session

    app.dependency_overrides[get_session] = _get_session
    return session


async def test_upload_rejects_unsupported_type(client, no_db_session, test_settings):
    resp = await client.post("/documents", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert resp.status_code == 415
    no_db_session.add.assert_not_called()
    assert not test_settings.upload_dir.exists() or not any(test_settings.upload_dir.iterdir())


async def test_upload_rejects_empty_file(client, no_db_session, test_settings):
    resp = await client.post("/documents", files={"file": ("a.pdf", b"", "application/pdf")})
    assert resp.status_code == 400
    assert not any(test_settings.upload_dir.iterdir())


async def test_upload_rejects_too_large_file(client, no_db_session, test_settings):
    big = b"x" * (test_settings.max_upload_mb * 1024 * 1024 + 1)
    resp = await client.post("/documents", files={"file": ("a.md", big, "text/markdown")})
    assert resp.status_code == 413
    assert not any(test_settings.upload_dir.iterdir())


async def test_get_document_rejects_bad_id(client, no_db_session):
    assert (await client.get("/documents/not-a-uuid")).status_code == 422


# ---------- Full pipeline against a real database ----------


async def test_upload_pdf_ingests_chunks_with_pages(db_client, session_factory, test_settings):
    n_pages = 5
    pdf = make_pdf([page_text(p) for p in range(1, n_pages + 1)])

    resp = await db_client.post(
        "/documents", files={"file": ("C:\\fakepath\\report.pdf", pdf, "application/pdf")}
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["status"] == "processing"
    assert body["filename"] == "report.pdf"
    doc_id = body["id"]
    assert (test_settings.upload_dir / f"{doc_id}.pdf").exists()

    # The ASGI test transport finishes background tasks before returning.
    detail = (await db_client.get(f"/documents/{doc_id}")).json()
    assert detail["status"] == "ready", detail
    assert detail["page_count"] == n_pages
    assert detail["error"] is None
    assert detail["chunk_count"] > n_pages  # several chunks per page at test chunk size

    async with session_factory() as s:
        chunks = (
            (await s.execute(select(Chunk).where(Chunk.document_id == uuid.UUID(doc_id))))
            .scalars()
            .all()
        )
    assert {c.page for c in chunks} == set(range(1, n_pages + 1))
    assert sorted(c.chunk_index for c in chunks) == list(range(len(chunks)))
    for c in chunks:
        # Every chunk's text comes from the page it claims.
        assert all(w.startswith(f"page{c.page}w") for w in c.text.split()), c.text[:80]
        assert len(c.embedding) == 384
        assert c.tsv  # generated full-text column is populated


async def test_upload_docx_and_markdown(db_client):
    docx = make_docx(["First page.", "Second page."])
    r1 = await db_client.post("/documents", files={"file": ("a.docx", docx, "application/x")})
    r2 = await db_client.post("/documents", files={"file": ("b.md", b"# Hi\n\nBody", "text/md")})
    d1 = (await db_client.get(f"/documents/{r1.json()['id']}")).json()
    d2 = (await db_client.get(f"/documents/{r2.json()['id']}")).json()
    assert (d1["status"], d1["page_count"], d1["chunk_count"]) == ("ready", 2, 2)
    assert (d2["status"], d2["page_count"], d2["chunk_count"]) == ("ready", 1, 1)


async def test_corrupt_pdf_is_marked_failed(db_client):
    resp = await db_client.post(
        "/documents", files={"file": ("bad.pdf", b"not really a pdf", "application/pdf")}
    )
    detail = (await db_client.get(f"/documents/{resp.json()['id']}")).json()
    assert detail["status"] == "failed"
    assert detail["error"] == "Could not process this file."
    assert detail["chunk_count"] == 0


async def test_pdf_without_text_is_marked_failed(db_client):
    resp = await db_client.post(
        "/documents", files={"file": ("scan.pdf", make_pdf(["", ""]), "application/pdf")}
    )
    detail = (await db_client.get(f"/documents/{resp.json()['id']}")).json()
    assert detail["status"] == "failed"
    assert "No extractable text" in detail["error"]
    assert detail["page_count"] == 2


async def test_list_documents_newest_first(db_client):
    for name in ("one.md", "two.md"):
        await db_client.post("/documents", files={"file": (name, b"text", "text/markdown")})
    docs = (await db_client.get("/documents")).json()
    assert [d["filename"] for d in docs] == ["two.md", "one.md"]
    assert all(d["status"] == "ready" for d in docs)


async def test_get_missing_document_404(db_client):
    assert (await db_client.get(f"/documents/{uuid.uuid4()}")).status_code == 404


async def test_fail_interrupted_documents(session_factory):
    async with session_factory() as s:
        stuck = await create_document(s, uuid.uuid4(), "stuck.pdf", TEST_USER_ID)
        done = await create_document(s, uuid.uuid4(), "done.pdf", TEST_USER_ID)
        done.status = DocumentStatus.READY
        await s.commit()
        assert await fail_interrupted_documents(s) == 1
    async with session_factory() as s:
        assert (await s.get(Document, stuck.id)).status == "failed"
        assert (await s.get(Document, done.id)).status == "ready"
