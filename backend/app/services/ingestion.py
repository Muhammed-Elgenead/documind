"""Background ingestion: extract -> chunk -> embed -> store."""

import asyncio
import logging
import uuid
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import EMBEDDING_DIM, Chunk, Document, DocumentStatus
from app.services.chunking import chunk_pages
from app.services.embeddings import Embedder
from app.services.extraction import extract_pages

logger = logging.getLogger(__name__)


class IngestionError(Exception):
    """An expected failure whose message is safe to show the user."""


async def ingest_document(
    document_id: uuid.UUID,
    path: Path,
    session_factory: async_sessionmaker[AsyncSession],
    embedder: Embedder,
    chunk_size: int = 500,
    overlap: int = 50,
) -> None:
    """Process an uploaded file. Never raises: failures are recorded on the document."""
    page_count: int | None = None
    try:
        # Parsing and embedding are CPU-bound; keep them off the event loop.
        pages = await asyncio.to_thread(extract_pages, path)
        page_count = len(pages)
        chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            raise IngestionError("No extractable text found (scanned or image-only file?).")

        vectors = await asyncio.to_thread(embedder.embed_documents, [c.text for c in chunks])
        if len(vectors) != len(chunks) or any(len(v) != EMBEDDING_DIM for v in vectors):
            raise IngestionError("Embedding provider returned vectors of the wrong shape.")

        async with session_factory() as session:
            doc = await session.get(Document, document_id)
            if doc is None:  # deleted while processing
                return
            session.add_all(
                Chunk(
                    document_id=document_id,
                    page=c.page,
                    chunk_index=c.chunk_index,
                    text=c.text,
                    embedding=v,
                )
                for c, v in zip(chunks, vectors, strict=True)
            )
            doc.page_count = page_count
            doc.status = DocumentStatus.READY
            doc.error = None
            await session.commit()
        logger.info("Ingested %s: %d pages, %d chunks", document_id, page_count, len(chunks))
    except Exception as exc:
        logger.exception("Ingestion failed for %s", document_id)
        message = str(exc) if isinstance(exc, IngestionError) else "Could not process this file."
        await _mark_failed(session_factory, document_id, message, page_count)


async def _mark_failed(
    session_factory: async_sessionmaker[AsyncSession],
    document_id: uuid.UUID,
    message: str,
    page_count: int | None,
) -> None:
    try:
        async with session_factory() as session:
            doc = await session.get(Document, document_id)
            if doc is not None:
                doc.status = DocumentStatus.FAILED
                doc.error = message
                doc.page_count = page_count
                await session.commit()
    except Exception:
        logger.exception("Could not mark document %s as failed", document_id)
