import uuid
from collections.abc import Sequence

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document, DocumentStatus


async def create_document(
    session: AsyncSession, document_id: uuid.UUID, filename: str, user_id: uuid.UUID
) -> Document:
    doc = Document(
        id=document_id, filename=filename, status=DocumentStatus.PROCESSING, user_id=user_id
    )
    session.add(doc)
    await session.commit()
    await session.refresh(doc)
    return doc


async def list_documents(session: AsyncSession, user_id: uuid.UUID) -> Sequence[Document]:
    result = await session.execute(
        select(Document).where(Document.user_id == user_id).order_by(Document.created_at.desc())
    )
    return result.scalars().all()


async def get_document(
    session: AsyncSession, document_id: uuid.UUID, user_id: uuid.UUID
) -> Document | None:
    """The document if it exists and belongs to the user (never someone else's)."""
    doc = await session.get(Document, document_id)
    return doc if doc is not None and doc.user_id == user_id else None


async def count_chunks(session: AsyncSession, document_id: uuid.UUID) -> int:
    result = await session.execute(
        select(func.count()).select_from(Chunk).where(Chunk.document_id == document_id)
    )
    return result.scalar_one()


async def fail_interrupted_documents(session: AsyncSession) -> int:
    """Mark documents left in 'processing' (e.g. by a restart) as failed. Returns count."""
    result = await session.execute(
        update(Document)
        .where(Document.status == DocumentStatus.PROCESSING)
        .values(status=DocumentStatus.FAILED, error="Processing was interrupted; please re-upload.")
    )
    await session.commit()
    return result.rowcount
