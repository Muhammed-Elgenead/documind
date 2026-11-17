import uuid
from datetime import datetime
from pathlib import Path, PurePath

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, status
from pydantic import BaseModel, ConfigDict

from app.api.deps import (
    CurrentUserDep,
    EmbedderDep,
    SessionDep,
    SessionFactoryDep,
    SettingsDep,
)
from app.services import documents as documents_service
from app.services.extraction import SUPPORTED_EXTENSIONS
from app.services.ingestion import ingest_document
from app.services.storage import EmptyFile, FileTooLarge, delete_file, save_upload

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    status: str
    page_count: int | None
    error: str | None
    created_at: datetime


class DocumentDetail(DocumentOut):
    chunk_count: int


def _safe_filename(name: str | None) -> str:
    # Keep only the base name; browsers may send full paths. Truncate to fit the column.
    base = PurePath((name or "").replace("\\", "/")).name.strip()
    return (base or "upload")[:255]


@router.post("", response_model=DocumentOut, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile,
    user: CurrentUserDep,
    background_tasks: BackgroundTasks,
    session: SessionDep,
    session_factory: SessionFactoryDep,
    settings: SettingsDep,
    embedder: EmbedderDep,
) -> DocumentOut:
    filename = _safe_filename(file.filename)
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported file type '{ext or '(none)'}'. Upload PDF, DOCX or Markdown.",
        )

    document_id = uuid.uuid4()
    path = settings.upload_dir / f"{document_id}{ext}"
    try:
        await save_upload(file, path, max_bytes=settings.max_upload_mb * 1024 * 1024)
    except FileTooLarge as exc:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE, f"File exceeds {settings.max_upload_mb} MB."
        ) from exc
    except EmptyFile as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "File is empty.") from exc

    try:
        doc = await documents_service.create_document(session, document_id, filename, user.id)
    except BaseException:
        await delete_file(path)
        raise

    background_tasks.add_task(
        ingest_document,
        document_id,
        path,
        session_factory,
        embedder,
        settings.chunk_size_tokens,
        settings.chunk_overlap_tokens,
    )
    return DocumentOut.model_validate(doc)


@router.get("", response_model=list[DocumentOut])
async def list_documents(session: SessionDep, user: CurrentUserDep) -> list[DocumentOut]:
    docs = await documents_service.list_documents(session, user.id)
    return [DocumentOut.model_validate(d) for d in docs]


@router.get("/{document_id}", response_model=DocumentDetail)
async def get_document(
    document_id: uuid.UUID, session: SessionDep, user: CurrentUserDep
) -> DocumentDetail:
    doc = await documents_service.get_document(session, document_id, user.id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found.")
    chunk_count = await documents_service.count_chunks(session, document_id)
    return DocumentDetail(**DocumentOut.model_validate(doc).model_dump(), chunk_count=chunk_count)
