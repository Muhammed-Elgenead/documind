"""POST /chat: retrieval-augmented answers streamed as Server-Sent Events.

Event stream (each `data:` is JSON):
  event: start   {"conversation_id", "sources": <number of retrieved chunks>}
  event: token   {"text": "..."}                      (repeated)
  event: done    {"conversation_id", "message_id", "answer", "citations": [...]}
  event: error   {"message": "..."}                   (instead of done, on failure)
"""

import json
import logging
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agents.citations import build_citations
from app.agents.llm import LLM, LLMError, get_llm
from app.agents.prompts import NOT_FOUND_MESSAGE, SYSTEM_PROMPT, build_messages
from app.api.deps import (
    CurrentUserDep,
    EmbedderDep,
    RerankerDep,
    SessionDep,
    SessionFactoryDep,
    SettingsDep,
    limit_chat,
)
from app.db.models import MessageRole
from app.services import chat as chat_service
from app.services.retrieval import RetrievalOptions, RetrievedChunk, retrieve

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

LLMDep = Annotated[LLM, Depends(get_llm)]


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    conversation_id: uuid.UUID | None = None
    # Restrict retrieval to these documents (default: all ready documents).
    document_ids: list[uuid.UUID] | None = None


class Citation(BaseModel):
    number: int
    document_id: uuid.UUID
    filename: str
    page: int
    chunk_id: int
    text: str


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    citations: list[Citation] | None
    created_at: datetime


class ConversationOut(BaseModel):
    id: uuid.UUID
    title: str
    created_at: datetime
    messages: list[MessageOut]


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat", dependencies=[Depends(limit_chat)])
async def chat(
    req: ChatRequest,
    user: CurrentUserDep,
    session: SessionDep,
    session_factory: SessionFactoryDep,
    settings: SettingsDep,
    embedder: EmbedderDep,
    reranker: RerankerDep,
    llm: LLMDep,
) -> StreamingResponse:
    question = req.question.strip()
    if not question:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Question is empty.")

    # Everything that can fail with a proper HTTP status happens before streaming starts.
    if req.conversation_id is not None:
        conv = await chat_service.get_conversation(session, req.conversation_id, user.id)
        if conv is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found.")
        history = await chat_service.recent_history(
            session, conv.id, settings.chat_history_messages
        )
    else:
        conv = await chat_service.create_conversation(session, user.id, question)
        history = []
    conversation_id = conv.id

    await chat_service.add_message(session, conversation_id, MessageRole.USER, question)
    chunks = await retrieve(
        session,
        embedder,
        question,
        RetrievalOptions.from_settings(settings),
        document_ids=req.document_ids,
        reranker=reranker,
        user_id=user.id,
    )

    return StreamingResponse(
        _answer_stream(conversation_id, question, history, chunks, llm, session_factory),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _answer_stream(
    conversation_id: uuid.UUID,
    question: str,
    history: list[tuple[str, str]],
    chunks: list[RetrievedChunk],
    llm: LLM,
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[str]:
    yield sse("start", {"conversation_id": str(conversation_id), "sources": len(chunks)})

    if not chunks:
        # No ready documents, or nothing passed the relevance gate: don't spend an LLM call.
        answer = NOT_FOUND_MESSAGE
        yield sse("token", {"text": answer})
    else:
        parts: list[str] = []
        try:
            async for text in llm.stream_answer(
                SYSTEM_PROMPT, build_messages(history, question, chunks)
            ):
                parts.append(text)
                yield sse("token", {"text": text})
        except LLMError as exc:
            yield sse("error", {"message": str(exc)})
            return
        except Exception:
            logger.exception("Answer generation failed")
            yield sse("error", {"message": "Something went wrong while generating the answer."})
            return
        answer = "".join(parts).strip()

    citations = build_citations(answer, chunks)
    # The request's session may already be closed while streaming; use a fresh one.
    async with session_factory() as session:
        msg = await chat_service.add_message(
            session, conversation_id, MessageRole.ASSISTANT, answer, citations
        )
    yield sse(
        "done",
        {
            "conversation_id": str(conversation_id),
            "message_id": msg.id,
            "answer": answer,
            "citations": citations,
        },
    )


class ConversationSummary(BaseModel):
    id: uuid.UUID
    title: str
    created_at: datetime


@router.get("/conversations", response_model=list[ConversationSummary])
async def list_conversations(
    session: SessionDep, user: CurrentUserDep, limit: int = 50
) -> list[ConversationSummary]:
    convs = await chat_service.list_conversations(session, user.id, limit=max(1, min(limit, 200)))
    return [ConversationSummary(id=c.id, title=c.title, created_at=c.created_at) for c in convs]


@router.get("/conversations/{conversation_id}", response_model=ConversationOut)
async def get_conversation(
    conversation_id: uuid.UUID, session: SessionDep, user: CurrentUserDep
) -> ConversationOut:
    conv = await chat_service.get_conversation(session, conversation_id, user.id)
    if conv is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found.")
    messages = await chat_service.list_messages(session, conversation_id)
    return ConversationOut(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        messages=[
            MessageOut(
                id=m.id,
                role=m.role,
                content=m.content,
                citations=m.citations,
                created_at=m.created_at,
            )
            for m in messages
        ],
    )
