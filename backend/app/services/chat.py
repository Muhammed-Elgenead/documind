"""Conversation and message persistence."""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Conversation, Message, MessageRole


async def create_conversation(
    session: AsyncSession, user_id: uuid.UUID, first_question: str
) -> Conversation:
    title = " ".join(first_question.split())[:200] or "New conversation"
    conv = Conversation(title=title, user_id=user_id)
    session.add(conv)
    await session.commit()
    await session.refresh(conv)
    return conv


async def get_conversation(
    session: AsyncSession, conversation_id: uuid.UUID, user_id: uuid.UUID
) -> Conversation | None:
    """The conversation if it exists and belongs to the user (else None)."""
    conv = await session.get(Conversation, conversation_id)
    return conv if conv is not None and conv.user_id == user_id else None


async def add_message(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    role: MessageRole,
    content: str,
    citations: list[dict] | None = None,
) -> Message:
    msg = Message(conversation_id=conversation_id, role=role, content=content, citations=citations)
    session.add(msg)
    await session.commit()
    await session.refresh(msg)
    return msg


async def list_conversations(
    session: AsyncSession, user_id: uuid.UUID, limit: int = 50
) -> list[Conversation]:
    """The user's conversations, newest first."""
    result = await session.execute(
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.created_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def list_messages(session: AsyncSession, conversation_id: uuid.UUID) -> list[Message]:
    result = await session.execute(
        select(Message).where(Message.conversation_id == conversation_id).order_by(Message.id)
    )
    return list(result.scalars().all())


async def recent_history(
    session: AsyncSession, conversation_id: uuid.UUID, limit: int
) -> list[tuple[str, str]]:
    """Last `limit` messages as (role, content), oldest first, starting with a user turn."""
    if limit <= 0:
        return []
    result = await session.execute(
        select(Message.role, Message.content)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.id.desc())
        .limit(limit)
    )
    rows = [(r.role, r.content) for r in reversed(result.all())]
    # The API expects alternating turns starting with "user". Drop a leading assistant
    # turn (cut off by the limit) and any trailing user turn whose answer was never saved.
    while rows and rows[0][0] != MessageRole.USER:
        rows.pop(0)
    while rows and rows[-1][0] != MessageRole.ASSISTANT:
        rows.pop()
    return rows
