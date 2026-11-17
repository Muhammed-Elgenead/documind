"""User accounts."""

import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import hash_password, verify_password
from app.db.models import User


class EmailTaken(Exception):
    pass


def normalize_email(email: str) -> str:
    return email.strip().lower()


async def create_user(session: AsyncSession, email: str, password: str) -> User:
    user = User(email=normalize_email(email), password_hash=hash_password(password))
    session.add(user)
    try:
        await session.commit()
    except IntegrityError as exc:  # unique email, also under concurrent sign-ups
        await session.rollback()
        raise EmailTaken from exc
    await session.refresh(user)
    return user


async def get_user(session: AsyncSession, user_id: uuid.UUID) -> User | None:
    return await session.get(User, user_id)


async def authenticate(session: AsyncSession, email: str, password: str) -> User | None:
    """The user if the email and password match, else None (same timing either way)."""
    user = (
        await session.execute(select(User).where(User.email == normalize_email(email)))
    ).scalar_one_or_none()
    ok = verify_password(password, user.password_hash if user else None)
    return user if ok else None
