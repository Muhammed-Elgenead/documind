"""Shared FastAPI dependencies (overridable in tests via app.dependency_overrides)."""

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings, get_settings
from app.core.security import InvalidToken, decode_access_token
from app.db.models import User
from app.db.session import SessionLocal, get_session
from app.services import users as users_service
from app.services.embeddings import Embedder, get_embedder
from app.services.rate_limit import limiter_for
from app.services.reranking import Reranker, get_reranker


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Background tasks outlive the request, so they open their own sessions."""
    return SessionLocal


SessionDep = Annotated[AsyncSession, Depends(get_session)]
SessionFactoryDep = Annotated[async_sessionmaker[AsyncSession], Depends(get_session_factory)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
RerankerDep = Annotated[Reranker, Depends(get_reranker)]

_bearer = HTTPBearer(auto_error=False)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status.HTTP_401_UNAUTHORIZED, detail, headers={"WWW-Authenticate": "Bearer"}
    )


async def get_current_user(
    session: SessionDep,
    settings: SettingsDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> User:
    """The user behind a valid `Authorization: Bearer <token>` header, or 401."""
    if credentials is None:
        raise _unauthorized("Not signed in.")
    try:
        user_id = decode_access_token(credentials.credentials, settings.jwt_secret)
    except InvalidToken as exc:
        raise _unauthorized("Your session has expired. Please sign in again.") from exc
    user = await users_service.get_user(session, user_id)
    if user is None:  # account deleted after the token was issued
        raise _unauthorized("Your session has expired. Please sign in again.")
    return user


CurrentUserDep = Annotated[User, Depends(get_current_user)]


def _too_many(retry_after: int) -> HTTPException:
    return HTTPException(
        status.HTTP_429_TOO_MANY_REQUESTS,
        f"Too many requests. Please try again in {retry_after} s.",
        headers={"Retry-After": str(retry_after)},
    )


async def limit_chat(user: CurrentUserDep, settings: SettingsDep) -> None:
    """Per-user limit on /chat: every request may cost an LLM call."""
    if wait := limiter_for("chat", settings.chat_rate_limit).hit(str(user.id)):
        raise _too_many(wait)


async def limit_auth(request: Request, settings: SettingsDep) -> None:
    """Per-client-IP limit on register/login, against password guessing.

    Behind a hosting proxy the client IP comes from X-Forwarded-For; uvicorn applies it
    when started with --proxy-headers (see the backend Dockerfile).
    """
    ip = request.client.host if request.client else "unknown"
    if wait := limiter_for("auth", settings.auth_rate_limit).hit(ip):
        raise _too_many(wait)
