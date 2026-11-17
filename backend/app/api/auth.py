"""Email + password accounts with bearer-token (JWT) sessions.

POST /auth/register  {email, password} -> 201 {access_token, token_type, user}
POST /auth/login     {email, password} -> 200 {access_token, token_type, user}
GET  /auth/me                          -> 200 {id, email, created_at}
"""

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.api.deps import CurrentUserDep, SessionDep, SettingsDep, limit_auth
from app.core.security import create_access_token
from app.db.models import User
from app.services import users as users_service

router = APIRouter(prefix="/auth", tags=["auth"])

MIN_PASSWORD = 8


class Credentials(BaseModel):
    email: EmailStr
    # Argon2 hashes any length; the upper bound just stops absurd payloads.
    password: str = Field(min_length=1, max_length=256)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    created_at: datetime


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


def _token_for(user: User, settings) -> TokenOut:
    token = create_access_token(user.id, settings.jwt_secret, settings.jwt_expire_minutes)
    return TokenOut(access_token=token, user=UserOut.model_validate(user))


@router.post(
    "/register",
    response_model=TokenOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(limit_auth)],
)
async def register(body: Credentials, session: SessionDep, settings: SettingsDep) -> TokenOut:
    if len(body.password) < MIN_PASSWORD:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"Password must be at least {MIN_PASSWORD} characters.",
        )
    try:
        user = await users_service.create_user(session, body.email, body.password)
    except users_service.EmailTaken as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "An account with this email already exists."
        ) from exc
    return _token_for(user, settings)


@router.post("/login", response_model=TokenOut, dependencies=[Depends(limit_auth)])
async def login(body: Credentials, session: SessionDep, settings: SettingsDep) -> TokenOut:
    user = await users_service.authenticate(session, body.email, body.password)
    if user is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _token_for(user, settings)


@router.get("/me", response_model=UserOut)
async def me(user: CurrentUserDep) -> UserOut:
    return UserOut.model_validate(user)
