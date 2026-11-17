"""Accounts, tokens, per-user isolation, rate limiting and production config helpers."""

import uuid

import pytest

from app.core import security
from app.core.config import Settings, normalize_database_url
from app.db.models import Document, User
from app.main import create_app
from app.services.rate_limit import RateLimiter, Rule, parse_rules
from tests.conftest import JWT_SECRET, TEST_EMAIL, TEST_PASSWORD, TEST_USER_ID, auth_header
from tests.test_chat_api import FakeLLM, parse_sse, use_llm
from tests.test_retrieval import seed_chunks

# ---------- Database URL normalisation (no database) ----------


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (
            "postgresql://u:p@ep-x.eu-central-1.aws.neon.tech/neondb"
            "?sslmode=require&channel_binding=require",
            "postgresql+asyncpg://u:p@ep-x.eu-central-1.aws.neon.tech/neondb?ssl=require",
        ),
        ("postgres://u:p@host:5432/db", "postgresql+asyncpg://u:p@host:5432/db"),
        (
            "postgresql+asyncpg://u:p@localhost/db?ssl=require",
            "postgresql+asyncpg://u:p@localhost/db?ssl=require",
        ),
        ("sqlite+aiosqlite:///x.db", "sqlite+aiosqlite:///x.db"),
    ],
)
def test_normalize_database_url(given, expected):
    assert normalize_database_url(given) == expected
    assert Settings(database_url=given).database_url == expected


# ---------- Passwords and tokens (no database) ----------


def test_password_hash_roundtrip_and_dummy_check():
    h = security.hash_password("s3cret-pass")
    assert h.startswith("$argon2") and "s3cret-pass" not in h
    assert security.verify_password("s3cret-pass", h)
    assert not security.verify_password("wrong", h)
    assert not security.verify_password("anything", None)  # unknown account


def test_token_roundtrip_and_rejections():
    uid = uuid.uuid4()
    token = security.create_access_token(uid, "k" * 32, expire_minutes=5)
    assert security.decode_access_token(token, "k" * 32) == uid
    with pytest.raises(security.InvalidToken):
        security.decode_access_token(token, "other-secret" * 3)  # forged / wrong key
    with pytest.raises(security.InvalidToken):
        security.decode_access_token(
            security.create_access_token(uid, "k" * 32, expire_minutes=-1), "k" * 32
        )  # expired
    with pytest.raises(security.InvalidToken):
        security.decode_access_token("not-a-jwt", "k" * 32)


# ---------- Rate limiter (no database) ----------


def test_parse_rules():
    assert parse_rules("10/minute; 200/day") == [Rule(10, 60), Rule(200, 86400)]
    assert parse_rules("3/seconds") == [Rule(3, 1)]
    assert parse_rules("") == []
    with pytest.raises(ValueError, match="Bad rate limit"):
        parse_rules("ten per minute")


def test_sliding_window_blocks_reports_wait_and_recovers():
    now = [0.0]
    rl = RateLimiter([Rule(2, 60)], clock=lambda: now[0])
    assert rl.hit("u") is None
    now[0] = 10
    assert rl.hit("u") is None
    now[0] = 20
    assert rl.hit("u") == 40  # the first request leaves the window at t=60
    assert rl.hit("other") is None  # limits are per key
    now[0] = 59
    assert rl.hit("u") == 1  # rejected requests don't extend the block
    now[0] = 60.5
    assert rl.hit("u") is None


def test_every_rule_applies_and_empty_means_unlimited():
    now = [0.0]
    rl = RateLimiter([Rule(5, 1), Rule(3, 3600)], clock=lambda: now[0])
    for t in range(3):
        now[0] = t * 10
        assert rl.hit("u") is None
    now[0] = 100
    assert rl.hit("u") == 3600 - 100  # the hourly rule is the binding one
    unlimited = RateLimiter([])
    assert all(unlimited.hit("u") is None for _ in range(1000))


# ---------- Auth API (real database) ----------


async def anon(db_client):
    db_client.headers.pop("Authorization", None)
    return db_client


async def test_register_returns_working_token(db_client):
    await anon(db_client)
    resp = await db_client.post(
        "/auth/register", json={"email": "  New.User@Example.com ", "password": "longenough"}
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["token_type"] == "bearer"
    assert body["user"]["email"] == "new.user@example.com"

    me = await db_client.get(
        "/auth/me", headers={"Authorization": f"Bearer {body['access_token']}"}
    )
    assert me.status_code == 200
    assert me.json()["id"] == body["user"]["id"]


@pytest.mark.parametrize(
    ("payload", "status", "detail"),
    [
        ({"email": TEST_EMAIL.upper(), "password": "longenough"}, 409, "already exists"),
        ({"email": "x@example.com", "password": "short"}, 422, "at least 8"),
        ({"email": "not-an-email", "password": "longenough"}, 422, None),
    ],
)
async def test_register_rejections(db_client, payload, status, detail):
    await anon(db_client)
    resp = await db_client.post("/auth/register", json=payload)
    assert resp.status_code == status
    if detail:
        assert detail in resp.json()["detail"]


async def test_login(db_client):
    await anon(db_client)
    ok = await db_client.post(
        "/auth/login", json={"email": TEST_EMAIL.upper(), "password": TEST_PASSWORD}
    )
    assert ok.status_code == 200
    assert ok.json()["user"]["id"] == str(TEST_USER_ID)
    for email, password in [(TEST_EMAIL, "wrong password"), ("nobody@example.com", "whatever1")]:
        bad = await db_client.post("/auth/login", json={"email": email, "password": password})
        assert bad.status_code == 401
        assert bad.json()["detail"] == "Incorrect email or password."  # same for both cases


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/auth/me"),
        ("get", "/documents"),
        ("post", "/documents"),
        ("get", f"/documents/{uuid.uuid4()}"),
        ("get", "/conversations"),
        ("get", f"/conversations/{uuid.uuid4()}"),
        ("post", "/chat"),
    ],
)
async def test_endpoints_require_a_valid_token(db_client, method, path):
    await anon(db_client)
    assert (await getattr(db_client, method)(path)).status_code == 401
    bad = {"Authorization": "Bearer not-a-token"}
    assert (await getattr(db_client, method)(path, headers=bad)).status_code == 401
    ghost = auth_header(uuid.uuid4())  # validly signed, but the account doesn't exist
    assert (await getattr(db_client, method)(path, headers=ghost)).status_code == 401


async def test_health_stays_public(db_client):
    await anon(db_client)
    assert (await db_client.get("/health")).status_code == 200


# ---------- Isolation between users (real database) ----------


async def make_user(session_factory, email="other@example.com") -> uuid.UUID:
    async with session_factory() as s:
        user = User(email=email, password_hash=security.hash_password("other-password"))
        s.add(user)
        await s.commit()
        return user.id


async def test_users_only_see_their_own_documents(db_client, session_factory):
    mine = await seed_chunks(session_factory, [(1, "my revenue notes", None)], "mine.pdf")
    other_id = await make_user(session_factory)
    theirs = await seed_chunks(
        session_factory, [(1, "secret revenue notes", None)], "theirs.pdf", user_id=other_id
    )
    async with session_factory() as s:  # a pre-accounts document with no owner
        s.add(Document(filename="legacy.pdf", status="ready"))
        await s.commit()

    listed = [d["filename"] for d in (await db_client.get("/documents")).json()]
    assert listed == ["mine.pdf"]
    assert (await db_client.get(f"/documents/{mine}")).status_code == 200
    assert (await db_client.get(f"/documents/{theirs}")).status_code == 404

    as_other = auth_header(other_id)
    listed = [d["filename"] for d in (await db_client.get("/documents", headers=as_other)).json()]
    assert listed == ["theirs.pdf"]


async def test_chat_only_retrieves_from_own_documents(db_client, session_factory):
    other_id = await make_user(session_factory)
    theirs = await seed_chunks(
        session_factory, [(4, "The launch code is 7421", None)], "theirs.pdf", user_id=other_id
    )
    llm = use_llm(FakeLLM(["should not be called"]))

    for body in [
        {"question": "launch code?"},
        # Even naming their document id explicitly doesn't reach it.
        {"question": "launch code?", "document_ids": [str(theirs)]},
    ]:
        events = parse_sse((await db_client.post("/chat", json=body)).text)
        assert events[0][1]["sources"] == 0
    assert llm.calls == []


async def test_conversations_are_private(db_client, session_factory):
    await seed_chunks(session_factory, [(1, "Revenue grew 12 percent", None)], "mine.pdf")
    use_llm(FakeLLM(["Revenue grew 12% [1]."]))
    done = parse_sse((await db_client.post("/chat", json={"question": "Revenue?"})).text)[-1]
    conv_id = done[1]["conversation_id"]

    other = auth_header(await make_user(session_factory))
    assert (await db_client.get("/conversations", headers=other)).json() == []
    assert (await db_client.get(f"/conversations/{conv_id}", headers=other)).status_code == 404
    follow_up = await db_client.post(
        "/chat", json={"question": "And costs?", "conversation_id": conv_id}, headers=other
    )
    assert follow_up.status_code == 404
    assert len((await db_client.get("/conversations")).json()) == 1  # still mine


# ---------- Rate limits (real database) ----------


async def test_chat_rate_limit_is_per_user(db_client, session_factory, test_settings):
    test_settings.chat_rate_limit = "2/minute"
    use_llm(FakeLLM(["ok"] * 10))
    for _ in range(2):
        assert (await db_client.post("/chat", json={"question": "hi"})).status_code == 200
    blocked = await db_client.post("/chat", json={"question": "hi"})
    assert blocked.status_code == 429
    assert 1 <= int(blocked.headers["Retry-After"]) <= 60
    assert "Too many requests" in blocked.json()["detail"]

    other = auth_header(await make_user(session_factory))
    assert (
        await db_client.post("/chat", json={"question": "hi"}, headers=other)
    ).status_code == 200


async def test_login_rate_limit_by_ip(db_client, test_settings):
    await anon(db_client)
    test_settings.auth_rate_limit = "3/minute"
    creds = {"email": TEST_EMAIL, "password": "wrong password"}
    codes = [(await db_client.post("/auth/login", json=creds)).status_code for _ in range(4)]
    assert codes == [401, 401, 401, 429]


# ---------- CORS for the deployed frontend (no database) ----------


def test_cors_allows_configured_origins_and_preview_regex(monkeypatch):
    from fastapi.testclient import TestClient

    import app.main as main

    settings = Settings(
        cors_origins=["https://documind.vercel.app"],
        cors_origin_regex=r"https://documind-[a-z0-9-]+\.vercel\.app",
        jwt_secret=JWT_SECRET,
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    client = TestClient(create_app())

    def preflight(origin):
        return client.options(
            "/chat",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type",
            },
        )

    for origin in ["https://documind.vercel.app", "https://documind-git-main-me.vercel.app"]:
        assert preflight(origin).headers.get("access-control-allow-origin") == origin
    assert "access-control-allow-origin" not in preflight("https://evil.example").headers
