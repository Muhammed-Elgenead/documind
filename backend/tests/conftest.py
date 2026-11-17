import os
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.deps import get_current_user, get_session_factory
from app.core.config import Settings, get_settings
from app.core.security import create_access_token, hash_password
from app.db.models import User
from app.db.session import get_session
from app.main import app
from app.services.embeddings import HashEmbedder, get_embedder
from app.services.rate_limit import limiter_for
from app.services.reranking import TermOverlapReranker, get_reranker

BACKEND_DIR = Path(__file__).resolve().parents[1]

# Every test database has this user; seeded documents and conversations belong to it.
TEST_USER_ID = uuid.UUID("00000000-0000-0000-0000-00000000000a")
TEST_EMAIL = "tester@example.com"
TEST_PASSWORD = "correct horse battery"
_TEST_PASSWORD_HASH = hash_password(TEST_PASSWORD)  # Argon2 is slow on purpose: hash once
JWT_SECRET = "test-secret-" + "x" * 32


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    return Settings(
        upload_dir=tmp_path / "uploads",
        max_upload_mb=1,
        embedding_provider="hash",
        reranker_provider="overlap",
        # Gates are calibrated for real models; hash/overlap scores use other scales.
        retrieval_min_similarity=0.0,
        rerank_min_score=0.0,
        # Small chunks so modest test documents produce several chunks per page.
        chunk_size_tokens=120,
        chunk_overlap_tokens=20,
        jwt_secret=JWT_SECRET,
        # Rate limits off unless a test turns them on.
        chat_rate_limit="",
        auth_rate_limit="",
    )


def auth_header(user_id: uuid.UUID = TEST_USER_ID) -> dict[str, str]:
    return {"Authorization": f"Bearer {create_access_token(user_id, JWT_SECRET, 60)}"}


@pytest.fixture
async def client(test_settings: Settings) -> AsyncIterator[AsyncClient]:
    """App client with no database: a stand-in user is signed in.

    Tests override get_session themselves if they need one; `db_client` uses real auth.
    """
    limiter_for.cache_clear()
    app.dependency_overrides[get_settings] = lambda: test_settings
    app.dependency_overrides[get_embedder] = HashEmbedder
    app.dependency_overrides[get_reranker] = TermOverlapReranker
    app.dependency_overrides[get_current_user] = lambda: User(
        id=TEST_USER_ID, email=TEST_EMAIL, password_hash=_TEST_PASSWORD_HASH
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()
    limiter_for.cache_clear()


# ---------- Real-database fixtures (skipped unless TEST_DATABASE_URL is set) ----------


@pytest.fixture(scope="session")
def test_database_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    return url


@pytest.fixture(scope="session")
def migrated_database(test_database_url: str) -> Iterator[str]:
    """Run Alembic migrations once per test session (also tests the migrations)."""
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "app" / "db" / "migrations"))
    cfg.set_main_option("sqlalchemy.url", test_database_url)
    cfg.attributes["configure_logger"] = False
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
    yield test_database_url


@pytest.fixture
async def session_factory(
    migrated_database: str,
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(migrated_database)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "TRUNCATE users, documents, chunks, conversations, messages "
                "RESTART IDENTITY CASCADE"
            )
        )
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        s.add(User(id=TEST_USER_ID, email=TEST_EMAIL, password_hash=_TEST_PASSWORD_HASH))
        await s.commit()
    yield factory
    await engine.dispose()


@pytest.fixture
async def db_client(
    client: AsyncClient, session_factory: async_sessionmaker[AsyncSession]
) -> AsyncClient:
    """App client wired to the real test database, signed in as the test user via a JWT."""

    async def _get_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as s:
            yield s

    app.dependency_overrides[get_session] = _get_session
    app.dependency_overrides[get_session_factory] = lambda: session_factory
    app.dependency_overrides.pop(get_current_user, None)  # exercise the real token check
    client.headers.update(auth_header())
    return client
