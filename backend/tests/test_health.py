from unittest.mock import AsyncMock

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.db.session import get_session
from app.main import app
from app.services.health import check_database


def _override_session(session: AsyncMock) -> None:
    async def _get_session():
        yield session

    app.dependency_overrides[get_session] = _get_session


async def test_health_ok(client):
    _override_session(AsyncMock(spec=AsyncSession))
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "healthy", "database": "ok"}


async def test_health_db_down(client):
    session = AsyncMock(spec=AsyncSession)
    session.execute.side_effect = ConnectionRefusedError("db down")
    _override_session(session)
    resp = await client.get("/health")
    assert resp.status_code == 503
    assert resp.json() == {"status": "unhealthy", "database": "error"}


async def test_check_database_true():
    assert await check_database(AsyncMock(spec=AsyncSession)) is True


async def test_check_database_false():
    session = AsyncMock(spec=AsyncSession)
    session.execute.side_effect = OSError("boom")
    assert await check_database(session) is False


async def test_check_database_real_db_with_vector(migrated_database):
    """Integration: real Postgres is reachable and migrations enabled pgvector."""
    engine = create_async_engine(migrated_database)
    try:
        async with AsyncSession(engine) as session:
            assert await check_database(session) is True
            result = await session.execute(
                text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            )
            assert result.scalar_one() == "vector"
    finally:
        await engine.dispose()
