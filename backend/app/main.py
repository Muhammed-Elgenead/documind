import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.core.config import get_settings
from app.db.session import SessionLocal
from app.services.documents import fail_interrupted_documents

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Background tasks die with the process; anything still "processing" at startup
    # will never finish. Single backend instance assumed (see docs/decisions.md).
    try:
        async with SessionLocal() as session:
            if n := await fail_interrupted_documents(session):
                logger.warning("Marked %d interrupted document(s) as failed", n)
    except Exception:
        logger.exception("Startup cleanup skipped (database unavailable?)")
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="DocuMind", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_origin_regex=settings.cors_origin_regex or None,
        # Auth is a bearer header, not cookies, so credentials mode isn't needed.
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        # Let the browser read Retry-After on 429 responses.
        expose_headers=["Retry-After"],
    )
    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(documents_router)
    app.include_router(chat_router)
    return app


app = create_app()
