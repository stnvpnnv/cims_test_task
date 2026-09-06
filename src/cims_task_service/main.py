"""FastAPI application entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from cims_task_service import __version__
from cims_task_service.api import router as api_router
from cims_task_service.api.dependencies import ApplicationResources
from cims_task_service.config import Settings
from cims_task_service.infrastructure.database.session import (
    create_database_engine,
    create_session_factory,
    dispose_database_engine,
)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build an isolated application instance."""

    resolved_settings = settings if settings is not None else Settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        engine = create_database_engine(resolved_settings)
        try:
            application.state.resources = ApplicationResources(
                session_factory=create_session_factory(engine),
                task_max_attempts=resolved_settings.task_max_attempts,
            )
            try:
                yield
            finally:
                del application.state.resources
        finally:
            await dispose_database_engine(engine)

    application = FastAPI(
        title="CIMS Task Service",
        description="Fault-tolerant asynchronous task processing service",
        version=__version__,
        debug=resolved_settings.debug,
        lifespan=lifespan,
    )
    application.include_router(api_router)
    return application


app = create_app()
