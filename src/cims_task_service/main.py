"""FastAPI application entry point."""

from fastapi import FastAPI

from cims_task_service import __version__
from cims_task_service.api import router as api_router
from cims_task_service.config import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build an isolated application instance."""

    resolved_settings = settings if settings is not None else Settings()
    application = FastAPI(
        title="CIMS Task Service",
        description="Fault-tolerant asynchronous task processing service",
        version=__version__,
        debug=resolved_settings.debug,
    )
    application.include_router(api_router)
    return application


app = create_app()
