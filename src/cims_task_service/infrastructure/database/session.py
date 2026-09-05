"""Factories for asynchronous SQLAlchemy database resources."""

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from cims_task_service.config import Settings

type AsyncSessionFactory = async_sessionmaker[AsyncSession]


def create_database_engine(settings: Settings) -> AsyncEngine:
    """Build a lazy async engine without opening a database connection."""

    return create_async_engine(
        settings.database_url.get_secret_value(),
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        pool_timeout=settings.database_pool_timeout_seconds,
        pool_recycle=settings.database_pool_recycle_seconds,
        pool_pre_ping=True,
        isolation_level="READ COMMITTED",
        hide_parameters=True,
        connect_args={
            "server_settings": {
                "application_name": "cims-task-service",
                "timezone": "UTC",
            }
        },
    )


def create_session_factory(engine: AsyncEngine) -> AsyncSessionFactory:
    """Create independent async sessions bound to the supplied engine."""

    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        autoflush=False,
        expire_on_commit=False,
    )


async def dispose_database_engine(engine: AsyncEngine) -> None:
    """Close all pooled database connections during application shutdown."""

    await engine.dispose()
