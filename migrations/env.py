"""Alembic environment for asynchronous PostgreSQL migrations."""

import asyncio

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from cims_task_service.config import Settings
from cims_task_service.infrastructure.database import models as database_models

config = context.config
target_metadata = database_models.Base.metadata


def _get_database_url() -> str:
    """Load the validated database URL without storing it in Alembic config."""

    return Settings().database_url.get_secret_value()


def run_migrations_offline() -> None:
    """Render migration SQL without establishing a database connection."""

    context.configure(
        url=_get_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def _run_migrations(connection: Connection) -> None:
    """Run migrations through Alembic's synchronous migration context."""

    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def _run_async_migrations() -> None:
    """Create and dispose a one-shot async engine for an online migration."""

    engine = create_async_engine(
        _get_database_url(),
        poolclass=pool.NullPool,
        hide_parameters=True,
        connect_args={
            "server_settings": {
                "application_name": "cims-task-service-alembic",
                "timezone": "UTC",
            }
        },
    )
    try:
        async with engine.connect() as connection:
            await connection.run_sync(_run_migrations)
    finally:
        await engine.dispose()


def run_migrations_online() -> None:
    """Run migrations with a supplied connection or a temporary async engine."""

    connection = config.attributes.get("connection")
    if connection is not None:
        if not isinstance(connection, Connection):
            message = "Alembic's shared connection must be a SQLAlchemy Connection"
            raise TypeError(message)
        _run_migrations(connection)
        return

    asyncio.run(_run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
