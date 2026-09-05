"""Opt-in PostgreSQL fixtures with a migrated schema owned by each test."""

import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast
from uuid import uuid4

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from pydantic import SecretStr
from sqlalchemy import event, text
from sqlalchemy.engine import Connection, make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.schema import CreateSchema, DropSchema

from cims_task_service.config import Settings
from cims_task_service.infrastructure.database.session import (
    AsyncSessionFactory,
    create_database_engine,
    create_session_factory,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def postgres_database_url() -> SecretStr:
    """Require an explicit test database; never fall back to the application URL."""

    raw_url = os.environ.get("CIMS_TEST_DATABASE_URL")
    if raw_url is None:
        pytest.skip("Set CIMS_TEST_DATABASE_URL to run PostgreSQL integration tests")

    try:
        url = make_url(raw_url)
    except (ArgumentError, ValueError):
        pytest.fail("CIMS_TEST_DATABASE_URL must be a valid SQLAlchemy URL", pytrace=False)

    if (
        url.drivername != "postgresql+asyncpg"
        or not url.host
        or not url.database
        or not url.database.endswith("_test")
        or url.query
    ):
        pytest.fail(
            "CIMS_TEST_DATABASE_URL must use postgresql+asyncpg, specify a host and a "
            "database ending in _test, and contain no query parameters",
            pytrace=False,
        )

    return SecretStr(raw_url)


def _configure_test_connections(engine: AsyncEngine, *, schema: str | None = None) -> None:
    """Apply bounded waits and an optional search path before driver connection."""

    @event.listens_for(engine.sync_engine, "do_connect")
    def configure_connection(
        _dialect: object,
        _connection_record: object,
        _args: object,
        parameters: dict[str, object],
    ) -> None:
        server_settings = dict(cast(dict[str, str], parameters["server_settings"]))
        server_settings.update(lock_timeout="15000", statement_timeout="20000")
        if schema is not None:
            # Startup settings survive transaction rollbacks; public is excluded.
            server_settings["search_path"] = schema
        parameters["server_settings"] = server_settings
        parameters["timeout"] = 10


def _upgrade_schema(connection: Connection) -> None:
    """Use the project's complete migration chain on the isolated connection."""

    config = Config(toml_file=_PROJECT_ROOT / "pyproject.toml")
    config.attributes["connection"] = connection
    command.upgrade(config, "head")


@pytest_asyncio.fixture
async def postgres_engine(postgres_database_url: SecretStr) -> AsyncIterator[AsyncEngine]:
    """Create, migrate, and finally remove only this test's random schema."""

    settings = Settings(
        database_url=postgres_database_url,
        database_pool_size=5,
        database_max_overflow=0,
        database_pool_timeout_seconds=5,
    )
    admin_engine = create_database_engine(settings)
    _configure_test_connections(admin_engine)
    schema = f"cims_test_{uuid4().hex}"

    try:
        async with admin_engine.begin() as connection:
            await connection.execute(CreateSchema(schema))

        try:
            # Create the schema before the engine discovers its default schema.
            test_engine = create_database_engine(settings)
            try:
                _configure_test_connections(test_engine, schema=schema)
                async with test_engine.begin() as connection:
                    assert await connection.scalar(text("SELECT current_schema()")) == schema
                    await connection.run_sync(_upgrade_schema)

                yield test_engine
            finally:
                await test_engine.dispose()
        finally:
            async with admin_engine.begin() as connection:
                await connection.execute(DropSchema(schema, cascade=True))
    finally:
        await admin_engine.dispose()


@pytest.fixture
def postgres_session_factory(postgres_engine: AsyncEngine) -> AsyncSessionFactory:
    """Use production session options with independent transactions per request."""

    return create_session_factory(postgres_engine)
