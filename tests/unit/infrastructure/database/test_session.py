"""Tests for asynchronous database resource factories."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from cims_task_service.config import Settings
from cims_task_service.infrastructure.database import session as database_session
from cims_task_service.infrastructure.database.session import (
    create_database_engine,
    create_session_factory,
    dispose_database_engine,
)


def test_engine_factory_applies_connection_and_pool_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Engine construction forwards the complete operational policy."""

    captured_url: list[str] = []
    captured_options: dict[str, object] = {}
    expected_engine = cast(AsyncEngine, object())

    def fake_create_async_engine(url: str, **options: object) -> AsyncEngine:
        captured_url.append(url)
        captured_options.update(options)
        return expected_engine

    monkeypatch.setattr(
        database_session,
        "create_async_engine",
        fake_create_async_engine,
    )
    settings = Settings(
        database_url=SecretStr("postgresql+asyncpg://service@postgres:5432/tasks"),
        database_pool_size=7,
        database_max_overflow=3,
        database_pool_timeout_seconds=11.5,
        database_pool_recycle_seconds=600,
    )

    engine = create_database_engine(settings)

    assert engine is expected_engine
    assert captured_url == ["postgresql+asyncpg://service@postgres:5432/tasks"]
    assert captured_options == {
        "pool_size": 7,
        "max_overflow": 3,
        "pool_timeout": 11.5,
        "pool_recycle": 600,
        "pool_pre_ping": True,
        "isolation_level": "READ COMMITTED",
        "hide_parameters": True,
        "connect_args": {
            "server_settings": {
                "application_name": "cims-task-service",
                "timezone": "UTC",
            }
        },
    }


@pytest.mark.asyncio
async def test_session_factory_does_not_require_a_connection() -> None:
    """Independent sessions can be configured before the first query."""

    settings = Settings(database_url=SecretStr("postgresql+asyncpg://service@localhost:5432/tasks"))
    engine = create_database_engine(settings)
    session_factory = create_session_factory(engine)

    try:
        async with session_factory() as first_session, session_factory() as second_session:
            assert isinstance(first_session, AsyncSession)
            assert first_session is not second_session
            assert first_session.sync_session.bind is engine.sync_engine
            assert first_session.sync_session.autoflush is False
            assert first_session.sync_session.expire_on_commit is False
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_engine_disposal_closes_the_pool_once() -> None:
    """Shutdown delegates exactly once to the engine pool owner."""

    dispose = AsyncMock()
    engine = cast(AsyncEngine, SimpleNamespace(dispose=dispose))

    await dispose_database_engine(engine)

    dispose.assert_awaited_once_with()
