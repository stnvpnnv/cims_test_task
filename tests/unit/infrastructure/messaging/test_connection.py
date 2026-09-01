"""Tests for robust RabbitMQ connection factories."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from aio_pika.abc import AbstractRobustConnection
from pydantic import SecretStr

from cims_task_service.config import Settings
from cims_task_service.infrastructure.messaging import connection as messaging_connection
from cims_task_service.infrastructure.messaging.connection import (
    close_rabbitmq_connection,
    connect_rabbitmq,
)


@pytest.fixture(autouse=True)
def clear_settings_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep connection tests independent from process-level settings."""

    for field_name in Settings.model_fields:
        monkeypatch.delenv(f"CIMS_{field_name.upper()}", raising=False)


@pytest.mark.asyncio
async def test_connection_factory_applies_startup_and_reconnect_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Connection creation forwards credentials and the resilience policy."""

    captured_url: list[str] = []
    captured_options: dict[str, object] = {}
    expected_connection = cast(AbstractRobustConnection, object())

    async def fake_connect_robust(
        url: str,
        **options: object,
    ) -> AbstractRobustConnection:
        captured_url.append(url)
        captured_options.update(options)
        return expected_connection

    monkeypatch.setattr(
        messaging_connection,
        "connect_robust",
        fake_connect_robust,
    )
    settings = Settings(
        rabbitmq_url=SecretStr("amqps://service:rabbit-secret@rabbitmq:5671/%2Ftasks"),
        rabbitmq_connection_timeout_seconds=12.5,
        rabbitmq_reconnect_interval_seconds=2.5,
    )

    connection = await connect_rabbitmq(settings)

    assert connection is expected_connection
    assert captured_url == ["amqps://service:rabbit-secret@rabbitmq:5671/%2Ftasks"]
    assert captured_options == {
        "timeout": 12.5,
        "reconnect_interval": 2.5,
        "fail_fast": True,
    }


@pytest.mark.asyncio
async def test_connection_factory_propagates_initial_connection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The process owner can react when fail-fast startup cannot connect."""

    expected_error = ConnectionError("broker unavailable")

    async def failing_connect_robust(
        url: str,
        **options: object,
    ) -> AbstractRobustConnection:
        raise expected_error

    monkeypatch.setattr(
        messaging_connection,
        "connect_robust",
        failing_connect_robust,
    )

    with pytest.raises(ConnectionError) as error_info:
        await connect_rabbitmq(Settings())

    assert error_info.value is expected_error


@pytest.mark.asyncio
async def test_connection_close_stops_the_owned_resource_once() -> None:
    """Shutdown delegates exactly once to the robust connection owner."""

    close = AsyncMock()
    connection = cast(AbstractRobustConnection, SimpleNamespace(close=close))

    await close_rabbitmq_connection(connection)

    close.assert_awaited_once_with()
