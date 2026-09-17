"""Tests for robust RabbitMQ connection factories."""

import asyncio
from contextlib import suppress
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

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

    connect = AsyncMock()
    close = AsyncMock()
    expected_connection = cast(
        AbstractRobustConnection,
        SimpleNamespace(connect=connect, close=close),
    )
    connection_factory = Mock(return_value=expected_connection)
    monkeypatch.setattr(messaging_connection, "RobustConnection", connection_factory)
    settings = Settings(
        rabbitmq_url=SecretStr("amqps://service:rabbit-secret@rabbitmq:5671/%2Ftasks"),
        rabbitmq_connection_timeout_seconds=12.5,
        rabbitmq_reconnect_interval_seconds=2.5,
    )

    connection = await connect_rabbitmq(settings)

    assert connection is expected_connection
    assert connection_factory.call_args is not None
    assert str(connection_factory.call_args.args[0]) == (
        "amqps://service:rabbit-secret@rabbitmq:5671/%2Ftasks"
    )
    assert connection_factory.call_args.kwargs == {
        "reconnect_interval": 2.5,
        "fail_fast": True,
    }
    connect.assert_awaited_once_with(timeout=12.5)
    close.assert_not_awaited()


@pytest.mark.asyncio
async def test_connection_factory_propagates_initial_connection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail-fast startup closes its connection before propagating the error."""

    expected_error = ConnectionError("broker unavailable")
    connect = AsyncMock(side_effect=expected_error)
    close = AsyncMock()
    connection = cast(
        AbstractRobustConnection,
        SimpleNamespace(connect=connect, close=close),
    )
    monkeypatch.setattr(
        messaging_connection,
        "RobustConnection",
        Mock(return_value=connection),
    )

    with pytest.raises(ConnectionError) as error_info:
        await connect_rabbitmq(Settings())

    assert error_info.value is expected_error
    close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_connection_factory_cleans_up_after_base_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-Exception failures cannot orphan the owned reconnect task."""

    class FatalConnectError(BaseException):
        pass

    expected_error = FatalConnectError()
    close = AsyncMock()
    connection = cast(
        AbstractRobustConnection,
        SimpleNamespace(
            connect=AsyncMock(side_effect=expected_error),
            close=close,
        ),
    )
    monkeypatch.setattr(
        messaging_connection,
        "RobustConnection",
        Mock(return_value=connection),
    )

    with pytest.raises(FatalConnectError) as error_info:
        await connect_rabbitmq(Settings())

    assert error_info.value is expected_error
    close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_connection_factory_closes_connection_after_task_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation stops the reconnect task before leaving the factory."""

    connect_started = asyncio.Event()
    never_connected = asyncio.Event()
    close_started = asyncio.Event()
    allow_close = asyncio.Event()
    close_completed = False

    async def wait_for_connection(**options: float) -> None:
        assert options == {"timeout": 10.0}
        connect_started.set()
        await never_connected.wait()

    async def close_connection() -> None:
        nonlocal close_completed
        close_started.set()
        await allow_close.wait()
        close_completed = True

    close = AsyncMock(side_effect=close_connection)
    connection = cast(
        AbstractRobustConnection,
        SimpleNamespace(
            connect=AsyncMock(side_effect=wait_for_connection),
            close=close,
        ),
    )
    monkeypatch.setattr(
        messaging_connection,
        "RobustConnection",
        Mock(return_value=connection),
    )
    connection_task = asyncio.create_task(connect_rabbitmq(Settings()))

    try:
        async with asyncio.timeout(1):
            await connect_started.wait()
        connection_task.cancel()
        async with asyncio.timeout(1):
            await close_started.wait()
        assert not connection_task.done()
        allow_close.set()

        with pytest.raises(asyncio.CancelledError):
            await connection_task
    finally:
        allow_close.set()
        if not connection_task.done():
            connection_task.cancel()
            with suppress(asyncio.CancelledError):
                await connection_task

    assert connection_task.cancelled()
    assert close_completed is True
    close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_connection_close_stops_the_owned_resource_once() -> None:
    """Shutdown delegates exactly once to the robust connection owner."""

    close = AsyncMock()
    connection = cast(AbstractRobustConnection, SimpleNamespace(close=close))

    await close_rabbitmq_connection(connection)

    close.assert_awaited_once_with()
