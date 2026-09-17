"""Tests for dispatcher process resource composition."""

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from datetime import timedelta
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from aio_pika.abc import (
    AbstractRobustChannel,
    AbstractRobustConnection,
    AbstractRobustExchange,
)
from sqlalchemy.ext.asyncio import AsyncEngine

from cims_task_service import dispatcher as dispatcher_module
from cims_task_service.application.task_dispatcher import TaskOutboxDispatcher
from cims_task_service.config import DispatcherSettings
from cims_task_service.infrastructure.database.session import AsyncSessionFactory
from cims_task_service.infrastructure.messaging.publisher import RabbitMQTaskPublisher


@dataclass(slots=True)
class _RuntimeHarness:
    engine: AsyncEngine
    session_factory: AsyncSessionFactory
    connection: AbstractRobustConnection
    channel: AbstractRobustChannel
    exchange: AbstractRobustExchange
    publisher: RabbitMQTaskPublisher
    dispatcher: TaskOutboxDispatcher
    create_engine: Mock
    create_sessions: Mock
    connect: AsyncMock
    close_connection: AsyncMock
    open_channel: AsyncMock
    close_channel: AsyncMock
    declare_topology: AsyncMock
    publisher_factory: Mock
    dispatcher_factory: Mock
    run_loop: AsyncMock
    dispose_engine: AsyncMock
    events: list[str]


@pytest.fixture(autouse=True)
def clear_settings_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep dispatcher settings independent from the developer environment."""

    for field_name in DispatcherSettings.model_fields:
        monkeypatch.delenv(f"CIMS_{field_name.upper()}", raising=False)


def _install_runtime_harness(
    monkeypatch: pytest.MonkeyPatch,
    *,
    failure_stage: str | None = None,
    expected_error: BaseException | None = None,
) -> _RuntimeHarness:
    events: list[str] = []
    engine = cast(AsyncEngine, object())
    session_factory = cast(AsyncSessionFactory, object())
    connection = cast(AbstractRobustConnection, object())
    exchange = cast(AbstractRobustExchange, object())
    close_channel = AsyncMock(side_effect=lambda: events.append("channel"))
    channel = cast(
        AbstractRobustChannel,
        SimpleNamespace(close=close_channel),
    )
    publisher = cast(RabbitMQTaskPublisher, object())
    dispatch_once = AsyncMock()
    dispatcher = cast(
        TaskOutboxDispatcher,
        SimpleNamespace(dispatch_once=dispatch_once),
    )

    create_engine = Mock(return_value=engine)
    create_sessions = Mock(return_value=session_factory)
    connect = AsyncMock(return_value=connection)
    close_connection = AsyncMock(side_effect=lambda _connection: events.append("connection"))
    open_channel = AsyncMock(return_value=channel)
    declare_topology = AsyncMock(
        return_value=SimpleNamespace(task_exchange=exchange),
    )
    publisher_factory = Mock(return_value=publisher)
    dispatcher_factory = Mock(return_value=dispatcher)
    run_loop = AsyncMock(side_effect=lambda *_args, **_kwargs: events.append("loop"))
    dispose_engine = AsyncMock(side_effect=lambda _engine: events.append("engine"))

    failure_targets: dict[str, Mock | AsyncMock] = {
        "engine": create_engine,
        "sessions": create_sessions,
        "connect": connect,
        "channel": open_channel,
        "topology": declare_topology,
        "publisher": publisher_factory,
        "dispatcher": dispatcher_factory,
        "loop": run_loop,
    }
    if failure_stage is not None:
        if expected_error is None:
            raise ValueError("expected_error is required with failure_stage")
        failure_targets[failure_stage].side_effect = expected_error

    monkeypatch.setattr(dispatcher_module, "create_database_engine", create_engine)
    monkeypatch.setattr(dispatcher_module, "create_session_factory", create_sessions)
    monkeypatch.setattr(dispatcher_module, "connect_rabbitmq", connect)
    monkeypatch.setattr(
        dispatcher_module,
        "close_rabbitmq_connection",
        close_connection,
    )
    monkeypatch.setattr(dispatcher_module, "open_publisher_channel", open_channel)
    monkeypatch.setattr(dispatcher_module, "declare_task_topology", declare_topology)
    monkeypatch.setattr(dispatcher_module, "RabbitMQTaskPublisher", publisher_factory)
    monkeypatch.setattr(dispatcher_module, "TaskOutboxDispatcher", dispatcher_factory)
    monkeypatch.setattr(dispatcher_module, "run_dispatcher_loop", run_loop)
    monkeypatch.setattr(dispatcher_module, "dispose_database_engine", dispose_engine)

    return _RuntimeHarness(
        engine=engine,
        session_factory=session_factory,
        connection=connection,
        channel=channel,
        exchange=exchange,
        publisher=publisher,
        dispatcher=dispatcher,
        create_engine=create_engine,
        create_sessions=create_sessions,
        connect=connect,
        close_connection=close_connection,
        open_channel=open_channel,
        close_channel=close_channel,
        declare_topology=declare_topology,
        publisher_factory=publisher_factory,
        dispatcher_factory=dispatcher_factory,
        run_loop=run_loop,
        dispose_engine=dispose_engine,
        events=events,
    )


def _settings() -> DispatcherSettings:
    return DispatcherSettings(
        database_pool_size=4,
        database_max_overflow=0,
        database_pool_timeout_seconds=5.0,
        dispatcher_batch_size=4,
        dispatcher_poll_interval_seconds=0.25,
        dispatcher_lease_duration_seconds=30.0,
        dispatcher_retry_initial_delay_seconds=2.0,
        dispatcher_retry_maximum_delay_seconds=8.0,
        rabbitmq_publish_timeout_seconds=3.0,
    )


async def _start_blocked_runtime(harness: _RuntimeHarness) -> asyncio.Task[None]:
    loop_started = asyncio.Event()

    async def wait_for_cancellation(*_args: object, **_kwargs: object) -> None:
        loop_started.set()
        await asyncio.Event().wait()

    harness.run_loop.side_effect = wait_for_cancellation
    runtime_task = asyncio.create_task(
        dispatcher_module.run_dispatcher(
            _settings(),
            stop_event=asyncio.Event(),
        )
    )
    try:
        async with asyncio.timeout(1):
            await loop_started.wait()
    except BaseException:
        runtime_task.cancel()
        with suppress(asyncio.CancelledError):
            await runtime_task
        raise

    return runtime_task


@pytest.mark.asyncio
async def test_dispatcher_builds_runtime_and_closes_resources_in_reverse_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Composition forwards validated settings and owns every opened resource."""

    harness = _install_runtime_harness(monkeypatch)
    settings = _settings()
    stop_event = asyncio.Event()

    await dispatcher_module.run_dispatcher(settings, stop_event=stop_event)

    harness.create_engine.assert_called_once_with(settings)
    harness.create_sessions.assert_called_once_with(harness.engine)
    harness.connect.assert_awaited_once_with(settings)
    harness.open_channel.assert_awaited_once_with(harness.connection)
    harness.declare_topology.assert_awaited_once_with(harness.channel)
    harness.publisher_factory.assert_called_once_with(
        harness.exchange,
        publish_timeout_seconds=3.0,
    )
    harness.dispatcher_factory.assert_called_once_with(
        harness.session_factory,
        harness.publisher,
        batch_size=4,
        lease_duration=timedelta(seconds=30),
        retry_initial_delay=timedelta(seconds=2),
        retry_maximum_delay=timedelta(seconds=8),
    )
    harness.run_loop.assert_awaited_once_with(
        harness.dispatcher.dispatch_once,
        stop_event=stop_event,
        poll_interval_seconds=0.25,
    )
    harness.close_channel.assert_awaited_once_with()
    harness.close_connection.assert_awaited_once_with(harness.connection)
    harness.dispose_engine.assert_awaited_once_with(harness.engine)
    assert harness.events == ["loop", "channel", "connection", "engine"]


@pytest.mark.parametrize(
    ("failure_stage", "expected_cleanup"),
    [
        ("engine", []),
        ("sessions", ["engine"]),
        ("connect", ["engine"]),
        ("channel", ["connection", "engine"]),
        ("topology", ["channel", "connection", "engine"]),
        ("publisher", ["channel", "connection", "engine"]),
        ("dispatcher", ["channel", "connection", "engine"]),
        ("loop", ["channel", "connection", "engine"]),
    ],
)
@pytest.mark.asyncio
async def test_dispatcher_cleans_only_resources_acquired_before_failure(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
    expected_cleanup: list[str],
) -> None:
    """Runtime failures preserve their identity and unwind acquired resources."""

    expected_error = RuntimeError(f"{failure_stage} failed")
    harness = _install_runtime_harness(
        monkeypatch,
        failure_stage=failure_stage,
        expected_error=expected_error,
    )

    with pytest.raises(RuntimeError) as error_info:
        await dispatcher_module.run_dispatcher(
            _settings(),
            stop_event=asyncio.Event(),
        )

    assert error_info.value is expected_error
    assert harness.events == expected_cleanup


@pytest.mark.asyncio
async def test_dispatcher_runs_remaining_cleanup_after_channel_close_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One cleanup failure cannot prevent the other owned resources from closing."""

    harness = _install_runtime_harness(monkeypatch)
    expected_error = RuntimeError("channel close failed")

    async def fail_channel_close() -> None:
        harness.events.append("channel")
        raise expected_error

    harness.close_channel.side_effect = fail_channel_close

    with pytest.raises(RuntimeError) as error_info:
        await dispatcher_module.run_dispatcher(
            _settings(),
            stop_event=asyncio.Event(),
        )

    assert error_info.value is expected_error
    assert harness.events == ["loop", "channel", "connection", "engine"]


@pytest.mark.asyncio
async def test_dispatcher_cleanup_does_not_swallow_task_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling a suspended runtime unwinds resources and keeps task state."""

    harness = _install_runtime_harness(monkeypatch)
    runtime_task = await _start_blocked_runtime(harness)

    try:
        runtime_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await runtime_task
    finally:
        if not runtime_task.done():
            runtime_task.cancel()
            with suppress(asyncio.CancelledError):
                await runtime_task

    assert runtime_task.cancelled()
    assert harness.events == ["channel", "connection", "engine"]


@pytest.mark.asyncio
async def test_dispatcher_surfaces_cleanup_failure_during_task_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken cleanup is reported while retaining cancellation as context."""

    harness = _install_runtime_harness(monkeypatch)
    expected_error = RuntimeError("channel close failed")

    async def fail_channel_close() -> None:
        harness.events.append("channel")
        raise expected_error

    harness.close_channel.side_effect = fail_channel_close
    runtime_task = await _start_blocked_runtime(harness)

    try:
        runtime_task.cancel()

        with pytest.raises(RuntimeError) as error_info:
            await runtime_task
    finally:
        if not runtime_task.done():
            runtime_task.cancel()
            with suppress(asyncio.CancelledError):
                await runtime_task

    assert error_info.value is expected_error
    assert isinstance(expected_error.__context__, asyncio.CancelledError)
    assert runtime_task.done()
    assert not runtime_task.cancelled()
    assert harness.events == ["channel", "connection", "engine"]
