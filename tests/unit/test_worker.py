"""Tests for worker process resource composition."""

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from datetime import timedelta
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from aio_pika.abc import (
    AbstractIncomingMessage,
    AbstractRobustChannel,
    AbstractRobustConnection,
    AbstractRobustQueue,
)
from sqlalchemy.ext.asyncio import AsyncEngine

import cims_task_service.worker as worker_module
from cims_task_service.application.execution_retry import ExecutionRetryDelayPolicy
from cims_task_service.application.task_execution import TaskExecutor
from cims_task_service.application.task_processor import TextStatisticsProcessor
from cims_task_service.config import WorkerSettings
from cims_task_service.infrastructure.database.session import AsyncSessionFactory


@dataclass(slots=True)
class _RuntimeHarness:
    engine: AsyncEngine
    session_factory: AsyncSessionFactory
    connection: AbstractRobustConnection
    channel: AbstractRobustChannel
    queue: AbstractRobustQueue
    retry_policy: ExecutionRetryDelayPolicy
    processor: TextStatisticsProcessor
    executor: TaskExecutor
    create_engine: Mock
    create_sessions: Mock
    connect: AsyncMock
    close_connection: AsyncMock
    open_channel: AsyncMock
    close_channel: AsyncMock
    declare_topology: AsyncMock
    retry_policy_factory: Mock
    processor_factory: Mock
    executor_factory: Mock
    handle_delivery: AsyncMock
    run_consumer: AsyncMock
    dispose_engine: AsyncMock
    events: list[str]


@pytest.fixture(autouse=True)
def clear_settings_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep worker settings independent from the developer environment."""

    for field_name in WorkerSettings.model_fields:
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
    close_channel = AsyncMock(side_effect=lambda: events.append("channel"))
    channel = cast(
        AbstractRobustChannel,
        SimpleNamespace(close=close_channel),
    )
    queue = cast(AbstractRobustQueue, object())
    retry_policy = cast(ExecutionRetryDelayPolicy, object())
    processor = cast(TextStatisticsProcessor, object())
    executor = cast(TaskExecutor, object())

    create_engine = Mock(return_value=engine)
    create_sessions = Mock(return_value=session_factory)
    connect = AsyncMock(return_value=connection)
    close_connection = AsyncMock(
        side_effect=lambda _connection: events.append("connection"),
    )
    open_channel = AsyncMock(return_value=channel)
    declare_topology = AsyncMock(
        return_value=SimpleNamespace(task_queue=queue),
    )
    retry_policy_factory = Mock(return_value=retry_policy)
    processor_factory = Mock(return_value=processor)
    executor_factory = Mock(return_value=executor)
    handle_delivery = AsyncMock()
    run_consumer = AsyncMock(side_effect=lambda *_args, **_kwargs: events.append("consumer"))
    dispose_engine = AsyncMock(side_effect=lambda _engine: events.append("engine"))

    failure_targets: dict[str, Mock | AsyncMock] = {
        "engine": create_engine,
        "sessions": create_sessions,
        "connect": connect,
        "channel": open_channel,
        "topology": declare_topology,
        "policy": retry_policy_factory,
        "processor": processor_factory,
        "executor": executor_factory,
        "consumer": run_consumer,
    }
    if failure_stage is not None:
        if expected_error is None:
            raise ValueError("expected_error is required with failure_stage")
        failure_targets[failure_stage].side_effect = expected_error

    monkeypatch.setattr(worker_module, "create_database_engine", create_engine)
    monkeypatch.setattr(worker_module, "create_session_factory", create_sessions)
    monkeypatch.setattr(worker_module, "connect_rabbitmq", connect)
    monkeypatch.setattr(
        worker_module,
        "close_rabbitmq_connection",
        close_connection,
    )
    monkeypatch.setattr(worker_module, "open_consumer_channel", open_channel)
    monkeypatch.setattr(worker_module, "declare_task_topology", declare_topology)
    monkeypatch.setattr(worker_module, "ExecutionRetryDelayPolicy", retry_policy_factory)
    monkeypatch.setattr(worker_module, "TextStatisticsProcessor", processor_factory)
    monkeypatch.setattr(worker_module, "TaskExecutor", executor_factory)
    monkeypatch.setattr(worker_module, "handle_task_delivery", handle_delivery)
    monkeypatch.setattr(worker_module, "run_task_consumer", run_consumer)
    monkeypatch.setattr(worker_module, "dispose_database_engine", dispose_engine)

    return _RuntimeHarness(
        engine=engine,
        session_factory=session_factory,
        connection=connection,
        channel=channel,
        queue=queue,
        retry_policy=retry_policy,
        processor=processor,
        executor=executor,
        create_engine=create_engine,
        create_sessions=create_sessions,
        connect=connect,
        close_connection=close_connection,
        open_channel=open_channel,
        close_channel=close_channel,
        declare_topology=declare_topology,
        retry_policy_factory=retry_policy_factory,
        processor_factory=processor_factory,
        executor_factory=executor_factory,
        handle_delivery=handle_delivery,
        run_consumer=run_consumer,
        dispose_engine=dispose_engine,
        events=events,
    )


def _settings() -> WorkerSettings:
    return WorkerSettings(
        database_pool_size=4,
        database_max_overflow=0,
        database_pool_timeout_seconds=5.0,
        worker_concurrency=4,
        worker_lease_duration_seconds=30.0,
        worker_heartbeat_interval_seconds=5.0,
        worker_processing_timeout_seconds=120.0,
        execution_retry_initial_delay_seconds=3.0,
        execution_retry_maximum_delay_seconds=12.0,
    )


@pytest.mark.asyncio
async def test_worker_builds_runtime_and_closes_resources_in_reverse_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Composition forwards settings, binds delivery handling, and owns resources."""

    harness = _install_runtime_harness(monkeypatch)
    settings = _settings()
    stop_event = asyncio.Event()

    await worker_module.run_worker(settings, stop_event=stop_event)

    harness.create_engine.assert_called_once_with(settings)
    harness.create_sessions.assert_called_once_with(harness.engine)
    harness.connect.assert_awaited_once_with(settings)
    harness.open_channel.assert_awaited_once_with(
        harness.connection,
        prefetch_count=4,
    )
    harness.declare_topology.assert_awaited_once_with(harness.channel)
    harness.retry_policy_factory.assert_called_once_with(
        initial_delay=timedelta(seconds=3),
        maximum_delay=timedelta(seconds=12),
    )
    harness.processor_factory.assert_called_once_with()
    harness.executor_factory.assert_called_once_with(
        harness.session_factory,
        harness.processor,
        lease_duration=timedelta(seconds=30),
        heartbeat_interval=timedelta(seconds=5),
        processing_timeout=timedelta(seconds=120),
        retry_delay_for_attempt=harness.retry_policy,
    )
    harness.run_consumer.assert_awaited_once()
    consumer_call = harness.run_consumer.await_args
    assert consumer_call is not None
    assert consumer_call.args[0] is harness.queue
    delivery_handler = consumer_call.args[1]
    assert consumer_call.kwargs == {
        "concurrency": 4,
        "stop_event": stop_event,
    }

    message = cast(AbstractIncomingMessage, object())
    await delivery_handler(message)

    harness.handle_delivery.assert_awaited_once_with(
        message,
        executor=harness.executor,
    )
    harness.close_channel.assert_awaited_once_with()
    harness.close_connection.assert_awaited_once_with(harness.connection)
    harness.dispose_engine.assert_awaited_once_with(harness.engine)
    assert harness.events == [
        "consumer",
        "channel",
        "connection",
        "engine",
    ]


@pytest.mark.parametrize(
    ("failure_stage", "expected_cleanup"),
    [
        ("engine", []),
        ("sessions", ["engine"]),
        ("connect", ["engine"]),
        ("channel", ["connection", "engine"]),
        ("topology", ["channel", "connection", "engine"]),
        ("policy", ["channel", "connection", "engine"]),
        ("processor", ["channel", "connection", "engine"]),
        ("executor", ["channel", "connection", "engine"]),
        ("consumer", ["channel", "connection", "engine"]),
    ],
)
@pytest.mark.asyncio
async def test_worker_cleans_only_resources_acquired_before_failure(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
    expected_cleanup: list[str],
) -> None:
    """Startup and runtime failures retain identity and unwind acquired resources."""

    expected_error = RuntimeError(f"{failure_stage} failed")
    harness = _install_runtime_harness(
        monkeypatch,
        failure_stage=failure_stage,
        expected_error=expected_error,
    )

    with pytest.raises(RuntimeError) as error_info:
        await worker_module.run_worker(
            _settings(),
            stop_event=asyncio.Event(),
        )

    assert error_info.value is expected_error
    assert harness.events == expected_cleanup


@pytest.mark.asyncio
async def test_worker_cancellation_unwinds_resources_and_remains_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling a running worker closes every owned process resource."""

    harness = _install_runtime_harness(monkeypatch)
    consumer_started = asyncio.Event()

    async def wait_for_cancellation(*_args: object, **_kwargs: object) -> None:
        consumer_started.set()
        await asyncio.Event().wait()

    harness.run_consumer.side_effect = wait_for_cancellation
    worker_task = asyncio.create_task(
        worker_module.run_worker(
            _settings(),
            stop_event=asyncio.Event(),
        )
    )
    try:
        async with asyncio.timeout(1):
            await consumer_started.wait()
        worker_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await worker_task
    finally:
        if not worker_task.done():
            worker_task.cancel()
            with suppress(asyncio.CancelledError):
                await worker_task

    assert worker_task.cancelled()
    assert harness.events == ["channel", "connection", "engine"]
    harness.close_channel.assert_awaited_once_with()
    harness.close_connection.assert_awaited_once_with(harness.connection)
    harness.dispose_engine.assert_awaited_once_with(harness.engine)


@pytest.mark.asyncio
async def test_worker_continues_cleanup_after_channel_close_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A channel cleanup failure cannot orphan the connection or database engine."""

    harness = _install_runtime_harness(monkeypatch)
    expected_error = RuntimeError("channel close failed")

    async def fail_channel_close() -> None:
        harness.events.append("channel")
        raise expected_error

    harness.close_channel.side_effect = fail_channel_close

    with pytest.raises(RuntimeError) as error_info:
        await worker_module.run_worker(
            _settings(),
            stop_event=asyncio.Event(),
        )

    assert error_info.value is expected_error
    assert harness.events == [
        "consumer",
        "channel",
        "connection",
        "engine",
    ]
    harness.close_channel.assert_awaited_once_with()
    harness.close_connection.assert_awaited_once_with(harness.connection)
    harness.dispose_engine.assert_awaited_once_with(harness.engine)
