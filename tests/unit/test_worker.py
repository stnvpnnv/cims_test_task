"""Tests for worker process resource composition."""

import asyncio
import signal
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import timedelta
from types import FrameType, SimpleNamespace
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


@pytest.mark.asyncio
async def test_signal_handlers_schedule_shutdown_and_restore_predecessors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Portable handlers keep signal callbacks small and leave no global state."""

    stop_event = asyncio.Event()
    event_loop = asyncio.get_running_loop()
    call_soon_threadsafe = Mock(wraps=event_loop.call_soon_threadsafe)
    monkeypatch.setattr(event_loop, "call_soon_threadsafe", call_soon_threadsafe)

    previous_handlers: dict[signal.Signals, object] = {
        signal.SIGINT: Mock(name="previous_sigint"),
        signal.SIGTERM: Mock(name="previous_sigterm"),
    }
    active_handlers: dict[signal.Signals, object] = dict(previous_handlers)
    replacements: list[tuple[signal.Signals, object]] = []

    def replace_handler(
        signal_number: signal.Signals,
        handler: object,
    ) -> object:
        previous_handler = active_handlers[signal_number]
        active_handlers[signal_number] = handler
        replacements.append((signal_number, handler))
        return previous_handler

    monkeypatch.setattr(signal, "signal", replace_handler)

    with worker_module._install_shutdown_signal_handlers(stop_event):
        installed_sigint = active_handlers[signal.SIGINT]
        installed_sigterm = active_handlers[signal.SIGTERM]
        installed_handler = cast(
            Callable[[int, FrameType | None], None],
            installed_sigterm,
        )
        installed_handler(signal.SIGTERM, None)
        call_soon_threadsafe.assert_called_once_with(stop_event.set)
        assert not stop_event.is_set()
        await asyncio.sleep(0)
        assert stop_event.is_set()

    assert replacements == [
        (signal.SIGINT, installed_sigint),
        (signal.SIGTERM, installed_sigterm),
        (signal.SIGTERM, previous_handlers[signal.SIGTERM]),
        (signal.SIGINT, previous_handlers[signal.SIGINT]),
    ]
    assert active_handlers == previous_handlers


@pytest.mark.asyncio
async def test_signal_handler_registration_rolls_back_partial_installation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later registration failure restores every handler already replaced."""

    previous_sigint = Mock(name="previous_sigint")
    expected_error = ValueError("signals require the main thread")
    replacements: list[tuple[signal.Signals, object]] = []

    def replace_handler(
        signal_number: signal.Signals,
        handler: object,
    ) -> object:
        replacements.append((signal_number, handler))
        if signal_number == signal.SIGTERM:
            raise expected_error
        return previous_sigint

    monkeypatch.setattr(signal, "signal", replace_handler)

    with (
        pytest.raises(ValueError, match="signals require the main thread") as error_info,
        worker_module._install_shutdown_signal_handlers(asyncio.Event()),
    ):
        pytest.fail("registration failure must prevent the runtime from starting")

    assert error_info.value is expected_error
    assert replacements == [
        (signal.SIGINT, replacements[0][1]),
        (signal.SIGTERM, replacements[1][1]),
        (signal.SIGINT, previous_sigint),
    ]


def _install_fake_signal_handlers(
    monkeypatch: pytest.MonkeyPatch,
    captured_stop_events: list[asyncio.Event],
    *,
    request_shutdown: bool,
) -> None:
    @contextmanager
    def install(stop_event: asyncio.Event) -> Iterator[None]:
        captured_stop_events.append(stop_event)
        if request_shutdown:
            asyncio.get_running_loop().call_soon(stop_event.set)
        yield

    monkeypatch.setattr(
        worker_module,
        "_install_shutdown_signal_handlers",
        install,
    )


@pytest.mark.asyncio
async def test_supervisor_returns_when_worker_finishes_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finite worker process completes without waiting for a signal."""

    captured_stop_events: list[asyncio.Event] = []
    _install_fake_signal_handlers(
        monkeypatch,
        captured_stop_events,
        request_shutdown=False,
    )
    run_worker = AsyncMock()
    monkeypatch.setattr(worker_module, "run_worker", run_worker)
    settings = _settings()

    await worker_module.supervise_worker(settings)

    run_worker.assert_awaited_once_with(
        settings,
        stop_event=captured_stop_events[0],
    )


@pytest.mark.asyncio
async def test_supervisor_preserves_worker_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup and runtime failures retain their original identity."""

    captured_stop_events: list[asyncio.Event] = []
    _install_fake_signal_handlers(
        monkeypatch,
        captured_stop_events,
        request_shutdown=False,
    )
    expected_error = RuntimeError("worker failed")
    monkeypatch.setattr(
        worker_module,
        "run_worker",
        AsyncMock(side_effect=expected_error),
    )

    with pytest.raises(RuntimeError) as error_info:
        await worker_module.supervise_worker(_settings())

    assert error_info.value is expected_error


@pytest.mark.asyncio
async def test_supervisor_allows_graceful_shutdown_after_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A signal lets the worker drain active deliveries and finish normally."""

    captured_stop_events: list[asyncio.Event] = []
    _install_fake_signal_handlers(
        monkeypatch,
        captured_stop_events,
        request_shutdown=True,
    )
    worker_observed_stop = asyncio.Event()
    allow_worker_to_stop = asyncio.Event()

    async def run_worker(
        _settings: WorkerSettings,
        *,
        stop_event: asyncio.Event,
    ) -> None:
        await stop_event.wait()
        worker_observed_stop.set()
        await allow_worker_to_stop.wait()

    monkeypatch.setattr(worker_module, "run_worker", run_worker)
    supervisor_task = asyncio.create_task(
        worker_module.supervise_worker(_settings()),
    )
    try:
        async with asyncio.timeout(1):
            await worker_observed_stop.wait()
        assert not supervisor_task.done()
        allow_worker_to_stop.set()
        await supervisor_task
    finally:
        if not supervisor_task.done():
            supervisor_task.cancel()
            with suppress(asyncio.CancelledError):
                await supervisor_task

    assert worker_observed_stop.is_set()


@pytest.mark.asyncio
async def test_supervisor_cancels_worker_after_grace_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unresponsive worker is cancelled, reaped, and reported."""

    captured_stop_events: list[asyncio.Event] = []
    _install_fake_signal_handlers(
        monkeypatch,
        captured_stop_events,
        request_shutdown=True,
    )
    worker_cancelled = asyncio.Event()

    async def run_worker(
        _settings: WorkerSettings,
        *,
        stop_event: asyncio.Event,
    ) -> None:
        del stop_event
        try:
            await asyncio.Event().wait()
        finally:
            worker_cancelled.set()

    monkeypatch.setattr(worker_module, "run_worker", run_worker)
    settings = _settings().model_copy(
        update={"worker_shutdown_grace_seconds": 0.001},
    )

    with pytest.raises(
        worker_module.WorkerShutdownTimeoutError,
        match=r"0\.001 seconds",
    ):
        await worker_module.supervise_worker(settings)

    assert worker_cancelled.is_set()


@pytest.mark.asyncio
async def test_supervisor_cancels_worker_when_it_is_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """External cancellation cannot orphan the owned worker task."""

    captured_stop_events: list[asyncio.Event] = []
    _install_fake_signal_handlers(
        monkeypatch,
        captured_stop_events,
        request_shutdown=False,
    )
    worker_started = asyncio.Event()
    worker_cancelled = asyncio.Event()

    async def run_worker(
        _settings: WorkerSettings,
        *,
        stop_event: asyncio.Event,
    ) -> None:
        del stop_event
        worker_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            worker_cancelled.set()

    monkeypatch.setattr(worker_module, "run_worker", run_worker)
    supervisor_task = asyncio.create_task(
        worker_module.supervise_worker(_settings()),
    )
    try:
        async with asyncio.timeout(1):
            await worker_started.wait()
        supervisor_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await supervisor_task
    finally:
        if not supervisor_task.done():
            supervisor_task.cancel()
            with suppress(asyncio.CancelledError):
                await supervisor_task

    assert worker_cancelled.is_set()


@pytest.mark.asyncio
async def test_supervisor_observes_worker_failure_racing_with_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed child failure takes priority over supervisor cancellation."""

    captured_stop_events: list[asyncio.Event] = []
    _install_fake_signal_handlers(
        monkeypatch,
        captured_stop_events,
        request_shutdown=False,
    )
    worker_started = asyncio.Event()
    shutdown_waiter_started = asyncio.Event()
    finish_worker = asyncio.Event()
    expected_error = RuntimeError("worker failed during cancellation")

    async def run_worker(
        _settings: WorkerSettings,
        *,
        stop_event: asyncio.Event,
    ) -> None:
        del stop_event
        worker_started.set()
        await finish_worker.wait()
        raise expected_error

    async def wait_for_shutdown(stop_event: asyncio.Event) -> None:
        shutdown_waiter_started.set()
        try:
            await stop_event.wait()
        finally:
            finish_worker.set()
            await asyncio.sleep(0)

    monkeypatch.setattr(worker_module, "run_worker", run_worker)
    monkeypatch.setattr(worker_module, "_wait_for_shutdown", wait_for_shutdown)
    supervisor_task = asyncio.create_task(
        worker_module.supervise_worker(_settings()),
    )
    try:
        async with asyncio.timeout(1):
            await worker_started.wait()
            await shutdown_waiter_started.wait()
        supervisor_task.cancel()
        with pytest.raises(RuntimeError) as error_info:
            await supervisor_task
    finally:
        if not supervisor_task.done():
            supervisor_task.cancel()
            with suppress(asyncio.CancelledError):
                await supervisor_task

    assert error_info.value is expected_error


@pytest.mark.asyncio
async def test_supervisor_surfaces_cleanup_failure_after_forced_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup failures remain more important than the shutdown timeout."""

    captured_stop_events: list[asyncio.Event] = []
    _install_fake_signal_handlers(
        monkeypatch,
        captured_stop_events,
        request_shutdown=True,
    )
    expected_error = RuntimeError("cleanup failed")

    async def run_worker(
        _settings: WorkerSettings,
        *,
        stop_event: asyncio.Event,
    ) -> None:
        del stop_event
        try:
            await asyncio.Event().wait()
        finally:
            raise expected_error

    monkeypatch.setattr(worker_module, "run_worker", run_worker)
    settings = _settings().model_copy(
        update={"worker_shutdown_grace_seconds": 0.001},
    )

    with pytest.raises(RuntimeError) as error_info:
        await worker_module.supervise_worker(settings)

    assert error_info.value is expected_error


def test_main_loads_worker_settings_and_runs_supervisor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The console entry point owns settings loading and the event loop."""

    received_settings: list[WorkerSettings] = []

    async def supervise_worker(settings: WorkerSettings) -> None:
        received_settings.append(settings)

    monkeypatch.setattr(worker_module, "supervise_worker", supervise_worker)

    worker_module.main()

    assert len(received_settings) == 1
    assert isinstance(received_settings[0], WorkerSettings)
