"""Tests for dispatcher process resource composition."""

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
    AbstractRobustChannel,
    AbstractRobustConnection,
    AbstractRobustExchange,
)
from sqlalchemy.ext.asyncio import AsyncEngine

from cims_task_service import dispatcher as dispatcher_module
from cims_task_service.application.execution_retry import ExecutionRetryDelayPolicy
from cims_task_service.application.task_dispatcher import TaskOutboxDispatcher
from cims_task_service.application.task_execution_recovery import TaskExecutionRecovery
from cims_task_service.config import DispatcherSettings
from cims_task_service.dispatcher import (
    DispatcherComponentExitedError,
    DispatcherShutdownTimeoutError,
)
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
    retry_policy: ExecutionRetryDelayPolicy
    recovery: TaskExecutionRecovery
    create_engine: Mock
    create_sessions: Mock
    connect: AsyncMock
    close_connection: AsyncMock
    open_channel: AsyncMock
    close_channel: AsyncMock
    declare_topology: AsyncMock
    publisher_factory: Mock
    dispatcher_factory: Mock
    retry_policy_factory: Mock
    recovery_factory: Mock
    run_loop: AsyncMock
    run_recovery_loop: AsyncMock
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
    retry_policy = cast(ExecutionRetryDelayPolicy, object())
    recover_once = AsyncMock()
    recovery = cast(
        TaskExecutionRecovery,
        SimpleNamespace(recover_once=recover_once),
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
    retry_policy_factory = Mock(return_value=retry_policy)
    recovery_factory = Mock(return_value=recovery)
    run_loop = AsyncMock(side_effect=lambda *_args, **_kwargs: events.append("loop"))
    run_recovery_loop = AsyncMock(
        side_effect=lambda *_args, **_kwargs: events.append("recovery"),
    )
    dispose_engine = AsyncMock(side_effect=lambda _engine: events.append("engine"))

    failure_targets: dict[str, Mock | AsyncMock] = {
        "engine": create_engine,
        "sessions": create_sessions,
        "connect": connect,
        "channel": open_channel,
        "topology": declare_topology,
        "publisher": publisher_factory,
        "dispatcher": dispatcher_factory,
        "retry_policy": retry_policy_factory,
        "recovery": recovery_factory,
        "loop": run_loop,
        "recovery_loop": run_recovery_loop,
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
    monkeypatch.setattr(dispatcher_module, "ExecutionRetryDelayPolicy", retry_policy_factory)
    monkeypatch.setattr(dispatcher_module, "TaskExecutionRecovery", recovery_factory)
    monkeypatch.setattr(dispatcher_module, "run_dispatcher_loop", run_loop)
    monkeypatch.setattr(
        dispatcher_module,
        "run_execution_recovery_loop",
        run_recovery_loop,
    )
    monkeypatch.setattr(dispatcher_module, "dispose_database_engine", dispose_engine)

    return _RuntimeHarness(
        engine=engine,
        session_factory=session_factory,
        connection=connection,
        channel=channel,
        exchange=exchange,
        publisher=publisher,
        dispatcher=dispatcher,
        retry_policy=retry_policy,
        recovery=recovery,
        create_engine=create_engine,
        create_sessions=create_sessions,
        connect=connect,
        close_connection=close_connection,
        open_channel=open_channel,
        close_channel=close_channel,
        declare_topology=declare_topology,
        publisher_factory=publisher_factory,
        dispatcher_factory=dispatcher_factory,
        retry_policy_factory=retry_policy_factory,
        recovery_factory=recovery_factory,
        run_loop=run_loop,
        run_recovery_loop=run_recovery_loop,
        dispose_engine=dispose_engine,
        events=events,
    )


def _settings() -> DispatcherSettings:
    return DispatcherSettings(
        database_pool_size=5,
        database_max_overflow=0,
        database_pool_timeout_seconds=5.0,
        dispatcher_batch_size=4,
        dispatcher_poll_interval_seconds=0.25,
        dispatcher_lease_duration_seconds=30.0,
        dispatcher_retry_initial_delay_seconds=2.0,
        dispatcher_retry_maximum_delay_seconds=8.0,
        execution_recovery_batch_size=7,
        execution_recovery_poll_interval_seconds=0.5,
        execution_retry_initial_delay_seconds=3.0,
        execution_retry_maximum_delay_seconds=12.0,
        rabbitmq_publish_timeout_seconds=3.0,
    )


async def _start_blocked_runtime(harness: _RuntimeHarness) -> asyncio.Task[None]:
    dispatcher_started = asyncio.Event()
    recovery_started = asyncio.Event()

    async def wait_for_dispatcher_cancellation(
        *_args: object,
        **_kwargs: object,
    ) -> None:
        dispatcher_started.set()
        await asyncio.Event().wait()

    async def wait_for_recovery_cancellation(
        *_args: object,
        **_kwargs: object,
    ) -> None:
        recovery_started.set()
        await asyncio.Event().wait()

    harness.run_loop.side_effect = wait_for_dispatcher_cancellation
    harness.run_recovery_loop.side_effect = wait_for_recovery_cancellation
    runtime_task = asyncio.create_task(
        dispatcher_module.run_dispatcher(
            _settings(),
            stop_event=asyncio.Event(),
        )
    )
    try:
        async with asyncio.timeout(1):
            await dispatcher_started.wait()
            await recovery_started.wait()
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
    stop_event.set()

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
    harness.retry_policy_factory.assert_called_once_with(
        initial_delay=timedelta(seconds=3),
        maximum_delay=timedelta(seconds=12),
    )
    harness.recovery_factory.assert_called_once_with(
        harness.session_factory,
        batch_size=7,
        retry_delay_for_attempt=harness.retry_policy,
    )
    harness.run_loop.assert_awaited_once_with(
        harness.dispatcher.dispatch_once,
        stop_event=stop_event,
        poll_interval_seconds=0.25,
    )
    harness.run_recovery_loop.assert_awaited_once_with(
        harness.recovery.recover_once,
        stop_event=stop_event,
        poll_interval_seconds=0.5,
    )
    harness.close_channel.assert_awaited_once_with()
    harness.close_connection.assert_awaited_once_with(harness.connection)
    harness.dispose_engine.assert_awaited_once_with(harness.engine)
    assert harness.events == [
        "loop",
        "recovery",
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
        ("publisher", ["channel", "connection", "engine"]),
        ("dispatcher", ["channel", "connection", "engine"]),
        ("retry_policy", ["channel", "connection", "engine"]),
        ("recovery", ["channel", "connection", "engine"]),
        ("loop", ["recovery", "channel", "connection", "engine"]),
        ("recovery_loop", ["loop", "channel", "connection", "engine"]),
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
    stop_event = asyncio.Event()
    stop_event.set()

    with pytest.raises(RuntimeError) as error_info:
        await dispatcher_module.run_dispatcher(
            _settings(),
            stop_event=stop_event,
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
    stop_event = asyncio.Event()
    stop_event.set()

    with pytest.raises(RuntimeError) as error_info:
        await dispatcher_module.run_dispatcher(
            _settings(),
            stop_event=stop_event,
        )

    assert error_info.value is expected_error
    assert harness.events == [
        "loop",
        "recovery",
        "channel",
        "connection",
        "engine",
    ]


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


@pytest.mark.parametrize(
    ("returning_component", "expected_component_name"),
    [
        ("dispatcher", "outbox dispatcher"),
        ("recovery", "execution recovery"),
    ],
)
@pytest.mark.asyncio
async def test_runtime_rejects_component_exit_before_shutdown(
    returning_component: str,
    expected_component_name: str,
) -> None:
    """Either long-running loop returning early is a process-level failure."""

    sibling_started = asyncio.Event()
    sibling_cancelled = asyncio.Event()

    async def return_after_sibling_starts() -> None:
        await sibling_started.wait()

    async def wait_for_cancellation() -> None:
        sibling_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            sibling_cancelled.set()

    dispatcher_loop = (
        return_after_sibling_starts
        if returning_component == "dispatcher"
        else wait_for_cancellation
    )
    recovery_loop = (
        return_after_sibling_starts if returning_component == "recovery" else wait_for_cancellation
    )

    with pytest.raises(DispatcherComponentExitedError) as error_info:
        await dispatcher_module._run_dispatcher_components(
            dispatcher_loop_factory=dispatcher_loop,
            recovery_loop_factory=recovery_loop,
            stop_event=asyncio.Event(),
        )

    assert error_info.value.component_name == expected_component_name
    assert str(error_info.value) == (
        f"dispatcher component {expected_component_name!r} exited before shutdown was requested"
    )
    assert sibling_cancelled.is_set()


@pytest.mark.parametrize("failing_component", ["dispatcher", "recovery"])
@pytest.mark.asyncio
async def test_runtime_preserves_single_component_failure_and_reaps_sibling(
    failing_component: str,
) -> None:
    """One loop failure keeps its identity and cannot orphan the other loop."""

    sibling_started = asyncio.Event()
    sibling_cancelled = asyncio.Event()
    expected_error = RuntimeError(f"{failing_component} failed")

    async def fail_after_sibling_starts() -> None:
        await sibling_started.wait()
        raise expected_error

    async def wait_for_cancellation() -> None:
        sibling_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            sibling_cancelled.set()

    dispatcher_loop = (
        fail_after_sibling_starts if failing_component == "dispatcher" else wait_for_cancellation
    )
    recovery_loop = (
        fail_after_sibling_starts if failing_component == "recovery" else wait_for_cancellation
    )

    with pytest.raises(RuntimeError) as error_info:
        await dispatcher_module._run_dispatcher_components(
            dispatcher_loop_factory=dispatcher_loop,
            recovery_loop_factory=recovery_loop,
            stop_event=asyncio.Event(),
        )

    assert error_info.value is expected_error
    assert sibling_cancelled.is_set()


@pytest.mark.parametrize(
    ("cancelled_component", "expected_component_name"),
    [
        ("dispatcher", "outbox dispatcher"),
        ("recovery", "execution recovery"),
    ],
)
@pytest.mark.asyncio
async def test_runtime_rejects_unrequested_component_cancellation(
    cancelled_component: str,
    expected_component_name: str,
) -> None:
    """An inner cancellation cannot silently leave the sibling loop running."""

    sibling_started = asyncio.Event()
    sibling_cancelled = asyncio.Event()

    async def cancel_after_sibling_starts() -> None:
        await sibling_started.wait()
        raise asyncio.CancelledError

    async def wait_for_cancellation() -> None:
        sibling_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            sibling_cancelled.set()

    dispatcher_loop = (
        cancel_after_sibling_starts
        if cancelled_component == "dispatcher"
        else wait_for_cancellation
    )
    recovery_loop = (
        cancel_after_sibling_starts if cancelled_component == "recovery" else wait_for_cancellation
    )

    with pytest.raises(DispatcherComponentExitedError) as error_info:
        await dispatcher_module._run_dispatcher_components(
            dispatcher_loop_factory=dispatcher_loop,
            recovery_loop_factory=recovery_loop,
            stop_event=asyncio.Event(),
        )

    assert error_info.value.component_name == expected_component_name
    assert isinstance(error_info.value.__cause__, asyncio.CancelledError)
    assert sibling_cancelled.is_set()


@pytest.mark.asyncio
async def test_runtime_groups_simultaneous_component_failures() -> None:
    """Independent failures from the same event-loop turn remain observable."""

    dispatcher_started = asyncio.Event()
    recovery_started = asyncio.Event()
    dispatcher_error = RuntimeError("dispatcher failed")
    recovery_error = ValueError("recovery failed")

    async def fail_dispatcher() -> None:
        dispatcher_started.set()
        await recovery_started.wait()
        raise dispatcher_error

    async def fail_recovery() -> None:
        recovery_started.set()
        await dispatcher_started.wait()
        raise recovery_error

    with pytest.raises(BaseExceptionGroup) as error_info:
        await dispatcher_module._run_dispatcher_components(
            dispatcher_loop_factory=fail_dispatcher,
            recovery_loop_factory=fail_recovery,
            stop_event=asyncio.Event(),
        )

    assert len(error_info.value.exceptions) == 2
    assert any(error is dispatcher_error for error in error_info.value.exceptions)
    assert any(error is recovery_error for error in error_info.value.exceptions)


@pytest.mark.asyncio
async def test_runtime_external_cancellation_reaps_both_components() -> None:
    """Cancelling the owner cannot leave either background loop running."""

    dispatcher_started = asyncio.Event()
    recovery_started = asyncio.Event()
    dispatcher_cancelled = asyncio.Event()
    recovery_cancelled = asyncio.Event()

    async def run_until_cancelled(
        started: asyncio.Event,
        cancelled: asyncio.Event,
    ) -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    runtime_task = asyncio.create_task(
        dispatcher_module._run_dispatcher_components(
            dispatcher_loop_factory=lambda: run_until_cancelled(
                dispatcher_started,
                dispatcher_cancelled,
            ),
            recovery_loop_factory=lambda: run_until_cancelled(
                recovery_started,
                recovery_cancelled,
            ),
            stop_event=asyncio.Event(),
        ),
    )
    try:
        async with asyncio.timeout(1):
            await dispatcher_started.wait()
            await recovery_started.wait()
        runtime_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await runtime_task
    finally:
        if not runtime_task.done():
            runtime_task.cancel()
            with suppress(asyncio.CancelledError):
                await runtime_task

    assert runtime_task.cancelled()
    assert dispatcher_cancelled.is_set()
    assert recovery_cancelled.is_set()


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

    with dispatcher_module._install_shutdown_signal_handlers(stop_event):
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
        dispatcher_module._install_shutdown_signal_handlers(asyncio.Event()),
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
        dispatcher_module,
        "_install_shutdown_signal_handlers",
        install,
    )


@pytest.mark.asyncio
async def test_supervisor_returns_when_dispatcher_finishes_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finite dispatcher process completes without waiting for a signal."""

    captured_stop_events: list[asyncio.Event] = []
    _install_fake_signal_handlers(
        monkeypatch,
        captured_stop_events,
        request_shutdown=False,
    )
    run_dispatcher = AsyncMock()
    monkeypatch.setattr(dispatcher_module, "run_dispatcher", run_dispatcher)
    settings = _settings()

    await dispatcher_module.supervise_dispatcher(settings)

    run_dispatcher.assert_awaited_once_with(
        settings,
        stop_event=captured_stop_events[0],
    )


@pytest.mark.asyncio
async def test_supervisor_preserves_dispatcher_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup and runtime failures retain their original identity."""

    captured_stop_events: list[asyncio.Event] = []
    _install_fake_signal_handlers(
        monkeypatch,
        captured_stop_events,
        request_shutdown=False,
    )
    expected_error = RuntimeError("dispatcher failed")
    monkeypatch.setattr(
        dispatcher_module,
        "run_dispatcher",
        AsyncMock(side_effect=expected_error),
    )

    with pytest.raises(RuntimeError) as error_info:
        await dispatcher_module.supervise_dispatcher(_settings())

    assert error_info.value is expected_error


@pytest.mark.asyncio
async def test_supervisor_allows_graceful_shutdown_after_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A signal lets the dispatcher observe its stop event and finish normally."""

    captured_stop_events: list[asyncio.Event] = []
    _install_fake_signal_handlers(
        monkeypatch,
        captured_stop_events,
        request_shutdown=True,
    )
    dispatcher_observed_stop = asyncio.Event()
    allow_dispatcher_to_stop = asyncio.Event()

    async def run_dispatcher(
        _settings: DispatcherSettings,
        *,
        stop_event: asyncio.Event,
    ) -> None:
        await stop_event.wait()
        dispatcher_observed_stop.set()
        await allow_dispatcher_to_stop.wait()

    monkeypatch.setattr(dispatcher_module, "run_dispatcher", run_dispatcher)
    supervisor_task = asyncio.create_task(
        dispatcher_module.supervise_dispatcher(_settings()),
    )
    try:
        async with asyncio.timeout(1):
            await dispatcher_observed_stop.wait()
        assert not supervisor_task.done()
        allow_dispatcher_to_stop.set()
        await supervisor_task
    finally:
        if not supervisor_task.done():
            supervisor_task.cancel()
            with suppress(asyncio.CancelledError):
                await supervisor_task

    assert dispatcher_observed_stop.is_set()


@pytest.mark.asyncio
async def test_supervisor_cancels_dispatcher_after_grace_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unresponsive dispatcher is cancelled, reaped, and reported."""

    captured_stop_events: list[asyncio.Event] = []
    _install_fake_signal_handlers(
        monkeypatch,
        captured_stop_events,
        request_shutdown=True,
    )
    dispatcher_cancelled = asyncio.Event()

    async def run_dispatcher(
        _settings: DispatcherSettings,
        *,
        stop_event: asyncio.Event,
    ) -> None:
        del stop_event
        try:
            await asyncio.Event().wait()
        finally:
            dispatcher_cancelled.set()

    monkeypatch.setattr(dispatcher_module, "run_dispatcher", run_dispatcher)
    settings = _settings().model_copy(
        update={"dispatcher_shutdown_grace_seconds": 0.001},
    )

    with pytest.raises(DispatcherShutdownTimeoutError, match=r"0\.001 seconds"):
        await dispatcher_module.supervise_dispatcher(settings)

    assert dispatcher_cancelled.is_set()


@pytest.mark.asyncio
async def test_supervisor_cancels_dispatcher_when_it_is_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """External cancellation cannot orphan the owned dispatcher task."""

    captured_stop_events: list[asyncio.Event] = []
    _install_fake_signal_handlers(
        monkeypatch,
        captured_stop_events,
        request_shutdown=False,
    )
    dispatcher_started = asyncio.Event()
    dispatcher_cancelled = asyncio.Event()

    async def run_dispatcher(
        _settings: DispatcherSettings,
        *,
        stop_event: asyncio.Event,
    ) -> None:
        del stop_event
        dispatcher_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            dispatcher_cancelled.set()

    monkeypatch.setattr(dispatcher_module, "run_dispatcher", run_dispatcher)
    supervisor_task = asyncio.create_task(
        dispatcher_module.supervise_dispatcher(_settings()),
    )
    try:
        async with asyncio.timeout(1):
            await dispatcher_started.wait()
        supervisor_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await supervisor_task
    finally:
        if not supervisor_task.done():
            supervisor_task.cancel()
            with suppress(asyncio.CancelledError):
                await supervisor_task

    assert dispatcher_cancelled.is_set()


@pytest.mark.asyncio
async def test_supervisor_observes_dispatcher_failure_racing_with_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed child failure takes priority over supervisor cancellation."""

    captured_stop_events: list[asyncio.Event] = []
    _install_fake_signal_handlers(
        monkeypatch,
        captured_stop_events,
        request_shutdown=False,
    )
    dispatcher_started = asyncio.Event()
    shutdown_waiter_started = asyncio.Event()
    finish_dispatcher = asyncio.Event()
    expected_error = RuntimeError("dispatcher failed during cancellation")

    async def run_dispatcher(
        _settings: DispatcherSettings,
        *,
        stop_event: asyncio.Event,
    ) -> None:
        del stop_event
        dispatcher_started.set()
        await finish_dispatcher.wait()
        raise expected_error

    async def wait_for_shutdown(stop_event: asyncio.Event) -> None:
        shutdown_waiter_started.set()
        try:
            await stop_event.wait()
        finally:
            finish_dispatcher.set()
            await asyncio.sleep(0)

    monkeypatch.setattr(dispatcher_module, "run_dispatcher", run_dispatcher)
    monkeypatch.setattr(dispatcher_module, "_wait_for_shutdown", wait_for_shutdown)
    supervisor_task = asyncio.create_task(
        dispatcher_module.supervise_dispatcher(_settings()),
    )
    try:
        async with asyncio.timeout(1):
            await dispatcher_started.wait()
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

    async def run_dispatcher(
        _settings: DispatcherSettings,
        *,
        stop_event: asyncio.Event,
    ) -> None:
        del stop_event
        try:
            await asyncio.Event().wait()
        finally:
            raise expected_error

    monkeypatch.setattr(dispatcher_module, "run_dispatcher", run_dispatcher)
    settings = _settings().model_copy(
        update={"dispatcher_shutdown_grace_seconds": 0.001},
    )

    with pytest.raises(RuntimeError) as error_info:
        await dispatcher_module.supervise_dispatcher(settings)

    assert error_info.value is expected_error


def test_main_loads_dispatcher_settings_and_runs_supervisor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The console entry point owns settings loading and the event loop."""

    received_settings: list[DispatcherSettings] = []

    async def supervise_dispatcher(settings: DispatcherSettings) -> None:
        received_settings.append(settings)

    monkeypatch.setattr(
        dispatcher_module,
        "supervise_dispatcher",
        supervise_dispatcher,
    )

    dispatcher_module.main()

    assert len(received_settings) == 1
    assert isinstance(received_settings[0], DispatcherSettings)
