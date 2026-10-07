"""Tests for one fenced task execution orchestration."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace, TracebackType
from typing import Literal, cast
from unittest.mock import AsyncMock, Mock, call
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from cims_task_service.application import task_execution as task_execution_module
from cims_task_service.application.task_execution import (
    INVALID_PROCESSOR_RESULT_CODE,
    PROCESSING_TIMEOUT_ERROR_CODE,
    TaskExecutionOutcome,
    TaskExecutor,
)
from cims_task_service.application.task_processor import (
    TaskProcessingError,
    TaskProcessingInput,
    TaskProcessor,
)
from cims_task_service.domain.task import TaskPriority
from cims_task_service.infrastructure.database.models import JsonObject, JsonValue
from cims_task_service.infrastructure.database.session import AsyncSessionFactory
from cims_task_service.infrastructure.database.task_execution_repository import (
    ClaimedTaskExecution,
)
from cims_task_service.infrastructure.messaging.topology import TASK_ROUTING_KEY

_TASK_ID = UUID("038d7d21-b1b8-4ddd-a3ce-f9865132669d")
_DISPATCH_TOKEN = UUID("93806a60-eb97-4b07-8244-f3a506479ef8")
_EXECUTION_TOKEN = UUID("a73632c4-4e1d-419f-83f2-e8894ef69a58")
_LEASE_DURATION = timedelta(seconds=60)
_HEARTBEAT_INTERVAL = timedelta(seconds=15)
_PROCESSING_TIMEOUT = timedelta(seconds=300)
_LEASE_EXPIRES_AT = datetime(2026, 9, 25, 2, 30, tzinfo=UTC)

type FinalizationPath = Literal["completion", "retry", "failure"]


@dataclass(frozen=True, slots=True)
class _ExecutionHarness:
    executor: TaskExecutor
    sessions: tuple[AsyncSession, ...]
    transactions: tuple[AsyncMock, ...]
    begin: Mock
    repository_factory: Mock
    claim_for_execution: AsyncMock
    renew_execution_lease: AsyncMock
    process: AsyncMock
    retry_delay_for_attempt: Mock
    complete_execution: AsyncMock
    schedule_execution_retry: AsyncMock
    fail_execution: AsyncMock


def _claimed_execution(
    *,
    priority: TaskPriority = TaskPriority.HIGH,
    attempt_count: int = 1,
    max_attempts: int = 3,
) -> ClaimedTaskExecution:
    return ClaimedTaskExecution(
        task_id=_TASK_ID,
        name="Quarterly report",
        description="Aggregate every region",
        priority=priority,
        attempt_count=attempt_count,
        max_attempts=max_attempts,
        execution_token=_EXECUTION_TOKEN,
        lease_expires_at=_LEASE_EXPIRES_AT,
    )


def _processing_input(claimed: ClaimedTaskExecution) -> TaskProcessingInput:
    return TaskProcessingInput(
        task_id=claimed.task_id,
        name=claimed.name,
        description=claimed.description,
        priority=claimed.priority,
        attempt_count=claimed.attempt_count,
        max_attempts=claimed.max_attempts,
    )


def _transaction(
    session: AsyncSession,
    *,
    on_exit: Callable[[], None] | None = None,
    exit_error: BaseException | None = None,
) -> AsyncMock:
    transaction = AsyncMock()
    transaction.__aenter__.return_value = session

    def exit_transaction(
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> bool:
        if on_exit is not None:
            on_exit()
        if exit_error is not None:
            raise exit_error
        return False

    transaction.__aexit__.side_effect = exit_transaction
    return transaction


def _execution_harness(
    monkeypatch: pytest.MonkeyPatch,
    claimed: ClaimedTaskExecution | None,
    *,
    process: AsyncMock | None = None,
    retry_delay_for_attempt: Mock | None = None,
    complete_result: bool = True,
    retry_result: bool = True,
    failure_result: bool = True,
    heartbeat_interval: timedelta = _HEARTBEAT_INTERVAL,
    processing_timeout: timedelta = _PROCESSING_TIMEOUT,
    heartbeat_transaction_count: int = 0,
    renew_execution_lease: AsyncMock | None = None,
    claim_on_exit: Callable[[], None] | None = None,
    claim_exit_error: BaseException | None = None,
    final_exit_error: BaseException | None = None,
) -> _ExecutionHarness:
    claim_session = cast(AsyncSession, object())
    heartbeat_sessions = tuple(
        cast(AsyncSession, object()) for _ in range(heartbeat_transaction_count)
    )
    final_session = cast(AsyncSession, object())
    sessions = (claim_session, *heartbeat_sessions, final_session)
    transactions = tuple(
        _transaction(
            session,
            on_exit=claim_on_exit if index == 0 else None,
            exit_error=(
                claim_exit_error
                if index == 0
                else final_exit_error
                if index == len(sessions) - 1
                else None
            ),
        )
        for index, session in enumerate(sessions)
    )
    begin = Mock(side_effect=transactions)
    session_factory = cast(AsyncSessionFactory, SimpleNamespace(begin=begin))

    claim_for_execution = AsyncMock(return_value=claimed)
    renew_mock = (
        renew_execution_lease if renew_execution_lease is not None else AsyncMock(return_value=True)
    )
    complete_execution = AsyncMock(return_value=complete_result)
    schedule_execution_retry = AsyncMock(return_value=retry_result)
    fail_execution = AsyncMock(return_value=failure_result)
    claim_repository = SimpleNamespace(claim_for_execution=claim_for_execution)
    heartbeat_repositories = tuple(
        SimpleNamespace(renew_execution_lease=renew_mock) for _ in heartbeat_sessions
    )
    final_repository = SimpleNamespace(
        complete_execution=complete_execution,
        schedule_execution_retry=schedule_execution_retry,
        fail_execution=fail_execution,
    )
    repository_factory = Mock(
        side_effect=(claim_repository, *heartbeat_repositories, final_repository),
    )
    monkeypatch.setattr(
        task_execution_module,
        "TaskExecutionRepository",
        repository_factory,
    )

    process_mock = process if process is not None else AsyncMock(return_value={})
    processor = cast(TaskProcessor, SimpleNamespace(process=process_mock))
    retry_policy = (
        retry_delay_for_attempt
        if retry_delay_for_attempt is not None
        else Mock(return_value=timedelta(seconds=10))
    )
    executor = TaskExecutor(
        session_factory,
        processor,
        lease_duration=_LEASE_DURATION,
        heartbeat_interval=heartbeat_interval,
        processing_timeout=processing_timeout,
        retry_delay_for_attempt=retry_policy,
    )
    return _ExecutionHarness(
        executor=executor,
        sessions=sessions,
        transactions=transactions,
        begin=begin,
        repository_factory=repository_factory,
        claim_for_execution=claim_for_execution,
        renew_execution_lease=renew_mock,
        process=process_mock,
        retry_delay_for_attempt=retry_policy,
        complete_execution=complete_execution,
        schedule_execution_retry=schedule_execution_retry,
        fail_execution=fail_execution,
    )


def _allow_heartbeat_renewals(
    monkeypatch: pytest.MonkeyPatch,
    count: int,
) -> None:
    remaining = count

    async def wait_for_stop(
        stop_event: asyncio.Event,
        *,
        interval_seconds: float,
    ) -> bool:
        nonlocal remaining
        assert interval_seconds == _HEARTBEAT_INTERVAL.total_seconds()
        if remaining > 0:
            remaining -= 1
            return False
        await stop_event.wait()
        return True

    monkeypatch.setattr(
        task_execution_module,
        "_wait_for_heartbeat_stop",
        wait_for_stop,
    )


async def _wait_for_processor_and_deadline_snapshot(
    tasks: tuple[asyncio.Task[object], ...],
    *,
    return_when: str,
) -> tuple[set[asyncio.Task[object]], set[asyncio.Task[object]]]:
    """Deterministically expose a simultaneous processor/deadline completion."""

    assert return_when == asyncio.FIRST_COMPLETED
    supervised = set(tasks)
    processor_task = next(
        task for task in supervised if task.get_name().startswith("task-processor-")
    )
    heartbeat_task = next(
        task for task in supervised if task.get_name().startswith("task-heartbeat-")
    )
    deadline_task = next(iter(supervised - {processor_task, heartbeat_task}))
    await asyncio.gather(processor_task, deadline_task, return_exceptions=True)
    return {processor_task, deadline_task}, {heartbeat_task}


@pytest.mark.parametrize("lease_duration", [timedelta(0), timedelta(microseconds=-1)])
def test_executor_rejects_a_non_positive_lease_before_database_access(
    lease_duration: timedelta,
) -> None:
    """Invalid worker configuration cannot open a transaction or run a processor."""

    begin = Mock()
    process = AsyncMock()

    with pytest.raises(ValueError, match=r"^lease_duration must be positive$"):
        TaskExecutor(
            cast(AsyncSessionFactory, SimpleNamespace(begin=begin)),
            cast(TaskProcessor, SimpleNamespace(process=process)),
            lease_duration=lease_duration,
            heartbeat_interval=_HEARTBEAT_INTERVAL,
            processing_timeout=_PROCESSING_TIMEOUT,
            retry_delay_for_attempt=Mock(),
        )

    begin.assert_not_called()
    process.assert_not_awaited()


@pytest.mark.parametrize(
    ("heartbeat_interval", "message"),
    [
        (timedelta(0), "heartbeat_interval must be positive"),
        (timedelta(microseconds=-1), "heartbeat_interval must be positive"),
        (
            _LEASE_DURATION,
            "heartbeat_interval must be shorter than lease_duration",
        ),
        (
            _LEASE_DURATION + timedelta(microseconds=1),
            "heartbeat_interval must be shorter than lease_duration",
        ),
    ],
)
def test_executor_rejects_an_invalid_heartbeat_before_database_access(
    heartbeat_interval: timedelta,
    message: str,
) -> None:
    """Heartbeat timing must leave a positive renewal window inside the lease."""

    begin = Mock()
    process = AsyncMock()

    with pytest.raises(ValueError, match=rf"^{message}$"):
        TaskExecutor(
            cast(AsyncSessionFactory, SimpleNamespace(begin=begin)),
            cast(TaskProcessor, SimpleNamespace(process=process)),
            lease_duration=_LEASE_DURATION,
            heartbeat_interval=heartbeat_interval,
            processing_timeout=_PROCESSING_TIMEOUT,
            retry_delay_for_attempt=Mock(),
        )

    begin.assert_not_called()
    process.assert_not_awaited()


@pytest.mark.parametrize("processing_timeout", [timedelta(0), timedelta(microseconds=-1)])
def test_executor_rejects_a_non_positive_processing_timeout_before_database_access(
    processing_timeout: timedelta,
) -> None:
    """Invalid processing deadlines cannot start database or processor work."""

    begin = Mock()
    process = AsyncMock()

    with pytest.raises(ValueError, match=r"^processing_timeout must be positive$"):
        TaskExecutor(
            cast(AsyncSessionFactory, SimpleNamespace(begin=begin)),
            cast(TaskProcessor, SimpleNamespace(process=process)),
            lease_duration=_LEASE_DURATION,
            heartbeat_interval=_HEARTBEAT_INTERVAL,
            processing_timeout=processing_timeout,
            retry_delay_for_attempt=Mock(),
        )

    begin.assert_not_called()
    process.assert_not_awaited()


@pytest.mark.asyncio
async def test_success_commits_the_claim_before_processing_and_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Processor I/O stays between two short transactions and receives detached data."""

    claimed = _claimed_execution()
    claim_committed = False
    shared_values: list[JsonValue] = [None, True, 7, 1.25, "данные", {"nested": []}]
    processor_result: JsonObject = {
        "left": shared_values,
        "right": shared_values,
        "empty": {},
    }

    def record_claim_commit() -> None:
        nonlocal claim_committed
        claim_committed = True

    harness = _execution_harness(
        monkeypatch,
        claimed,
        claim_on_exit=record_claim_commit,
    )

    def process_task(task: TaskProcessingInput) -> JsonObject:
        assert claim_committed is True
        assert harness.begin.call_count == 1
        assert task == _processing_input(claimed)
        return processor_result

    harness.process.side_effect = process_task

    outcome = await harness.executor.execute(_TASK_ID, dispatch_token=_DISPATCH_TOKEN)

    assert outcome is TaskExecutionOutcome.COMPLETED
    harness.claim_for_execution.assert_awaited_once_with(
        _TASK_ID,
        dispatch_token=_DISPATCH_TOKEN,
        lease_duration=_LEASE_DURATION,
    )
    harness.process.assert_awaited_once_with(_processing_input(claimed))
    harness.complete_execution.assert_awaited_once_with(
        _TASK_ID,
        execution_token=_EXECUTION_TOKEN,
        result=processor_result,
    )
    complete_call = harness.complete_execution.await_args
    assert complete_call is not None
    persisted_result = cast(JsonObject, complete_call.kwargs["result"])
    assert persisted_result is not processor_result
    assert persisted_result["left"] is not shared_values
    assert persisted_result["right"] is not shared_values
    harness.schedule_execution_retry.assert_not_awaited()
    harness.fail_execution.assert_not_awaited()
    harness.renew_execution_lease.assert_not_awaited()
    assert harness.repository_factory.call_args_list == [
        call(harness.sessions[0]),
        call(harness.sessions[1]),
    ]
    assert harness.sessions[0] is not harness.sessions[1]
    for transaction in harness.transactions:
        transaction.__aexit__.assert_awaited_once_with(None, None, None)


@pytest.mark.asyncio
async def test_heartbeat_renews_the_execution_in_separate_transactions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Long processing renews its fenced lease without holding a database session."""

    _allow_heartbeat_renewals(monkeypatch, 2)
    processor_started = asyncio.Event()
    release_processor = asyncio.Event()
    renewal_count = 0

    async def process_task(_task: TaskProcessingInput) -> JsonObject:
        processor_started.set()
        await release_processor.wait()
        return {"processed": True}

    async def renew_lease(*_args: object, **_kwargs: object) -> bool:
        nonlocal renewal_count
        assert processor_started.is_set()
        renewal_count += 1
        if renewal_count == 2:
            release_processor.set()
        return True

    renew_execution_lease = AsyncMock(side_effect=renew_lease)
    harness = _execution_harness(
        monkeypatch,
        _claimed_execution(),
        process=AsyncMock(side_effect=process_task),
        heartbeat_transaction_count=2,
        renew_execution_lease=renew_execution_lease,
    )

    outcome = await asyncio.wait_for(
        harness.executor.execute(_TASK_ID, dispatch_token=_DISPATCH_TOKEN),
        timeout=1,
    )

    assert outcome is TaskExecutionOutcome.COMPLETED
    assert renew_execution_lease.await_args_list == [
        call(
            _TASK_ID,
            execution_token=_EXECUTION_TOKEN,
            lease_duration=_LEASE_DURATION,
        ),
        call(
            _TASK_ID,
            execution_token=_EXECUTION_TOKEN,
            lease_duration=_LEASE_DURATION,
        ),
    ]
    assert harness.repository_factory.call_args_list == [
        call(harness.sessions[0]),
        call(harness.sessions[1]),
        call(harness.sessions[2]),
        call(harness.sessions[3]),
    ]
    assert len({id(session) for session in harness.sessions}) == 4
    for transaction in harness.transactions:
        transaction.__aexit__.assert_awaited_once_with(None, None, None)


@pytest.mark.asyncio
async def test_finalization_waits_for_an_in_flight_heartbeat_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker stops and reaps renewal before it writes a terminal outcome."""

    _allow_heartbeat_renewals(monkeypatch, 1)
    release_processor = asyncio.Event()
    heartbeat_commit_started = asyncio.Event()
    allow_heartbeat_commit = asyncio.Event()
    heartbeat_committed = asyncio.Event()

    async def process_task(_task: TaskProcessingInput) -> JsonObject:
        await release_processor.wait()
        return {"processed": True}

    async def renew_lease(*_args: object, **_kwargs: object) -> bool:
        release_processor.set()
        return True

    harness = _execution_harness(
        monkeypatch,
        _claimed_execution(),
        process=AsyncMock(side_effect=process_task),
        heartbeat_transaction_count=1,
        renew_execution_lease=AsyncMock(side_effect=renew_lease),
    )

    async def commit_heartbeat(
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> bool:
        heartbeat_commit_started.set()
        await allow_heartbeat_commit.wait()
        heartbeat_committed.set()
        return False

    harness.transactions[1].__aexit__.side_effect = commit_heartbeat

    def complete_after_heartbeat(*_args: object, **_kwargs: object) -> bool:
        assert heartbeat_committed.is_set()
        return True

    harness.complete_execution.side_effect = complete_after_heartbeat
    execution = asyncio.create_task(
        harness.executor.execute(_TASK_ID, dispatch_token=_DISPATCH_TOKEN),
    )

    await asyncio.wait_for(heartbeat_commit_started.wait(), timeout=1)
    await asyncio.sleep(0)
    harness.complete_execution.assert_not_awaited()
    assert not execution.done()

    allow_heartbeat_commit.set()
    outcome = await asyncio.wait_for(execution, timeout=1)

    assert outcome is TaskExecutionOutcome.COMPLETED
    harness.transactions[1].__aexit__.assert_awaited_once_with(None, None, None)
    harness.transactions[2].__aenter__.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_processing_timeout_keeps_heartbeat_active_during_processor_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Timed-out work is reaped before heartbeat shutdown and retry finalization."""

    processor_started = asyncio.Event()
    processor_cleanup_started = asyncio.Event()
    allow_processor_cleanup = asyncio.Event()
    processor_reaped = asyncio.Event()
    heartbeat_renewed_during_cleanup = asyncio.Event()
    heartbeat_stop_requested = asyncio.Event()
    never_finish = asyncio.Event()
    heartbeat_wait_count = 0

    async def process_task(_task: TaskProcessingInput) -> JsonObject:
        processor_started.set()
        try:
            await never_finish.wait()
        except asyncio.CancelledError:
            processor_cleanup_started.set()
            await allow_processor_cleanup.wait()
            raise
        finally:
            processor_reaped.set()
        raise AssertionError("unreachable")

    async def wait_for_deadline(*, timeout_seconds: float) -> None:
        assert timeout_seconds == _PROCESSING_TIMEOUT.total_seconds()
        await processor_started.wait()

    async def wait_for_heartbeat_stop(
        stop_event: asyncio.Event,
        *,
        interval_seconds: float,
    ) -> bool:
        nonlocal heartbeat_wait_count
        assert interval_seconds == _HEARTBEAT_INTERVAL.total_seconds()
        heartbeat_wait_count += 1
        if heartbeat_wait_count > 1:
            await stop_event.wait()
            return True

        set_stop_event = stop_event.set

        def record_heartbeat_stop() -> None:
            set_stop_event()
            heartbeat_stop_requested.set()

        monkeypatch.setattr(stop_event, "set", record_heartbeat_stop)
        await processor_cleanup_started.wait()
        return False

    async def renew_lease(*_args: object, **_kwargs: object) -> bool:
        assert processor_cleanup_started.is_set()
        assert not processor_reaped.is_set()
        heartbeat_renewed_during_cleanup.set()
        allow_processor_cleanup.set()
        return True

    monkeypatch.setattr(
        task_execution_module,
        "_wait_for_processing_deadline",
        wait_for_deadline,
    )
    monkeypatch.setattr(
        task_execution_module,
        "_wait_for_heartbeat_stop",
        wait_for_heartbeat_stop,
    )
    harness = _execution_harness(
        monkeypatch,
        _claimed_execution(),
        process=AsyncMock(side_effect=process_task),
        heartbeat_transaction_count=1,
        renew_execution_lease=AsyncMock(side_effect=renew_lease),
    )

    def schedule_after_supervised_shutdown(*_args: object, **_kwargs: object) -> bool:
        assert heartbeat_renewed_during_cleanup.is_set()
        assert processor_reaped.is_set()
        assert heartbeat_stop_requested.is_set()
        return True

    harness.schedule_execution_retry.side_effect = schedule_after_supervised_shutdown

    outcome = await asyncio.wait_for(
        harness.executor.execute(_TASK_ID, dispatch_token=_DISPATCH_TOKEN),
        timeout=1,
    )

    assert outcome is TaskExecutionOutcome.RETRY_SCHEDULED
    harness.retry_delay_for_attempt.assert_called_once_with(1)
    harness.schedule_execution_retry.assert_awaited_once_with(
        _TASK_ID,
        execution_token=_EXECUTION_TOKEN,
        retry_delay=timedelta(seconds=10),
        event_type=TASK_ROUTING_KEY,
        message_priority=3,
    )
    harness.complete_execution.assert_not_awaited()
    harness.fail_execution.assert_not_awaited()


@pytest.mark.asyncio
async def test_lost_renewal_during_timeout_cleanup_prevents_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ownership loss during cancellation outranks the pending timeout outcome."""

    processor_started = asyncio.Event()
    processor_cleanup_started = asyncio.Event()
    allow_processor_cleanup = asyncio.Event()
    processor_reaped = asyncio.Event()
    never_finish = asyncio.Event()
    heartbeat_wait_count = 0

    async def process_task(_task: TaskProcessingInput) -> JsonObject:
        processor_started.set()
        try:
            await never_finish.wait()
        except asyncio.CancelledError:
            processor_cleanup_started.set()
            await allow_processor_cleanup.wait()
            raise
        finally:
            processor_reaped.set()
        raise AssertionError("unreachable")

    async def wait_for_deadline(*, timeout_seconds: float) -> None:
        assert timeout_seconds == _PROCESSING_TIMEOUT.total_seconds()
        await processor_started.wait()

    async def wait_for_heartbeat_stop(
        stop_event: asyncio.Event,
        *,
        interval_seconds: float,
    ) -> bool:
        nonlocal heartbeat_wait_count
        assert interval_seconds == _HEARTBEAT_INTERVAL.total_seconds()
        heartbeat_wait_count += 1
        if heartbeat_wait_count == 1:
            await processor_cleanup_started.wait()
            return False
        await stop_event.wait()
        return True

    async def lose_ownership(*_args: object, **_kwargs: object) -> bool:
        assert processor_cleanup_started.is_set()
        allow_processor_cleanup.set()
        return False

    monkeypatch.setattr(
        task_execution_module,
        "_wait_for_processing_deadline",
        wait_for_deadline,
    )
    monkeypatch.setattr(
        task_execution_module,
        "_wait_for_heartbeat_stop",
        wait_for_heartbeat_stop,
    )
    harness = _execution_harness(
        monkeypatch,
        _claimed_execution(),
        process=AsyncMock(side_effect=process_task),
        heartbeat_transaction_count=1,
        renew_execution_lease=AsyncMock(side_effect=lose_ownership),
    )

    outcome = await asyncio.wait_for(
        harness.executor.execute(_TASK_ID, dispatch_token=_DISPATCH_TOKEN),
        timeout=1,
    )

    assert outcome is TaskExecutionOutcome.LOST_OWNERSHIP
    assert processor_reaped.is_set()
    harness.complete_execution.assert_not_awaited()
    harness.schedule_execution_retry.assert_not_awaited()
    harness.fail_execution.assert_not_awaited()


@pytest.mark.asyncio
async def test_processor_cleanup_failure_outranks_processing_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A processor cancellation defect remains visible instead of becoming a retry."""

    cleanup_error = RuntimeError("processor timeout cleanup failed")
    processor_started = asyncio.Event()
    never_finish = asyncio.Event()

    async def process_task(_task: TaskProcessingInput) -> JsonObject:
        processor_started.set()
        try:
            await never_finish.wait()
        finally:
            raise cleanup_error

    async def wait_for_deadline(*, timeout_seconds: float) -> None:
        assert timeout_seconds == _PROCESSING_TIMEOUT.total_seconds()
        await processor_started.wait()

    monkeypatch.setattr(
        task_execution_module,
        "_wait_for_processing_deadline",
        wait_for_deadline,
    )
    harness = _execution_harness(
        monkeypatch,
        _claimed_execution(),
        process=AsyncMock(side_effect=process_task),
    )

    with pytest.raises(
        RuntimeError,
        match=r"^processor timeout cleanup failed$",
    ) as error_info:
        await harness.executor.execute(_TASK_ID, dispatch_token=_DISPATCH_TOKEN)

    assert error_info.value is cleanup_error
    harness.retry_delay_for_attempt.assert_not_called()
    harness.complete_execution.assert_not_awaited()
    harness.schedule_execution_retry.assert_not_awaited()
    harness.fail_execution.assert_not_awaited()


@pytest.mark.asyncio
async def test_heartbeat_failure_during_timeout_cleanup_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A renewal error during cancellation remains an infrastructure failure."""

    expected_error = OSError("timeout cleanup heartbeat query failed")
    processor_started = asyncio.Event()
    processor_cleanup_started = asyncio.Event()
    allow_processor_cleanup = asyncio.Event()
    processor_reaped = asyncio.Event()
    never_finish = asyncio.Event()
    heartbeat_wait_count = 0

    async def process_task(_task: TaskProcessingInput) -> JsonObject:
        processor_started.set()
        try:
            await never_finish.wait()
        except asyncio.CancelledError:
            processor_cleanup_started.set()
            await allow_processor_cleanup.wait()
            raise
        finally:
            processor_reaped.set()
        raise AssertionError("unreachable")

    async def wait_for_deadline(*, timeout_seconds: float) -> None:
        assert timeout_seconds == _PROCESSING_TIMEOUT.total_seconds()
        await processor_started.wait()

    async def wait_for_heartbeat_stop(
        stop_event: asyncio.Event,
        *,
        interval_seconds: float,
    ) -> bool:
        nonlocal heartbeat_wait_count
        assert interval_seconds == _HEARTBEAT_INTERVAL.total_seconds()
        heartbeat_wait_count += 1
        if heartbeat_wait_count == 1:
            await processor_cleanup_started.wait()
            return False
        await stop_event.wait()
        return True

    async def fail_renewal(*_args: object, **_kwargs: object) -> bool:
        assert processor_cleanup_started.is_set()
        allow_processor_cleanup.set()
        raise expected_error

    monkeypatch.setattr(
        task_execution_module,
        "_wait_for_processing_deadline",
        wait_for_deadline,
    )
    monkeypatch.setattr(
        task_execution_module,
        "_wait_for_heartbeat_stop",
        wait_for_heartbeat_stop,
    )
    harness = _execution_harness(
        monkeypatch,
        _claimed_execution(),
        process=AsyncMock(side_effect=process_task),
        heartbeat_transaction_count=1,
        renew_execution_lease=AsyncMock(side_effect=fail_renewal),
    )

    with pytest.raises(
        OSError,
        match=r"^timeout cleanup heartbeat query failed$",
    ) as error_info:
        await asyncio.wait_for(
            harness.executor.execute(_TASK_ID, dispatch_token=_DISPATCH_TOKEN),
            timeout=1,
        )

    assert error_info.value is expected_error
    assert processor_reaped.is_set()
    harness.complete_execution.assert_not_awaited()
    harness.schedule_execution_retry.assert_not_awaited()
    harness.fail_execution.assert_not_awaited()


@pytest.mark.asyncio
async def test_normal_heartbeat_stop_during_timeout_cleanup_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An impossible heartbeat stop during cancellation is not a timeout outcome."""

    processor_started = asyncio.Event()
    processor_cleanup_started = asyncio.Event()
    allow_processor_cleanup = asyncio.Event()
    processor_reaped = asyncio.Event()
    never_finish = asyncio.Event()

    async def process_task(_task: TaskProcessingInput) -> JsonObject:
        processor_started.set()
        try:
            await never_finish.wait()
        except asyncio.CancelledError:
            processor_cleanup_started.set()
            await allow_processor_cleanup.wait()
            await asyncio.sleep(0)
            raise
        finally:
            processor_reaped.set()
        raise AssertionError("unreachable")

    async def wait_for_deadline(*, timeout_seconds: float) -> None:
        assert timeout_seconds == _PROCESSING_TIMEOUT.total_seconds()
        await processor_started.wait()

    async def stop_heartbeat_during_cleanup(
        _stop_event: asyncio.Event,
        *,
        interval_seconds: float,
    ) -> bool:
        assert interval_seconds == _HEARTBEAT_INTERVAL.total_seconds()
        await processor_cleanup_started.wait()
        allow_processor_cleanup.set()
        return True

    monkeypatch.setattr(
        task_execution_module,
        "_wait_for_processing_deadline",
        wait_for_deadline,
    )
    monkeypatch.setattr(
        task_execution_module,
        "_wait_for_heartbeat_stop",
        stop_heartbeat_during_cleanup,
    )
    harness = _execution_harness(
        monkeypatch,
        _claimed_execution(),
        process=AsyncMock(side_effect=process_task),
    )

    with pytest.raises(
        RuntimeError,
        match=r"^execution heartbeat stopped before processing finished$",
    ):
        await harness.executor.execute(_TASK_ID, dispatch_token=_DISPATCH_TOKEN)

    assert processor_reaped.is_set()
    harness.renew_execution_lease.assert_not_awaited()
    harness.complete_execution.assert_not_awaited()
    harness.schedule_execution_retry.assert_not_awaited()
    harness.fail_execution.assert_not_awaited()


@pytest.mark.asyncio
async def test_exhausted_processing_timeout_is_persisted_as_retryable_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Attempt exhaustion retains the timeout classification without another retry."""

    processor_started = asyncio.Event()
    processor_reaped = asyncio.Event()
    never_finish = asyncio.Event()

    async def process_task(_task: TaskProcessingInput) -> JsonObject:
        processor_started.set()
        try:
            await never_finish.wait()
        finally:
            processor_reaped.set()
        raise AssertionError("unreachable")

    async def wait_for_deadline(*, timeout_seconds: float) -> None:
        assert timeout_seconds == _PROCESSING_TIMEOUT.total_seconds()
        await processor_started.wait()

    monkeypatch.setattr(
        task_execution_module,
        "_wait_for_processing_deadline",
        wait_for_deadline,
    )
    harness = _execution_harness(
        monkeypatch,
        _claimed_execution(attempt_count=3, max_attempts=3),
        process=AsyncMock(side_effect=process_task),
    )

    outcome = await asyncio.wait_for(
        harness.executor.execute(_TASK_ID, dispatch_token=_DISPATCH_TOKEN),
        timeout=1,
    )

    assert outcome is TaskExecutionOutcome.FAILED
    assert processor_reaped.is_set()
    harness.fail_execution.assert_awaited_once_with(
        _TASK_ID,
        execution_token=_EXECUTION_TOKEN,
        error={"code": PROCESSING_TIMEOUT_ERROR_CODE, "retryable": True},
    )
    harness.retry_delay_for_attempt.assert_not_called()
    harness.complete_execution.assert_not_awaited()
    harness.schedule_execution_retry.assert_not_awaited()


@pytest.mark.asyncio
async def test_processor_timeout_error_propagates_without_becoming_our_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A processor's own TimeoutError remains an unexpected infrastructure error."""

    expected_error = TimeoutError("processor dependency timed out")
    harness = _execution_harness(
        monkeypatch,
        _claimed_execution(),
        process=AsyncMock(side_effect=expected_error),
    )

    with pytest.raises(TimeoutError, match=r"^processor dependency timed out$") as error_info:
        await harness.executor.execute(_TASK_ID, dispatch_token=_DISPATCH_TOKEN)

    assert error_info.value is expected_error
    harness.retry_delay_for_attempt.assert_not_called()
    harness.complete_execution.assert_not_awaited()
    harness.schedule_execution_retry.assert_not_awaited()
    harness.fail_execution.assert_not_awaited()


@pytest.mark.asyncio
async def test_simultaneous_processor_and_deadline_completion_prefers_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A result already ready in the same wait snapshot is not misclassified as timeout."""

    async def finish_deadline(*, timeout_seconds: float) -> None:
        assert timeout_seconds == _PROCESSING_TIMEOUT.total_seconds()

    monkeypatch.setattr(
        task_execution_module,
        "_wait_for_processing_deadline",
        finish_deadline,
    )
    monkeypatch.setattr(
        asyncio,
        "wait",
        _wait_for_processor_and_deadline_snapshot,
    )
    harness = _execution_harness(
        monkeypatch,
        _claimed_execution(),
        process=AsyncMock(return_value={"processed": True}),
    )

    outcome = await harness.executor.execute(_TASK_ID, dispatch_token=_DISPATCH_TOKEN)

    assert outcome is TaskExecutionOutcome.COMPLETED
    harness.complete_execution.assert_awaited_once_with(
        _TASK_ID,
        execution_token=_EXECUTION_TOKEN,
        result={"processed": True},
    )
    harness.retry_delay_for_attempt.assert_not_called()
    harness.schedule_execution_retry.assert_not_awaited()
    harness.fail_execution.assert_not_awaited()


@pytest.mark.asyncio
async def test_simultaneous_processor_failure_and_deadline_propagates_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected processor defects outrank a simultaneous business timeout."""

    expected_error = RuntimeError("processor defect")

    async def finish_deadline(*, timeout_seconds: float) -> None:
        assert timeout_seconds == _PROCESSING_TIMEOUT.total_seconds()

    monkeypatch.setattr(
        task_execution_module,
        "_wait_for_processing_deadline",
        finish_deadline,
    )
    monkeypatch.setattr(
        asyncio,
        "wait",
        _wait_for_processor_and_deadline_snapshot,
    )
    harness = _execution_harness(
        monkeypatch,
        _claimed_execution(),
        process=AsyncMock(side_effect=expected_error),
    )

    with pytest.raises(RuntimeError, match=r"^processor defect$") as error_info:
        await harness.executor.execute(_TASK_ID, dispatch_token=_DISPATCH_TOKEN)

    assert error_info.value is expected_error
    harness.retry_delay_for_attempt.assert_not_called()
    harness.complete_execution.assert_not_awaited()
    harness.schedule_execution_retry.assert_not_awaited()
    harness.fail_execution.assert_not_awaited()


@pytest.mark.asyncio
async def test_simultaneous_heartbeat_loss_and_deadline_prefers_ownership_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale worker cannot schedule a timeout retry from the same wait snapshot."""

    processor_started = asyncio.Event()
    processor_reaped = asyncio.Event()
    never_finish = asyncio.Event()

    async def process_task(_task: TaskProcessingInput) -> JsonObject:
        processor_started.set()
        try:
            await never_finish.wait()
        finally:
            processor_reaped.set()
        raise AssertionError("unreachable")

    async def finish_deadline(*, timeout_seconds: float) -> None:
        assert timeout_seconds == _PROCESSING_TIMEOUT.total_seconds()
        await processor_started.wait()

    async def wait_for_heartbeat_and_deadline(
        tasks: tuple[asyncio.Task[object], ...],
        *,
        return_when: str,
    ) -> tuple[set[asyncio.Task[object]], set[asyncio.Task[object]]]:
        assert return_when == asyncio.FIRST_COMPLETED
        supervised = set(tasks)
        processor_task = next(
            task for task in supervised if task.get_name().startswith("task-processor-")
        )
        heartbeat_task = next(
            task for task in supervised if task.get_name().startswith("task-heartbeat-")
        )
        deadline_task = next(iter(supervised - {processor_task, heartbeat_task}))
        await asyncio.gather(heartbeat_task, deadline_task)
        return {heartbeat_task, deadline_task}, {processor_task}

    _allow_heartbeat_renewals(monkeypatch, 1)
    monkeypatch.setattr(
        task_execution_module,
        "_wait_for_processing_deadline",
        finish_deadline,
    )
    monkeypatch.setattr(asyncio, "wait", wait_for_heartbeat_and_deadline)
    harness = _execution_harness(
        monkeypatch,
        _claimed_execution(),
        process=AsyncMock(side_effect=process_task),
        heartbeat_transaction_count=1,
        renew_execution_lease=AsyncMock(return_value=False),
    )

    outcome = await harness.executor.execute(_TASK_ID, dispatch_token=_DISPATCH_TOKEN)

    assert outcome is TaskExecutionOutcome.LOST_OWNERSHIP
    assert processor_reaped.is_set()
    harness.complete_execution.assert_not_awaited()
    harness.schedule_execution_retry.assert_not_awaited()
    harness.fail_execution.assert_not_awaited()


@pytest.mark.asyncio
async def test_lost_heartbeat_ownership_cancels_and_reaps_the_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed fenced renewal wins the race and prevents stale finalization."""

    _allow_heartbeat_renewals(monkeypatch, 1)
    processor_started = asyncio.Event()
    processor_reaped = asyncio.Event()
    never_finish = asyncio.Event()

    async def process_task(_task: TaskProcessingInput) -> JsonObject:
        processor_started.set()
        try:
            await never_finish.wait()
        finally:
            processor_reaped.set()
        raise AssertionError("unreachable")

    renew_execution_lease = AsyncMock(return_value=False)
    harness = _execution_harness(
        monkeypatch,
        _claimed_execution(),
        process=AsyncMock(side_effect=process_task),
        heartbeat_transaction_count=1,
        renew_execution_lease=renew_execution_lease,
    )

    outcome = await asyncio.wait_for(
        harness.executor.execute(_TASK_ID, dispatch_token=_DISPATCH_TOKEN),
        timeout=1,
    )

    assert outcome is TaskExecutionOutcome.LOST_OWNERSHIP
    assert processor_started.is_set()
    assert processor_reaped.is_set()
    renew_execution_lease.assert_awaited_once_with(
        _TASK_ID,
        execution_token=_EXECUTION_TOKEN,
        lease_duration=_LEASE_DURATION,
    )
    harness.complete_execution.assert_not_awaited()
    harness.schedule_execution_retry.assert_not_awaited()
    harness.fail_execution.assert_not_awaited()
    assert harness.begin.call_count == 2
    harness.transactions[2].__aenter__.assert_not_awaited()


@pytest.mark.asyncio
async def test_lost_ownership_preserves_a_processor_error_after_wait_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup retains an error from a task that finished after wait classified it pending."""

    _allow_heartbeat_renewals(monkeypatch, 1)
    cleanup_error = RuntimeError("processor cancellation cleanup failed")
    processor_started = asyncio.Event()
    never_finish = asyncio.Event()

    async def process_task(_task: TaskProcessingInput) -> JsonObject:
        processor_started.set()
        try:
            await never_finish.wait()
        finally:
            raise cleanup_error

    original_wait = asyncio.wait

    async def wait_after_pending_finishes(
        tasks: tuple[asyncio.Task[object], ...],
        *,
        return_when: str,
    ) -> tuple[set[asyncio.Task[object]], set[asyncio.Task[object]]]:
        supervised = set(tasks)
        completed, pending = await original_wait(
            supervised,
            return_when=return_when,
        )
        assert len(completed) == 1
        assert len(pending) == 2
        processor_task = next(
            task for task in pending if task.get_name().startswith("task-processor-")
        )
        processor_task.cancel()
        await asyncio.gather(processor_task, return_exceptions=True)
        return completed, pending

    monkeypatch.setattr(asyncio, "wait", wait_after_pending_finishes)
    harness = _execution_harness(
        monkeypatch,
        _claimed_execution(),
        process=AsyncMock(side_effect=process_task),
        heartbeat_transaction_count=1,
        renew_execution_lease=AsyncMock(return_value=False),
    )

    async with asyncio.timeout(1):
        with pytest.raises(
            RuntimeError,
            match=r"^processor cancellation cleanup failed$",
        ) as error_info:
            await harness.executor.execute(_TASK_ID, dispatch_token=_DISPATCH_TOKEN)

    assert error_info.value is cleanup_error
    assert processor_started.is_set()
    harness.complete_execution.assert_not_awaited()
    harness.schedule_execution_retry.assert_not_awaited()
    harness.fail_execution.assert_not_awaited()


@pytest.mark.asyncio
async def test_heartbeat_query_failure_cancels_processor_and_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A renewal database error fails fast and leaves the execution to recovery."""

    _allow_heartbeat_renewals(monkeypatch, 1)
    expected_error = OSError("heartbeat query failed")
    processor_reaped = asyncio.Event()
    never_finish = asyncio.Event()

    async def process_task(_task: TaskProcessingInput) -> JsonObject:
        try:
            await never_finish.wait()
        finally:
            processor_reaped.set()
        raise AssertionError("unreachable")

    harness = _execution_harness(
        monkeypatch,
        _claimed_execution(),
        process=AsyncMock(side_effect=process_task),
        heartbeat_transaction_count=1,
        renew_execution_lease=AsyncMock(side_effect=expected_error),
    )

    with pytest.raises(OSError, match=r"^heartbeat query failed$") as error_info:
        await asyncio.wait_for(
            harness.executor.execute(_TASK_ID, dispatch_token=_DISPATCH_TOKEN),
            timeout=1,
        )

    assert error_info.value is expected_error
    assert processor_reaped.is_set()
    heartbeat_exit = harness.transactions[1].__aexit__.await_args
    assert heartbeat_exit is not None
    assert heartbeat_exit.args[0] is OSError
    assert heartbeat_exit.args[1] is expected_error
    harness.complete_execution.assert_not_awaited()
    harness.schedule_execution_retry.assert_not_awaited()
    harness.fail_execution.assert_not_awaited()


@pytest.mark.asyncio
async def test_heartbeat_and_processor_cleanup_failures_are_grouped_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent supervisor failures retain identity without duplicate group leaves."""

    _allow_heartbeat_renewals(monkeypatch, 1)
    heartbeat_error = OSError("heartbeat query failed")
    processor_cleanup_error = RuntimeError("processor cancellation cleanup failed")
    processor_started = asyncio.Event()
    never_finish = asyncio.Event()

    async def process_task(_task: TaskProcessingInput) -> JsonObject:
        processor_started.set()
        try:
            await never_finish.wait()
        finally:
            raise processor_cleanup_error

    harness = _execution_harness(
        monkeypatch,
        _claimed_execution(),
        process=AsyncMock(side_effect=process_task),
        heartbeat_transaction_count=1,
        renew_execution_lease=AsyncMock(side_effect=heartbeat_error),
    )

    with pytest.raises(BaseExceptionGroup) as error_info:
        await asyncio.wait_for(
            harness.executor.execute(_TASK_ID, dispatch_token=_DISPATCH_TOKEN),
            timeout=1,
        )

    grouped_errors = error_info.value.exceptions
    assert grouped_errors == (heartbeat_error, processor_cleanup_error)
    assert processor_started.is_set()
    harness.complete_execution.assert_not_awaited()
    harness.schedule_execution_retry.assert_not_awaited()
    harness.fail_execution.assert_not_awaited()


@pytest.mark.asyncio
async def test_heartbeat_commit_failure_cancels_processor_and_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An uncertain renewal commit cannot be treated as retained ownership."""

    _allow_heartbeat_renewals(monkeypatch, 1)
    expected_error = OSError("heartbeat commit outcome unknown")
    processor_reaped = asyncio.Event()
    never_finish = asyncio.Event()

    async def process_task(_task: TaskProcessingInput) -> JsonObject:
        try:
            await never_finish.wait()
        finally:
            processor_reaped.set()
        raise AssertionError("unreachable")

    harness = _execution_harness(
        monkeypatch,
        _claimed_execution(),
        process=AsyncMock(side_effect=process_task),
        heartbeat_transaction_count=1,
    )
    harness.transactions[1].__aexit__.side_effect = expected_error

    with pytest.raises(
        OSError,
        match=r"^heartbeat commit outcome unknown$",
    ) as error_info:
        await asyncio.wait_for(
            harness.executor.execute(_TASK_ID, dispatch_token=_DISPATCH_TOKEN),
            timeout=1,
        )

    assert error_info.value is expected_error
    assert processor_reaped.is_set()
    harness.renew_execution_lease.assert_awaited_once()
    harness.complete_execution.assert_not_awaited()
    harness.schedule_execution_retry.assert_not_awaited()
    harness.fail_execution.assert_not_awaited()


@pytest.mark.asyncio
async def test_external_cancellation_reaps_processor_and_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shutdown cancellation leaves no owned child task or durable finalization."""

    _allow_heartbeat_renewals(monkeypatch, 1)
    processor_started = asyncio.Event()
    processor_reaped = asyncio.Event()
    heartbeat_started = asyncio.Event()
    heartbeat_reaped = asyncio.Event()
    deadline_started = asyncio.Event()
    deadline_reaped = asyncio.Event()
    never_finish = asyncio.Event()

    async def process_task(_task: TaskProcessingInput) -> JsonObject:
        processor_started.set()
        try:
            await never_finish.wait()
        finally:
            processor_reaped.set()
        raise AssertionError("unreachable")

    async def renew_lease(*_args: object, **_kwargs: object) -> bool:
        heartbeat_started.set()
        try:
            await never_finish.wait()
        finally:
            heartbeat_reaped.set()
        raise AssertionError("unreachable")

    async def wait_for_deadline(*, timeout_seconds: float) -> None:
        assert timeout_seconds == _PROCESSING_TIMEOUT.total_seconds()
        deadline_started.set()
        try:
            await never_finish.wait()
        finally:
            deadline_reaped.set()

    monkeypatch.setattr(
        task_execution_module,
        "_wait_for_processing_deadline",
        wait_for_deadline,
    )

    harness = _execution_harness(
        monkeypatch,
        _claimed_execution(),
        process=AsyncMock(side_effect=process_task),
        heartbeat_transaction_count=1,
        renew_execution_lease=AsyncMock(side_effect=renew_lease),
    )
    execution = asyncio.create_task(
        harness.executor.execute(_TASK_ID, dispatch_token=_DISPATCH_TOKEN),
    )

    await asyncio.wait_for(processor_started.wait(), timeout=1)
    await asyncio.wait_for(heartbeat_started.wait(), timeout=1)
    await asyncio.wait_for(deadline_started.wait(), timeout=1)
    execution.cancel()

    with pytest.raises(asyncio.CancelledError):
        await execution

    assert processor_reaped.is_set()
    assert heartbeat_reaped.is_set()
    assert deadline_reaped.is_set()
    harness.complete_execution.assert_not_awaited()
    harness.schedule_execution_retry.assert_not_awaited()
    harness.fail_execution.assert_not_awaited()
    harness.transactions[2].__aenter__.assert_not_awaited()


@pytest.mark.asyncio
async def test_heartbeat_cannot_stop_normally_while_processor_is_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An impossible early stop fails fast instead of silently losing supervision."""

    processor_reaped = asyncio.Event()
    never_finish = asyncio.Event()

    async def stop_heartbeat(
        _stop_event: asyncio.Event,
        *,
        interval_seconds: float,
    ) -> bool:
        assert interval_seconds > 0
        return True

    async def process_task(_task: TaskProcessingInput) -> JsonObject:
        try:
            await never_finish.wait()
        finally:
            processor_reaped.set()
        raise AssertionError("unreachable")

    monkeypatch.setattr(
        task_execution_module,
        "_wait_for_heartbeat_stop",
        stop_heartbeat,
    )
    harness = _execution_harness(
        monkeypatch,
        _claimed_execution(),
        process=AsyncMock(side_effect=process_task),
    )

    with pytest.raises(
        RuntimeError,
        match=r"^execution heartbeat stopped before processing finished$",
    ):
        await asyncio.wait_for(
            harness.executor.execute(_TASK_ID, dispatch_token=_DISPATCH_TOKEN),
            timeout=1,
        )

    assert processor_reaped.is_set()
    harness.renew_execution_lease.assert_not_awaited()
    harness.complete_execution.assert_not_awaited()
    harness.schedule_execution_retry.assert_not_awaited()
    harness.fail_execution.assert_not_awaited()


@pytest.mark.asyncio
async def test_lost_ownership_wins_after_processor_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A renewal already in flight can fence a result that has just become ready."""

    heartbeat_stopped = asyncio.Event()
    heartbeat_query_started = asyncio.Event()
    allow_heartbeat_query = asyncio.Event()
    release_processor = asyncio.Event()
    processor_returned = asyncio.Event()

    async def begin_heartbeat(
        stop_event: asyncio.Event,
        *,
        interval_seconds: float,
    ) -> bool:
        assert interval_seconds > 0
        set_stop_event = stop_event.set

        def record_heartbeat_stop() -> None:
            set_stop_event()
            heartbeat_stopped.set()

        monkeypatch.setattr(stop_event, "set", record_heartbeat_stop)
        return False

    async def process_task(_task: TaskProcessingInput) -> JsonObject:
        await release_processor.wait()
        processor_returned.set()
        return {"processed": True}

    async def lose_ownership(*_args: object, **_kwargs: object) -> bool:
        heartbeat_query_started.set()
        await allow_heartbeat_query.wait()
        return False

    monkeypatch.setattr(
        task_execution_module,
        "_wait_for_heartbeat_stop",
        begin_heartbeat,
    )
    harness = _execution_harness(
        monkeypatch,
        _claimed_execution(),
        process=AsyncMock(side_effect=process_task),
        heartbeat_transaction_count=1,
        renew_execution_lease=AsyncMock(side_effect=lose_ownership),
    )
    execution = asyncio.create_task(
        harness.executor.execute(_TASK_ID, dispatch_token=_DISPATCH_TOKEN),
    )

    await asyncio.wait_for(heartbeat_query_started.wait(), timeout=1)
    release_processor.set()
    await asyncio.wait_for(processor_returned.wait(), timeout=1)
    await asyncio.wait_for(heartbeat_stopped.wait(), timeout=1)
    allow_heartbeat_query.set()
    outcome = await asyncio.wait_for(execution, timeout=1)

    assert outcome is TaskExecutionOutcome.LOST_OWNERSHIP
    harness.complete_execution.assert_not_awaited()
    harness.schedule_execution_retry.assert_not_awaited()
    harness.fail_execution.assert_not_awaited()
    harness.transactions[2].__aenter__.assert_not_awaited()


@pytest.mark.asyncio
async def test_external_cancellation_preserves_child_cleanup_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent cleanup defects remain visible alongside shutdown cancellation."""

    _allow_heartbeat_renewals(monkeypatch, 1)
    processor_cleanup_error = RuntimeError("processor cancellation cleanup failed")
    heartbeat_cleanup_error = OSError("heartbeat cancellation cleanup failed")
    processor_started = asyncio.Event()
    heartbeat_started = asyncio.Event()
    never_finish = asyncio.Event()

    async def process_task(_task: TaskProcessingInput) -> JsonObject:
        processor_started.set()
        try:
            await never_finish.wait()
        finally:
            raise processor_cleanup_error

    async def renew_lease(*_args: object, **_kwargs: object) -> bool:
        heartbeat_started.set()
        try:
            await never_finish.wait()
        finally:
            raise heartbeat_cleanup_error

    harness = _execution_harness(
        monkeypatch,
        _claimed_execution(),
        process=AsyncMock(side_effect=process_task),
        heartbeat_transaction_count=1,
        renew_execution_lease=AsyncMock(side_effect=renew_lease),
    )
    execution = asyncio.create_task(
        harness.executor.execute(_TASK_ID, dispatch_token=_DISPATCH_TOKEN),
    )

    await asyncio.wait_for(processor_started.wait(), timeout=1)
    await asyncio.wait_for(heartbeat_started.wait(), timeout=1)
    execution.cancel()

    with pytest.raises(BaseExceptionGroup) as error_info:
        await execution

    grouped_errors = error_info.value.exceptions
    assert len(grouped_errors) == 3
    assert isinstance(grouped_errors[0], asyncio.CancelledError)
    assert grouped_errors[1] is processor_cleanup_error
    assert grouped_errors[2] is heartbeat_cleanup_error
    harness.complete_execution.assert_not_awaited()
    harness.schedule_execution_retry.assert_not_awaited()
    harness.fail_execution.assert_not_awaited()


@pytest.mark.asyncio
async def test_heartbeat_wait_reports_its_own_elapsed_interval() -> None:
    """An elapsed heartbeat timer requests a renewal without leaking TimeoutError."""

    stopped = await asyncio.wait_for(
        task_execution_module._wait_for_heartbeat_stop(
            asyncio.Event(),
            interval_seconds=0.001,
        ),
        timeout=1,
    )

    assert stopped is False


@pytest.mark.asyncio
async def test_heartbeat_wait_preserves_an_unrelated_timeout_error() -> None:
    """A TimeoutError raised by the awaited operation is not mistaken for the timer."""

    expected_error = TimeoutError("event wait failed")
    wait = AsyncMock(side_effect=expected_error)
    stop_event = cast(asyncio.Event, SimpleNamespace(wait=wait))

    with pytest.raises(TimeoutError, match=r"^event wait failed$") as error_info:
        await task_execution_module._wait_for_heartbeat_stop(
            stop_event,
            interval_seconds=1,
        )

    assert error_info.value is expected_error
    wait.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_processing_deadline_wait_completes_after_its_timer() -> None:
    """The dedicated deadline helper completes normally instead of raising TimeoutError."""

    await asyncio.wait_for(
        task_execution_module._wait_for_processing_deadline(timeout_seconds=0),
        timeout=1,
    )


@pytest.mark.asyncio
async def test_stale_delivery_commits_the_lookup_without_processing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A duplicate or stale dispatch token is a successful no-op for the consumer."""

    harness = _execution_harness(monkeypatch, None)

    outcome = await harness.executor.execute(_TASK_ID, dispatch_token=_DISPATCH_TOKEN)

    assert outcome is TaskExecutionOutcome.NOT_CLAIMED
    harness.claim_for_execution.assert_awaited_once_with(
        _TASK_ID,
        dispatch_token=_DISPATCH_TOKEN,
        lease_duration=_LEASE_DURATION,
    )
    harness.process.assert_not_awaited()
    harness.retry_delay_for_attempt.assert_not_called()
    harness.complete_execution.assert_not_awaited()
    harness.schedule_execution_retry.assert_not_awaited()
    harness.fail_execution.assert_not_awaited()
    harness.begin.assert_called_once_with()
    harness.repository_factory.assert_called_once_with(harness.sessions[0])
    harness.transactions[0].__aexit__.assert_awaited_once_with(None, None, None)
    harness.transactions[1].__aenter__.assert_not_awaited()


@pytest.mark.asyncio
async def test_claim_failure_rolls_back_without_processing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A database error remains an infrastructure failure for the consumer supervisor."""

    expected_error = OSError("database unavailable")
    harness = _execution_harness(monkeypatch, _claimed_execution())
    harness.claim_for_execution.side_effect = expected_error

    with pytest.raises(OSError, match=r"^database unavailable$") as error_info:
        await harness.executor.execute(_TASK_ID, dispatch_token=_DISPATCH_TOKEN)

    assert error_info.value is expected_error
    exit_call = harness.transactions[0].__aexit__.await_args
    assert exit_call is not None
    assert exit_call.args[0] is OSError
    assert exit_call.args[1] is expected_error
    assert exit_call.args[2] is not None
    harness.process.assert_not_awaited()
    harness.complete_execution.assert_not_awaited()
    harness.schedule_execution_retry.assert_not_awaited()
    harness.fail_execution.assert_not_awaited()


@pytest.mark.asyncio
async def test_uncertain_claim_commit_never_runs_the_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Processing cannot start until durable execution ownership is known."""

    expected_error = OSError("claim commit outcome unknown")
    harness = _execution_harness(
        monkeypatch,
        _claimed_execution(),
        claim_exit_error=expected_error,
    )

    with pytest.raises(OSError, match=r"^claim commit outcome unknown$") as error_info:
        await harness.executor.execute(_TASK_ID, dispatch_token=_DISPATCH_TOKEN)

    assert error_info.value is expected_error
    harness.claim_for_execution.assert_awaited_once()
    harness.process.assert_not_awaited()
    harness.complete_execution.assert_not_awaited()
    harness.schedule_execution_retry.assert_not_awaited()
    harness.fail_execution.assert_not_awaited()
    harness.begin.assert_called_once_with()


@pytest.mark.parametrize(
    ("priority", "message_priority"),
    [
        (TaskPriority.LOW, 1),
        (TaskPriority.MEDIUM, 2),
        (TaskPriority.HIGH, 3),
    ],
)
@pytest.mark.asyncio
async def test_retryable_processing_error_schedules_a_bounded_retry(
    monkeypatch: pytest.MonkeyPatch,
    priority: TaskPriority,
    message_priority: int,
) -> None:
    """An expected transient failure uses the consumed attempt and task priority."""

    claimed = _claimed_execution(priority=priority, attempt_count=2, max_attempts=3)
    processing_error = TaskProcessingError("UPSTREAM_UNAVAILABLE", retryable=True)
    process = AsyncMock(side_effect=processing_error)
    retry_delay_for_attempt = Mock(return_value=timedelta(seconds=17))
    harness = _execution_harness(
        monkeypatch,
        claimed,
        process=process,
        retry_delay_for_attempt=retry_delay_for_attempt,
    )

    outcome = await harness.executor.execute(_TASK_ID, dispatch_token=_DISPATCH_TOKEN)

    assert outcome is TaskExecutionOutcome.RETRY_SCHEDULED
    retry_delay_for_attempt.assert_called_once_with(2)
    harness.schedule_execution_retry.assert_awaited_once_with(
        _TASK_ID,
        execution_token=_EXECUTION_TOKEN,
        retry_delay=timedelta(seconds=17),
        event_type=TASK_ROUTING_KEY,
        message_priority=message_priority,
    )
    harness.complete_execution.assert_not_awaited()
    harness.fail_execution.assert_not_awaited()
    harness.transactions[1].__aexit__.assert_awaited_once_with(None, None, None)


@pytest.mark.asyncio
async def test_permanent_processing_error_is_persisted_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The processor's safe code crosses the boundary without arbitrary details."""

    claimed = _claimed_execution()
    processing_error = TaskProcessingError("INVALID_TASK_INPUT", retryable=False)
    harness = _execution_harness(
        monkeypatch,
        claimed,
        process=AsyncMock(side_effect=processing_error),
    )

    outcome = await harness.executor.execute(_TASK_ID, dispatch_token=_DISPATCH_TOKEN)

    assert outcome is TaskExecutionOutcome.FAILED
    harness.fail_execution.assert_awaited_once_with(
        _TASK_ID,
        execution_token=_EXECUTION_TOKEN,
        error={"code": "INVALID_TASK_INPUT", "retryable": False},
    )
    harness.retry_delay_for_attempt.assert_not_called()
    harness.complete_execution.assert_not_awaited()
    harness.schedule_execution_retry.assert_not_awaited()


@pytest.mark.asyncio
async def test_exhausted_retryable_error_preserves_its_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Attempt exhaustion prevents scheduling without rewriting the processor error."""

    claimed = _claimed_execution(attempt_count=3, max_attempts=3)
    processing_error = TaskProcessingError("UPSTREAM_UNAVAILABLE", retryable=True)
    harness = _execution_harness(
        monkeypatch,
        claimed,
        process=AsyncMock(side_effect=processing_error),
    )

    outcome = await harness.executor.execute(_TASK_ID, dispatch_token=_DISPATCH_TOKEN)

    assert outcome is TaskExecutionOutcome.FAILED
    harness.fail_execution.assert_awaited_once_with(
        _TASK_ID,
        execution_token=_EXECUTION_TOKEN,
        error={"code": "UPSTREAM_UNAVAILABLE", "retryable": True},
    )
    harness.retry_delay_for_attempt.assert_not_called()
    harness.schedule_execution_retry.assert_not_awaited()
    harness.complete_execution.assert_not_awaited()


@pytest.mark.parametrize("retry_delay", [timedelta(0), timedelta(microseconds=-1)])
@pytest.mark.asyncio
async def test_invalid_retry_delay_leaves_the_claim_for_lease_recovery(
    monkeypatch: pytest.MonkeyPatch,
    retry_delay: timedelta,
) -> None:
    """A broken retry policy fails before opening a finalization transaction."""

    claimed = _claimed_execution(attempt_count=1, max_attempts=3)
    harness = _execution_harness(
        monkeypatch,
        claimed,
        process=AsyncMock(side_effect=TaskProcessingError("UPSTREAM_UNAVAILABLE", retryable=True)),
        retry_delay_for_attempt=Mock(return_value=retry_delay),
    )

    with pytest.raises(ValueError, match=r"^retry delay must be positive$"):
        await harness.executor.execute(_TASK_ID, dispatch_token=_DISPATCH_TOKEN)

    harness.retry_delay_for_attempt.assert_called_once_with(1)
    harness.begin.assert_called_once_with()
    harness.repository_factory.assert_called_once_with(harness.sessions[0])
    harness.complete_execution.assert_not_awaited()
    harness.schedule_execution_retry.assert_not_awaited()
    harness.fail_execution.assert_not_awaited()


@pytest.mark.asyncio
async def test_retry_policy_failure_propagates_without_finalizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Policy infrastructure errors remain visible and let lease recovery take over."""

    expected_error = ArithmeticError("retry policy failed")
    harness = _execution_harness(
        monkeypatch,
        _claimed_execution(),
        process=AsyncMock(side_effect=TaskProcessingError("UPSTREAM_UNAVAILABLE", retryable=True)),
        retry_delay_for_attempt=Mock(side_effect=expected_error),
    )

    with pytest.raises(ArithmeticError, match=r"^retry policy failed$") as error_info:
        await harness.executor.execute(_TASK_ID, dispatch_token=_DISPATCH_TOKEN)

    assert error_info.value is expected_error
    harness.begin.assert_called_once_with()
    harness.complete_execution.assert_not_awaited()
    harness.schedule_execution_retry.assert_not_awaited()
    harness.fail_execution.assert_not_awaited()


@pytest.mark.asyncio
async def test_unexpected_processor_error_propagates_without_persisting_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown processor defects fail fast and leave the fenced lease to recovery."""

    expected_error = RuntimeError(
        "secret at postgresql://worker:password@database/tasks token=private"
    )
    claim_committed = False

    def record_claim_commit() -> None:
        nonlocal claim_committed
        claim_committed = True

    harness = _execution_harness(
        monkeypatch,
        _claimed_execution(),
        process=AsyncMock(side_effect=expected_error),
        claim_on_exit=record_claim_commit,
    )

    with pytest.raises(RuntimeError) as error_info:
        await harness.executor.execute(_TASK_ID, dispatch_token=_DISPATCH_TOKEN)

    assert error_info.value is expected_error
    assert claim_committed is True
    harness.begin.assert_called_once_with()
    harness.retry_delay_for_attempt.assert_not_called()
    harness.complete_execution.assert_not_awaited()
    harness.schedule_execution_retry.assert_not_awaited()
    harness.fail_execution.assert_not_awaited()


@pytest.mark.asyncio
async def test_processor_cancellation_propagates_without_finalizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forced shutdown is never converted into a task failure or retry."""

    expected_error = asyncio.CancelledError()
    harness = _execution_harness(
        monkeypatch,
        _claimed_execution(),
        process=AsyncMock(side_effect=expected_error),
    )

    with pytest.raises(asyncio.CancelledError) as error_info:
        await harness.executor.execute(_TASK_ID, dispatch_token=_DISPATCH_TOKEN)

    assert error_info.value is expected_error
    harness.begin.assert_called_once_with()
    harness.retry_delay_for_attempt.assert_not_called()
    harness.complete_execution.assert_not_awaited()
    harness.schedule_execution_retry.assert_not_awaited()
    harness.fail_execution.assert_not_awaited()


def _invalid_processor_result(case: str) -> object:
    if case == "top-level-list":
        return []
    if case == "non-string-key":
        return {1: "value"}
    if case == "bytes":
        return {"value": b"not-json"}
    if case == "tuple":
        return {"value": (1, 2)}
    if case == "nan":
        return {"value": float("nan")}
    if case == "positive-infinity":
        return {"value": float("inf")}
    if case == "negative-infinity":
        return {"value": float("-inf")}
    if case == "oversized-integer":
        return {"value": 10**5000}
    if case == "nul-in-key":
        return {"invalid\x00key": "value"}
    if case == "nul-in-value":
        return {"value": "invalid\x00value"}
    if case == "surrogate-in-key":
        return {"invalid\ud800key": "value"}
    if case == "surrogate-in-value":
        return {"value": "invalid\udfffvalue"}
    if case == "dictionary-cycle":
        cyclic_mapping: dict[str, object] = {}
        cyclic_mapping["self"] = cyclic_mapping
        return cyclic_mapping
    if case == "list-cycle":
        cyclic_list: list[object] = []
        cyclic_list.append(cyclic_list)
        return {"value": cyclic_list}
    raise AssertionError(f"unknown invalid-result case: {case}")


@pytest.mark.parametrize(
    "case",
    [
        "top-level-list",
        "non-string-key",
        "bytes",
        "tuple",
        "nan",
        "positive-infinity",
        "negative-infinity",
        "oversized-integer",
        "nul-in-key",
        "nul-in-value",
        "surrogate-in-key",
        "surrogate-in-value",
        "dictionary-cycle",
        "list-cycle",
    ],
)
@pytest.mark.asyncio
async def test_invalid_processor_result_becomes_a_safe_terminal_failure(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    """Malformed output never reaches JSONB and cannot trigger a futile retry loop."""

    invalid_result = _invalid_processor_result(case)
    harness = _execution_harness(
        monkeypatch,
        _claimed_execution(),
        process=AsyncMock(return_value=invalid_result),
    )

    outcome = await harness.executor.execute(_TASK_ID, dispatch_token=_DISPATCH_TOKEN)

    assert outcome is TaskExecutionOutcome.FAILED
    harness.fail_execution.assert_awaited_once_with(
        _TASK_ID,
        execution_token=_EXECUTION_TOKEN,
        error={"code": INVALID_PROCESSOR_RESULT_CODE, "retryable": False},
    )
    harness.retry_delay_for_attempt.assert_not_called()
    harness.complete_execution.assert_not_awaited()
    harness.schedule_execution_retry.assert_not_awaited()


@pytest.mark.parametrize("path", ["completion", "retry", "failure"])
@pytest.mark.asyncio
async def test_fenced_finalization_reports_lost_ownership(
    monkeypatch: pytest.MonkeyPatch,
    path: FinalizationPath,
) -> None:
    """A cancellation or recovered lease wins without being overwritten by this worker."""

    process: AsyncMock
    if path == "completion":
        process = AsyncMock(return_value={"processed": True})
    elif path == "retry":
        process = AsyncMock(side_effect=TaskProcessingError("UPSTREAM_UNAVAILABLE", retryable=True))
    else:
        process = AsyncMock(side_effect=TaskProcessingError("INVALID_INPUT", retryable=False))

    harness = _execution_harness(
        monkeypatch,
        _claimed_execution(),
        process=process,
        complete_result=path != "completion",
        retry_result=path != "retry",
        failure_result=path != "failure",
    )

    outcome = await harness.executor.execute(_TASK_ID, dispatch_token=_DISPATCH_TOKEN)

    assert outcome is TaskExecutionOutcome.LOST_OWNERSHIP
    harness.transactions[1].__aexit__.assert_awaited_once_with(None, None, None)
    expected_calls = {
        "completion": harness.complete_execution.await_count,
        "retry": harness.schedule_execution_retry.await_count,
        "failure": harness.fail_execution.await_count,
    }
    assert expected_calls == {
        "completion": int(path == "completion"),
        "retry": int(path == "retry"),
        "failure": int(path == "failure"),
    }


def _set_finalization_failure(
    harness: _ExecutionHarness,
    path: FinalizationPath,
    error: BaseException,
) -> None:
    if path == "completion":
        harness.complete_execution.side_effect = error
    elif path == "retry":
        harness.schedule_execution_retry.side_effect = error
    else:
        harness.fail_execution.side_effect = error


def _processor_for_finalization(path: FinalizationPath) -> AsyncMock:
    if path == "completion":
        return AsyncMock(return_value={"processed": True})
    if path == "retry":
        return AsyncMock(side_effect=TaskProcessingError("UPSTREAM_UNAVAILABLE", retryable=True))
    return AsyncMock(side_effect=TaskProcessingError("INVALID_INPUT", retryable=False))


@pytest.mark.parametrize("path", ["completion", "retry", "failure"])
@pytest.mark.asyncio
async def test_finalization_write_failure_propagates_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
    path: FinalizationPath,
) -> None:
    """Database failures are not mistaken for processor failures or retried in place."""

    expected_error = OSError(f"{path} write failed")
    harness = _execution_harness(
        monkeypatch,
        _claimed_execution(),
        process=_processor_for_finalization(path),
    )
    _set_finalization_failure(harness, path, expected_error)

    with pytest.raises(OSError, match=rf"^{path} write failed$") as error_info:
        await harness.executor.execute(_TASK_ID, dispatch_token=_DISPATCH_TOKEN)

    assert error_info.value is expected_error
    exit_call = harness.transactions[1].__aexit__.await_args
    assert exit_call is not None
    assert exit_call.args[0] is OSError
    assert exit_call.args[1] is expected_error
    assert exit_call.args[2] is not None
    assert (
        harness.complete_execution.await_count
        + harness.schedule_execution_retry.await_count
        + harness.fail_execution.await_count
        == 1
    )


@pytest.mark.parametrize("path", ["completion", "retry", "failure"])
@pytest.mark.asyncio
async def test_uncertain_finalization_commit_propagates(
    monkeypatch: pytest.MonkeyPatch,
    path: FinalizationPath,
) -> None:
    """The consumer cannot acknowledge before the selected durable outcome commits."""

    expected_error = OSError(f"{path} commit outcome unknown")
    harness = _execution_harness(
        monkeypatch,
        _claimed_execution(),
        process=_processor_for_finalization(path),
        final_exit_error=expected_error,
    )

    with pytest.raises(
        OSError,
        match=rf"^{path} commit outcome unknown$",
    ) as error_info:
        await harness.executor.execute(_TASK_ID, dispatch_token=_DISPATCH_TOKEN)

    assert error_info.value is expected_error
    assert (
        harness.complete_execution.await_count
        + harness.schedule_execution_retry.await_count
        + harness.fail_execution.await_count
        == 1
    )
    harness.begin.assert_has_calls([call(), call()])
