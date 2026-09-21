"""Tests for atomic recovery of expired task executions."""

import asyncio
from contextlib import suppress
from dataclasses import FrozenInstanceError, dataclass
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Literal, cast
from unittest.mock import AsyncMock, Mock, call
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from cims_task_service.application import task_execution_recovery as recovery_module
from cims_task_service.application.task_execution_recovery import (
    MAX_RECOVERY_BATCH_SIZE,
    ExecutionRecoveryInvariantError,
    RecoveryBatchResult,
    TaskExecutionRecovery,
    run_execution_recovery_loop,
)
from cims_task_service.domain.task import TaskPriority
from cims_task_service.infrastructure.database.session import AsyncSessionFactory
from cims_task_service.infrastructure.database.task_execution_repository import (
    LockedExpiredTaskExecution,
)
from cims_task_service.infrastructure.messaging.topology import TASK_ROUTING_KEY

_LEASE_EXPIRES_AT = datetime(2026, 9, 21, 1, tzinfo=UTC)


async def _cancel_and_wait[T](task: asyncio.Task[T]) -> None:
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


@dataclass(frozen=True, slots=True)
class _RecoveryHarness:
    recovery: TaskExecutionRecovery
    session: AsyncSession
    begin: Mock
    transaction: AsyncMock
    repository_factory: Mock
    lock_batch: AsyncMock
    schedule_retry: AsyncMock
    fail_execution: AsyncMock
    retry_delay_for_attempt: Mock


def _locked_execution(
    task_number: int,
    *,
    priority: TaskPriority = TaskPriority.HIGH,
    attempt_count: int = 1,
    max_attempts: int = 3,
) -> LockedExpiredTaskExecution:
    return LockedExpiredTaskExecution(
        task_id=UUID(int=task_number),
        priority=priority,
        attempt_count=attempt_count,
        max_attempts=max_attempts,
        execution_token=UUID(int=task_number + 100),
        lease_expires_at=_LEASE_EXPIRES_AT,
    )


def _recovery_harness(
    monkeypatch: pytest.MonkeyPatch,
    locked_executions: tuple[LockedExpiredTaskExecution, ...],
    *,
    schedule_result: bool = True,
    failure_result: bool = True,
    retry_delay_for_attempt: Mock | None = None,
) -> _RecoveryHarness:
    session = cast(AsyncSession, object())
    lock_batch = AsyncMock(return_value=locked_executions)
    schedule_retry = AsyncMock(return_value=schedule_result)
    fail_execution = AsyncMock(return_value=failure_result)
    repository = SimpleNamespace(
        lock_expired_execution_batch=lock_batch,
        schedule_execution_retry=schedule_retry,
        fail_execution=fail_execution,
    )
    repository_factory = Mock(return_value=repository)
    monkeypatch.setattr(recovery_module, "TaskExecutionRepository", repository_factory)

    transaction = AsyncMock()
    transaction.__aenter__.return_value = session
    transaction.__aexit__.return_value = False
    begin = Mock(return_value=transaction)
    session_factory = cast(AsyncSessionFactory, SimpleNamespace(begin=begin))
    delay_factory = retry_delay_for_attempt or Mock(
        side_effect=lambda attempt_count: timedelta(seconds=attempt_count * 10)
    )
    recovery = TaskExecutionRecovery(
        session_factory,
        batch_size=10,
        retry_delay_for_attempt=delay_factory,
    )
    return _RecoveryHarness(
        recovery=recovery,
        session=session,
        begin=begin,
        transaction=transaction,
        repository_factory=repository_factory,
        lock_batch=lock_batch,
        schedule_retry=schedule_retry,
        fail_execution=fail_execution,
        retry_delay_for_attempt=delay_factory,
    )


@pytest.mark.asyncio
async def test_recovery_retries_and_fails_one_locked_batch_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every locked row is resolved before the caller-owned transaction commits."""

    low = _locked_execution(1, priority=TaskPriority.LOW)
    medium = _locked_execution(2, priority=TaskPriority.MEDIUM, attempt_count=2)
    high = _locked_execution(3, priority=TaskPriority.HIGH)
    exhausted = _locked_execution(4, attempt_count=3)
    harness = _recovery_harness(monkeypatch, (low, medium, high, exhausted))

    result = await harness.recovery.recover_once()

    assert result == RecoveryBatchResult(locked=4, retried=3, failed=1)
    with pytest.raises(FrozenInstanceError):
        result.retried = 0  # type: ignore[misc]
    harness.begin.assert_called_once_with()
    harness.transaction.__aenter__.assert_awaited_once_with()
    harness.repository_factory.assert_called_once_with(harness.session)
    harness.lock_batch.assert_awaited_once_with(batch_size=10)
    assert harness.retry_delay_for_attempt.call_args_list == [call(1), call(2), call(1)]
    assert harness.schedule_retry.await_args_list == [
        call(
            low.task_id,
            execution_token=low.execution_token,
            retry_delay=timedelta(seconds=10),
            event_type=TASK_ROUTING_KEY,
            message_priority=1,
        ),
        call(
            medium.task_id,
            execution_token=medium.execution_token,
            retry_delay=timedelta(seconds=20),
            event_type=TASK_ROUTING_KEY,
            message_priority=2,
        ),
        call(
            high.task_id,
            execution_token=high.execution_token,
            retry_delay=timedelta(seconds=10),
            event_type=TASK_ROUTING_KEY,
            message_priority=3,
        ),
    ]
    harness.fail_execution.assert_awaited_once_with(
        exhausted.task_id,
        execution_token=exhausted.execution_token,
        error={"code": "EXECUTION_LEASE_EXPIRED", "retryable": False},
    )
    harness.transaction.__aexit__.assert_awaited_once_with(None, None, None)


@pytest.mark.asyncio
async def test_empty_recovery_pass_commits_zero_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An idle pass remains a finite transaction without consulting retry policy."""

    harness = _recovery_harness(monkeypatch, ())

    result = await harness.recovery.recover_once()

    assert result == RecoveryBatchResult(locked=0, retried=0, failed=0)
    harness.lock_batch.assert_awaited_once_with(batch_size=10)
    harness.retry_delay_for_attempt.assert_not_called()
    harness.schedule_retry.assert_not_awaited()
    harness.fail_execution.assert_not_awaited()
    harness.transaction.__aexit__.assert_awaited_once_with(None, None, None)


@pytest.mark.parametrize(
    ("batch_size", "message"),
    [
        (0, "batch_size must be at least 1"),
        (MAX_RECOVERY_BATCH_SIZE + 1, f"batch_size must be at most {MAX_RECOVERY_BATCH_SIZE}"),
    ],
)
def test_recovery_rejects_an_invalid_batch_before_database_access(
    batch_size: int,
    message: str,
) -> None:
    """Operational configuration cannot create an unbounded recovery transaction."""

    begin = Mock()

    with pytest.raises(ValueError, match=f"^{message}$"):
        TaskExecutionRecovery(
            cast(AsyncSessionFactory, SimpleNamespace(begin=begin)),
            batch_size=batch_size,
            retry_delay_for_attempt=Mock(),
        )

    begin.assert_not_called()


@pytest.mark.parametrize("retry_delay", [timedelta(0), timedelta(microseconds=-1)])
@pytest.mark.asyncio
async def test_recovery_rejects_a_non_positive_retry_delay_before_writing(
    monkeypatch: pytest.MonkeyPatch,
    retry_delay: timedelta,
) -> None:
    """A broken policy aborts the transaction before any task is mutated."""

    delay_factory = Mock(return_value=retry_delay)
    harness = _recovery_harness(
        monkeypatch,
        (_locked_execution(1),),
        retry_delay_for_attempt=delay_factory,
    )

    with pytest.raises(ValueError, match=r"^retry delay must be positive$"):
        await harness.recovery.recover_once()

    delay_factory.assert_called_once_with(1)
    harness.schedule_retry.assert_not_awaited()
    harness.fail_execution.assert_not_awaited()
    exception = harness.transaction.__aexit__.await_args.args
    assert exception[0] is ValueError
    assert isinstance(exception[1], ValueError)


@pytest.mark.asyncio
async def test_recovery_calculates_every_delay_before_writing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A policy failure cannot leave earlier rows flushed before the rollback."""

    policy_error = ArithmeticError("retry policy failed")
    delay_factory = Mock(side_effect=[timedelta(seconds=1), policy_error])
    harness = _recovery_harness(
        monkeypatch,
        (_locked_execution(1), _locked_execution(2, attempt_count=2)),
        retry_delay_for_attempt=delay_factory,
    )

    with pytest.raises(ArithmeticError) as error_info:
        await harness.recovery.recover_once()

    assert error_info.value is policy_error
    assert delay_factory.call_args_list == [call(1), call(2)]
    harness.schedule_retry.assert_not_awaited()
    harness.fail_execution.assert_not_awaited()


@pytest.mark.parametrize("operation", ["retry", "failure"])
@pytest.mark.asyncio
async def test_recovery_rolls_back_if_locked_ownership_is_unexpectedly_lost(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    """An impossible compare-and-set loss is surfaced instead of counted as recovery."""

    execution = _locked_execution(
        1,
        attempt_count=1 if operation == "retry" else 3,
    )
    harness = _recovery_harness(
        monkeypatch,
        (execution,),
        schedule_result=operation != "retry",
        failure_result=operation != "failure",
    )

    with pytest.raises(ExecutionRecoveryInvariantError) as error_info:
        await harness.recovery.recover_once()

    assert error_info.value.task_id == execution.task_id
    assert str(execution.task_id) in str(error_info.value)
    exception = harness.transaction.__aexit__.await_args.args
    assert exception[0] is ExecutionRecoveryInvariantError
    assert exception[1] is error_info.value


@pytest.mark.parametrize(
    "poll_interval_seconds",
    [0.0, -0.1, float("nan"), float("inf"), float("-inf")],
)
@pytest.mark.asyncio
async def test_recovery_loop_rejects_an_invalid_poll_interval_before_recovery(
    poll_interval_seconds: float,
) -> None:
    """Invalid timing cannot reach the database even when already stopped."""

    recover_once = AsyncMock()
    stop_event = asyncio.Event()
    stop_event.set()

    with pytest.raises(
        ValueError,
        match=r"^poll_interval_seconds must be finite and positive$",
    ):
        await run_execution_recovery_loop(
            recover_once,
            stop_event=stop_event,
            poll_interval_seconds=poll_interval_seconds,
        )

    recover_once.assert_not_awaited()


@pytest.mark.asyncio
async def test_recovery_loop_does_not_recover_when_already_stopped() -> None:
    """A pre-existing shutdown request prevents a new transaction."""

    recover_once = AsyncMock()
    stop_event = asyncio.Event()
    stop_event.set()

    await run_execution_recovery_loop(
        recover_once,
        stop_event=stop_event,
        poll_interval_seconds=1.0,
    )

    recover_once.assert_not_awaited()


@pytest.mark.asyncio
async def test_recovery_loop_drains_nonempty_batches_before_idle_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backlogged work is processed sequentially without a polling delay."""

    stop_event = asyncio.Event()
    recover_once = AsyncMock(
        side_effect=[
            RecoveryBatchResult(locked=2, retried=2, failed=0),
            RecoveryBatchResult(locked=1, retried=0, failed=1),
            RecoveryBatchResult(locked=0, retried=0, failed=0),
        ]
    )

    async def request_stop(
        received_stop_event: asyncio.Event,
        *,
        poll_interval_seconds: float,
    ) -> None:
        assert received_stop_event is stop_event
        assert poll_interval_seconds == 0.25
        stop_event.set()

    wait_for_stop = AsyncMock(side_effect=request_stop)
    monkeypatch.setattr(recovery_module, "_wait_for_recovery_stop", wait_for_stop)

    await run_execution_recovery_loop(
        recover_once,
        stop_event=stop_event,
        poll_interval_seconds=0.25,
    )

    assert recover_once.await_count == 3
    wait_for_stop.assert_awaited_once_with(
        stop_event,
        poll_interval_seconds=0.25,
    )


@pytest.mark.asyncio
async def test_stop_event_wakes_the_production_recovery_idle_wait() -> None:
    """Shutdown interrupts an idle wait without starting another recovery pass."""

    recovery_started = asyncio.Event()
    stop_event = asyncio.Event()

    async def recover_once() -> RecoveryBatchResult:
        recovery_started.set()
        return RecoveryBatchResult(locked=0, retried=0, failed=0)

    loop_task = asyncio.create_task(
        run_execution_recovery_loop(
            recover_once,
            stop_event=stop_event,
            poll_interval_seconds=60.0,
        )
    )
    try:
        async with asyncio.timeout(1):
            await recovery_started.wait()
        stop_event.set()
        async with asyncio.timeout(1):
            await loop_task
    finally:
        if not loop_task.done():
            await _cancel_and_wait(loop_task)


@pytest.mark.asyncio
async def test_recovery_idle_deadline_triggers_the_next_poll() -> None:
    """An expired idle interval starts exactly one subsequent pass."""

    stop_event = asyncio.Event()
    recovery_calls = 0

    async def recover_once() -> RecoveryBatchResult:
        nonlocal recovery_calls
        recovery_calls += 1
        if recovery_calls == 2:
            stop_event.set()
        return RecoveryBatchResult(locked=0, retried=0, failed=0)

    async with asyncio.timeout(1):
        await run_execution_recovery_loop(
            recover_once,
            stop_event=stop_event,
            poll_interval_seconds=0.001,
        )

    assert recovery_calls == 2


@pytest.mark.asyncio
async def test_recovery_idle_wait_does_not_swallow_an_unrelated_timeout() -> None:
    """Only expiration of the loop's own deadline is treated as normal polling."""

    expected_error = TimeoutError("event wait failed")

    class FailingStopEvent(asyncio.Event):
        async def wait(self) -> Literal[True]:
            raise expected_error

    recover_once = AsyncMock(return_value=RecoveryBatchResult(locked=0, retried=0, failed=0))

    with pytest.raises(TimeoutError) as error_info:
        await run_execution_recovery_loop(
            recover_once,
            stop_event=FailingStopEvent(),
            poll_interval_seconds=60.0,
        )

    assert error_info.value is expected_error
    recover_once.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_stop_during_active_recovery_finishes_it_without_another_pass() -> None:
    """Graceful shutdown lets the current transaction complete exactly once."""

    recovery_started = asyncio.Event()
    finish_recovery = asyncio.Event()
    stop_event = asyncio.Event()
    recovery_calls = 0

    async def recover_once() -> RecoveryBatchResult:
        nonlocal recovery_calls
        recovery_calls += 1
        recovery_started.set()
        await finish_recovery.wait()
        return RecoveryBatchResult(locked=1, retried=1, failed=0)

    loop_task = asyncio.create_task(
        run_execution_recovery_loop(
            recover_once,
            stop_event=stop_event,
            poll_interval_seconds=1.0,
        )
    )
    try:
        async with asyncio.timeout(1):
            await recovery_started.wait()
        stop_event.set()
        assert not loop_task.done()
        finish_recovery.set()
        async with asyncio.timeout(1):
            await loop_task
    finally:
        if not loop_task.done():
            await _cancel_and_wait(loop_task)

    assert recovery_calls == 1


@pytest.mark.asyncio
async def test_recovery_failure_terminates_the_loop() -> None:
    """Unexpected failures retain their identity for process supervision."""

    expected_error = RuntimeError("recovery failed")
    recover_once = AsyncMock(side_effect=expected_error)

    with pytest.raises(RuntimeError) as error_info:
        await run_execution_recovery_loop(
            recover_once,
            stop_event=asyncio.Event(),
            poll_interval_seconds=1.0,
        )

    assert error_info.value is expected_error


@pytest.mark.asyncio
async def test_recovery_timeout_is_not_mistaken_for_an_idle_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A database timeout terminates the loop instead of becoming a repoll."""

    expected_error = TimeoutError("recovery timed out")
    recover_once = AsyncMock(side_effect=expected_error)
    wait_for_stop = AsyncMock()
    monkeypatch.setattr(recovery_module, "_wait_for_recovery_stop", wait_for_stop)

    with pytest.raises(TimeoutError) as error_info:
        await run_execution_recovery_loop(
            recover_once,
            stop_event=asyncio.Event(),
            poll_interval_seconds=1.0,
        )

    assert error_info.value is expected_error
    wait_for_stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelling_idle_recovery_loop_propagates() -> None:
    """External cancellation interrupts the production idle wait unchanged."""

    class ObservedStopEvent(asyncio.Event):
        def __init__(self) -> None:
            super().__init__()
            self.wait_started = asyncio.Event()

        async def wait(self) -> Literal[True]:
            self.wait_started.set()
            return await super().wait()

    stop_event = ObservedStopEvent()
    recover_once = AsyncMock(return_value=RecoveryBatchResult(locked=0, retried=0, failed=0))
    loop_task = asyncio.create_task(
        run_execution_recovery_loop(
            recover_once,
            stop_event=stop_event,
            poll_interval_seconds=60.0,
        )
    )
    try:
        async with asyncio.timeout(1):
            await stop_event.wait_started.wait()
        loop_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await loop_task
    finally:
        if not loop_task.done():
            await _cancel_and_wait(loop_task)

    recover_once.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_cancelling_active_recovery_propagates_into_the_current_pass() -> None:
    """Forced shutdown cannot orphan a recovery transaction."""

    recovery_started = asyncio.Event()
    recovery_cancelled = asyncio.Event()

    async def recover_once() -> RecoveryBatchResult:
        recovery_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            recovery_cancelled.set()
        raise AssertionError("unreachable")

    loop_task = asyncio.create_task(
        run_execution_recovery_loop(
            recover_once,
            stop_event=asyncio.Event(),
            poll_interval_seconds=1.0,
        )
    )
    try:
        async with asyncio.timeout(1):
            await recovery_started.wait()
        loop_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await loop_task
    finally:
        if not loop_task.done():
            await _cancel_and_wait(loop_task)

    assert recovery_cancelled.is_set()
