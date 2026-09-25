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
_LEASE_EXPIRES_AT = datetime(2026, 9, 25, 2, 30, tzinfo=UTC)

type FinalizationPath = Literal["completion", "retry", "failure"]


@dataclass(frozen=True, slots=True)
class _ExecutionHarness:
    executor: TaskExecutor
    sessions: tuple[AsyncSession, AsyncSession]
    transactions: tuple[AsyncMock, AsyncMock]
    begin: Mock
    repository_factory: Mock
    claim_for_execution: AsyncMock
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
    claim_on_exit: Callable[[], None] | None = None,
    claim_exit_error: BaseException | None = None,
    final_exit_error: BaseException | None = None,
) -> _ExecutionHarness:
    claim_session = cast(AsyncSession, object())
    final_session = cast(AsyncSession, object())
    transactions = (
        _transaction(
            claim_session,
            on_exit=claim_on_exit,
            exit_error=claim_exit_error,
        ),
        _transaction(final_session, exit_error=final_exit_error),
    )
    begin = Mock(side_effect=transactions)
    session_factory = cast(AsyncSessionFactory, SimpleNamespace(begin=begin))

    claim_for_execution = AsyncMock(return_value=claimed)
    complete_execution = AsyncMock(return_value=complete_result)
    schedule_execution_retry = AsyncMock(return_value=retry_result)
    fail_execution = AsyncMock(return_value=failure_result)
    claim_repository = SimpleNamespace(claim_for_execution=claim_for_execution)
    final_repository = SimpleNamespace(
        complete_execution=complete_execution,
        schedule_execution_retry=schedule_execution_retry,
        fail_execution=fail_execution,
    )
    repository_factory = Mock(side_effect=(claim_repository, final_repository))
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
        retry_delay_for_attempt=retry_policy,
    )
    return _ExecutionHarness(
        executor=executor,
        sessions=(claim_session, final_session),
        transactions=transactions,
        begin=begin,
        repository_factory=repository_factory,
        claim_for_execution=claim_for_execution,
        process=process_mock,
        retry_delay_for_attempt=retry_policy,
        complete_execution=complete_execution,
        schedule_execution_retry=schedule_execution_retry,
        fail_execution=fail_execution,
    )


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
    assert harness.repository_factory.call_args_list == [
        call(harness.sessions[0]),
        call(harness.sessions[1]),
    ]
    assert harness.sessions[0] is not harness.sessions[1]
    for transaction in harness.transactions:
        transaction.__aexit__.assert_awaited_once_with(None, None, None)


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
