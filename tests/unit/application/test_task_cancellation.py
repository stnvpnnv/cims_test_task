"""Tests for transactional task cancellation orchestration."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock
from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from cims_task_service.application import task_cancellation as task_cancellation_module
from cims_task_service.application.task_cancellation import cancel_task
from cims_task_service.application.task_errors import (
    TaskNotCancellableError,
    TaskNotFoundError,
)
from cims_task_service.domain.task import TaskPriority, TaskStatus
from cims_task_service.infrastructure.database.models import TaskModel
from cims_task_service.infrastructure.database.session import AsyncSessionFactory
from cims_task_service.infrastructure.database.task_repository import StoredTaskCancellation
from cims_task_service.infrastructure.messaging.topology import TASK_ROUTING_KEY

_TASK_ID = UUID("dc2e9988-f896-4316-bfb3-56d2b66ed186")
_CREATED_AT = datetime(2026, 9, 5, 1, 2, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class _CancellationHarness:
    session: AsyncSession
    session_factory: AsyncSessionFactory
    begin: Mock
    transaction: AsyncMock
    repository_factory: Mock
    cancel_with_outbox: AsyncMock


def _task(status: TaskStatus) -> TaskModel:
    started = status in {
        TaskStatus.IN_PROGRESS,
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
    }
    terminal = status in {
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    }
    return TaskModel(
        id=_TASK_ID,
        name="Daily report",
        description="Aggregate daily metrics",
        priority=TaskPriority.HIGH,
        status=status,
        created_at=_CREATED_AT,
        idempotency_key_hash=None,
        request_fingerprint=None,
        started_at=_CREATED_AT + timedelta(seconds=1) if started else None,
        finished_at=_CREATED_AT + timedelta(seconds=2) if terminal else None,
        result={} if status is TaskStatus.COMPLETED else None,
        error={"type": "processing_failed"} if status is TaskStatus.FAILED else None,
        attempt_count=1 if started else 0,
        max_attempts=3,
        dispatch_token=uuid4() if status in {TaskStatus.NEW, TaskStatus.PENDING} else None,
        execution_token=uuid4() if status is TaskStatus.IN_PROGRESS else None,
        lease_expires_at=_CREATED_AT + timedelta(minutes=1)
        if status is TaskStatus.IN_PROGRESS
        else None,
    )


def _cancellation_harness(
    monkeypatch: pytest.MonkeyPatch,
    stored: StoredTaskCancellation | None,
) -> _CancellationHarness:
    session = cast(AsyncSession, object())
    cancel_with_outbox = AsyncMock(return_value=stored)
    repository = SimpleNamespace(cancel_with_outbox=cancel_with_outbox)
    repository_factory = Mock(return_value=repository)
    monkeypatch.setattr(task_cancellation_module, "TaskRepository", repository_factory)

    transaction = AsyncMock()
    transaction.__aenter__.return_value = session
    transaction.__aexit__.return_value = False
    begin = Mock(return_value=transaction)
    session_factory = cast(AsyncSessionFactory, SimpleNamespace(begin=begin))
    return _CancellationHarness(
        session=session,
        session_factory=session_factory,
        begin=begin,
        transaction=transaction,
        repository_factory=repository_factory,
        cancel_with_outbox=cancel_with_outbox,
    )


@pytest.mark.parametrize("changed", [True, False])
@pytest.mark.asyncio
async def test_cancel_task_commits_new_and_repeated_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    changed: bool,
) -> None:
    """Both the transition and its idempotent replay return the cancelled task."""

    task = _task(TaskStatus.CANCELLED)
    harness = _cancellation_harness(
        monkeypatch,
        StoredTaskCancellation(task=task, changed=changed),
    )

    result = await cancel_task(_TASK_ID, session_factory=harness.session_factory)

    assert result is task
    harness.begin.assert_called_once_with()
    harness.transaction.__aenter__.assert_awaited_once_with()
    harness.repository_factory.assert_called_once_with(harness.session)
    harness.cancel_with_outbox.assert_awaited_once_with(
        _TASK_ID,
        event_type=TASK_ROUTING_KEY,
    )
    harness.transaction.__aexit__.assert_awaited_once_with(None, None, None)


@pytest.mark.asyncio
async def test_cancel_task_raises_not_found_inside_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown identifier rolls back through the transaction owner."""

    harness = _cancellation_harness(monkeypatch, None)

    with pytest.raises(TaskNotFoundError) as error_info:
        await cancel_task(_TASK_ID, session_factory=harness.session_factory)

    assert error_info.value.task_id == _TASK_ID
    exit_call = harness.transaction.__aexit__.await_args
    assert exit_call is not None
    assert exit_call.args[0] is TaskNotFoundError
    assert exit_call.args[1] is error_info.value
    assert exit_call.args[2] is not None


@pytest.mark.parametrize("current_status", [TaskStatus.COMPLETED, TaskStatus.FAILED])
@pytest.mark.asyncio
async def test_cancel_task_rejects_completed_and_failed_tasks(
    monkeypatch: pytest.MonkeyPatch,
    current_status: TaskStatus,
) -> None:
    """A final processing outcome cannot be replaced by cancellation."""

    task = _task(current_status)
    harness = _cancellation_harness(
        monkeypatch,
        StoredTaskCancellation(task=task, changed=False),
    )

    with pytest.raises(TaskNotCancellableError) as error_info:
        await cancel_task(_TASK_ID, session_factory=harness.session_factory)

    assert error_info.value.task_id == _TASK_ID
    assert error_info.value.current_status is current_status
    assert str(error_info.value) == (
        f"Task {_TASK_ID} cannot be cancelled from status {current_status.value}"
    )
    exit_call = harness.transaction.__aexit__.await_args
    assert exit_call is not None
    assert exit_call.args[0] is TaskNotCancellableError
    assert exit_call.args[1] is error_info.value
    assert exit_call.args[2] is not None


@pytest.mark.asyncio
async def test_cancel_task_rejects_impossible_active_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An active state after a failed compare-and-set is treated as an invariant breach."""

    task = _task(TaskStatus.NEW)
    harness = _cancellation_harness(
        monkeypatch,
        StoredTaskCancellation(task=task, changed=False),
    )

    with pytest.raises(
        RuntimeError,
        match=r"^task cancellation compare-and-set returned an active task$",
    ):
        await cancel_task(_TASK_ID, session_factory=harness.session_factory)

    exit_call = harness.transaction.__aexit__.await_args
    assert exit_call is not None
    assert exit_call.args[0] is RuntimeError
    assert exit_call.args[1] is not None
    assert exit_call.args[2] is not None


@pytest.mark.asyncio
async def test_cancel_task_propagates_repository_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistence errors escape the use case so its transaction is rolled back."""

    harness = _cancellation_harness(
        monkeypatch,
        StoredTaskCancellation(task=_task(TaskStatus.CANCELLED), changed=True),
    )
    expected_error = OSError("database unavailable")
    harness.cancel_with_outbox.side_effect = expected_error

    with pytest.raises(OSError, match=r"^database unavailable$") as error_info:
        await cancel_task(_TASK_ID, session_factory=harness.session_factory)

    assert error_info.value is expected_error
    exit_call = harness.transaction.__aexit__.await_args
    assert exit_call is not None
    assert exit_call.args[0] is OSError
    assert exit_call.args[1] is expected_error
    assert exit_call.args[2] is not None
