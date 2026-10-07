"""Tests for transactional task creation orchestration."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from cims_task_service.application import task_creation as task_creation_module
from cims_task_service.application.idempotency import (
    fingerprint_task_creation_request,
    hash_task_creation_idempotency_key,
)
from cims_task_service.application.task_creation import (
    CreateTaskCommand,
    IdempotencyKeyConflictError,
    create_task,
)
from cims_task_service.domain.task import TaskPriority, TaskStatus
from cims_task_service.infrastructure.database.models import TaskModel
from cims_task_service.infrastructure.database.session import AsyncSessionFactory
from cims_task_service.infrastructure.database.task_repository import StoredTaskCreation
from cims_task_service.infrastructure.messaging.topology import TASK_ROUTING_KEY


@dataclass(frozen=True, slots=True)
class _ApplicationHarness:
    session: AsyncSession
    session_factory: AsyncSessionFactory
    begin: Mock
    transaction: AsyncMock
    repository_factory: Mock
    create_with_outbox: AsyncMock


def _task_for(
    command: CreateTaskCommand,
    *,
    idempotency_key_hash: bytes | None,
    request_fingerprint: bytes | None,
) -> TaskModel:
    return TaskModel(
        id=uuid4(),
        name=command.name,
        description=command.description,
        priority=command.priority,
        status=TaskStatus.NEW,
        idempotency_key_hash=idempotency_key_hash,
        request_fingerprint=request_fingerprint,
        started_at=None,
        finished_at=None,
        result=None,
        error=None,
        attempt_count=0,
        max_attempts=3,
        dispatch_token=uuid4(),
        execution_token=None,
        lease_expires_at=None,
    )


def _application_harness(
    monkeypatch: pytest.MonkeyPatch,
    stored: StoredTaskCreation,
) -> _ApplicationHarness:
    session = cast(AsyncSession, object())
    create_with_outbox = AsyncMock(return_value=stored)
    repository = SimpleNamespace(create_with_outbox=create_with_outbox)
    repository_factory = Mock(return_value=repository)
    monkeypatch.setattr(task_creation_module, "TaskRepository", repository_factory)

    transaction = AsyncMock()
    transaction.__aenter__.return_value = session
    transaction.__aexit__.return_value = False
    begin = Mock(return_value=transaction)
    session_factory = cast(
        AsyncSessionFactory,
        SimpleNamespace(begin=begin),
    )
    return _ApplicationHarness(
        session=session,
        session_factory=session_factory,
        begin=begin,
        transaction=transaction,
        repository_factory=repository_factory,
        create_with_outbox=create_with_outbox,
    )


@pytest.mark.parametrize(
    ("priority", "message_priority"),
    [
        (TaskPriority.LOW, 1),
        (TaskPriority.MEDIUM, 2),
        (TaskPriority.HIGH, 3),
    ],
)
@pytest.mark.asyncio
async def test_create_keyed_task_inside_owned_transaction(
    monkeypatch: pytest.MonkeyPatch,
    priority: TaskPriority,
    message_priority: int,
) -> None:
    """A keyed request forwards stable metadata and the broker priority."""

    command = CreateTaskCommand(
        name="Monthly report",
        description="Aggregate the source records",
        priority=priority,
        idempotency_key="request-42",
    )
    key_hash = hash_task_creation_idempotency_key("request-42")
    request_fingerprint = fingerprint_task_creation_request(
        name=command.name,
        description=command.description,
        priority=command.priority,
    )
    task = _task_for(
        command,
        idempotency_key_hash=key_hash,
        request_fingerprint=request_fingerprint,
    )
    harness = _application_harness(
        monkeypatch,
        StoredTaskCreation(task=task, created=True),
    )

    result = await create_task(
        command,
        session_factory=harness.session_factory,
        max_attempts=3,
    )

    assert result.task is task
    assert result.created is True
    harness.begin.assert_called_once_with()
    harness.transaction.__aenter__.assert_awaited_once_with()
    harness.repository_factory.assert_called_once_with(harness.session)
    harness.create_with_outbox.assert_awaited_once_with(
        name=command.name,
        description=command.description,
        priority=priority,
        max_attempts=3,
        idempotency_key_hash=key_hash,
        request_fingerprint=request_fingerprint,
        event_type=TASK_ROUTING_KEY,
        message_priority=message_priority,
    )
    harness.transaction.__aexit__.assert_awaited_once_with(None, None, None)


@pytest.mark.asyncio
async def test_create_unkeyed_task_omits_idempotency_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requests without a key remain non-idempotent and persist no hashes."""

    command = CreateTaskCommand(
        name="Ad hoc report",
        description="Run once",
        priority=TaskPriority.MEDIUM,
    )
    task = _task_for(
        command,
        idempotency_key_hash=None,
        request_fingerprint=None,
    )
    harness = _application_harness(
        monkeypatch,
        StoredTaskCreation(task=task, created=True),
    )

    result = await create_task(
        command,
        session_factory=harness.session_factory,
        max_attempts=3,
    )

    assert result.task is task
    assert result.created is True
    harness.create_with_outbox.assert_awaited_once_with(
        name=command.name,
        description=command.description,
        priority=TaskPriority.MEDIUM,
        max_attempts=3,
        idempotency_key_hash=None,
        request_fingerprint=None,
        event_type=TASK_ROUTING_KEY,
        message_priority=2,
    )


@pytest.mark.asyncio
async def test_matching_idempotency_replay_returns_the_stored_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A matching replay is distinguished from a newly created task."""

    idempotency_key = "daily-2026-09-05"
    command = CreateTaskCommand(
        name="Daily report",
        description="Aggregate yesterday",
        priority=TaskPriority.HIGH,
        idempotency_key=idempotency_key,
    )
    key_hash = hash_task_creation_idempotency_key(idempotency_key)
    request_fingerprint = fingerprint_task_creation_request(
        name=command.name,
        description=command.description,
        priority=command.priority,
    )
    task = _task_for(
        command,
        idempotency_key_hash=key_hash,
        request_fingerprint=request_fingerprint,
    )
    harness = _application_harness(
        monkeypatch,
        StoredTaskCreation(task=task, created=False),
    )

    result = await create_task(
        command,
        session_factory=harness.session_factory,
        max_attempts=3,
    )

    assert result.task is task
    assert result.created is False
    harness.transaction.__aexit__.assert_awaited_once_with(None, None, None)


@pytest.mark.asyncio
async def test_reused_key_with_different_request_raises_inside_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A key cannot silently identify a request with another fingerprint."""

    command = CreateTaskCommand(
        name="Changed report",
        description="Different request",
        priority=TaskPriority.LOW,
        idempotency_key="shared-key",
    )
    task = _task_for(
        command,
        idempotency_key_hash=hash_task_creation_idempotency_key("shared-key"),
        request_fingerprint=b"x" * 32,
    )
    harness = _application_harness(
        monkeypatch,
        StoredTaskCreation(task=task, created=False),
    )

    with pytest.raises(IdempotencyKeyConflictError) as error_info:
        await create_task(
            command,
            session_factory=harness.session_factory,
            max_attempts=3,
        )

    assert error_info.value.task_id == task.id
    exit_call = harness.transaction.__aexit__.await_args
    assert exit_call is not None
    assert exit_call.args[0] is IdempotencyKeyConflictError
    assert exit_call.args[1] is error_info.value
    assert exit_call.args[2] is not None


@pytest.mark.parametrize("max_attempts", [0, -1])
@pytest.mark.asyncio
async def test_invalid_max_attempts_fails_before_transaction(max_attempts: int) -> None:
    """Invalid retry policy cannot open a database transaction."""

    begin = Mock()
    session_factory = cast(
        AsyncSessionFactory,
        SimpleNamespace(begin=begin),
    )

    with pytest.raises(ValueError, match="max_attempts must be at least 1"):
        await create_task(
            CreateTaskCommand(
                name="Task",
                description="Description",
                priority=TaskPriority.LOW,
            ),
            session_factory=session_factory,
            max_attempts=max_attempts,
        )

    begin.assert_not_called()


@pytest.mark.asyncio
async def test_repository_failure_reaches_the_transaction_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistence errors propagate so the transaction context can roll back."""

    command = CreateTaskCommand(
        name="Task",
        description="Description",
        priority=TaskPriority.LOW,
    )
    task = _task_for(
        command,
        idempotency_key_hash=None,
        request_fingerprint=None,
    )
    harness = _application_harness(
        monkeypatch,
        StoredTaskCreation(task=task, created=True),
    )
    expected_error = RuntimeError("database unavailable")
    harness.create_with_outbox.side_effect = expected_error

    with pytest.raises(RuntimeError) as error_info:
        await create_task(
            command,
            session_factory=harness.session_factory,
            max_attempts=3,
        )

    assert error_info.value is expected_error
    exit_call = harness.transaction.__aexit__.await_args
    assert exit_call is not None
    assert exit_call.args[0] is RuntimeError
    assert exit_call.args[1] is expected_error
    assert exit_call.args[2] is not None
