"""Tests for task creation persistence and transactional outbox behavior."""

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from typing import cast
from unittest.mock import AsyncMock, Mock
from uuid import UUID

import pytest
from sqlalchemy.dialects.postgresql.base import PGDialect
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import ClauseElement

from cims_task_service.domain.task import TaskPriority, TaskStatus
from cims_task_service.infrastructure.database.models import (
    OutboxEventModel,
    TaskModel,
)
from cims_task_service.infrastructure.database.task_repository import (
    StoredTaskCreation,
    TaskRepository,
)

_POSTGRESQL_DIALECT = PGDialect()  # type: ignore[no-untyped-call]


class _ScalarResult:
    """Minimal scalar-result double with explicit terminal operations."""

    def __init__(self, *, one_or_none: TaskModel | None = None) -> None:
        self._one_or_none = one_or_none

    def one_or_none(self) -> TaskModel | None:
        return self._one_or_none


@pytest.fixture
def inserted_task() -> TaskModel:
    """Return the complete row PostgreSQL would produce via RETURNING."""

    return TaskModel(
        id=UUID("dc2e9988-f896-4316-bfb3-56d2b66ed186"),
        name="Daily report",
        description="Aggregate daily metrics",
        priority=TaskPriority.HIGH,
        status=TaskStatus.NEW,
        created_at=datetime(2026, 9, 5, 1, 2, tzinfo=UTC),
        idempotency_key_hash=b"k" * 32,
        request_fingerprint=b"f" * 32,
        started_at=None,
        finished_at=None,
        result=None,
        error=None,
        attempt_count=0,
        max_attempts=3,
        dispatch_token=UUID("917d89f6-537a-4166-980d-b512932423a5"),
        execution_token=None,
        lease_expires_at=None,
    )


def _session_with_scalar_results(
    *results: _ScalarResult,
) -> tuple[AsyncSession, AsyncMock, Mock, AsyncMock]:
    scalars = AsyncMock(side_effect=results)
    add = Mock()
    flush = AsyncMock()
    session = cast(
        AsyncSession,
        Mock(scalars=scalars, add=add, flush=flush),
    )
    return session, scalars, add, flush


def _compiled_sql(statement: ClauseElement) -> str:
    return " ".join(str(statement.compile(dialect=_POSTGRESQL_DIALECT)).split())


def _compiled_parameters(statement: ClauseElement) -> dict[str, object]:
    return cast(
        dict[str, object],
        statement.compile(dialect=_POSTGRESQL_DIALECT).params,
    )


@pytest.mark.asyncio
async def test_new_task_uses_scoped_conflict_target_and_creates_one_outbox_event(
    inserted_task: TaskModel,
) -> None:
    """A successful RETURNING row produces the matching execution event."""

    session, scalars, add, flush = _session_with_scalar_results(
        _ScalarResult(one_or_none=inserted_task)
    )
    repository = TaskRepository(session)

    stored = await repository.create_with_outbox(
        name="Daily report",
        description="Aggregate daily metrics",
        priority=TaskPriority.HIGH,
        max_attempts=3,
        idempotency_key_hash=b"k" * 32,
        request_fingerprint=b"f" * 32,
        event_type="task.execute.v1",
        message_priority=3,
    )

    assert stored.task is inserted_task
    assert stored.created is True
    scalars.assert_awaited_once()
    assert scalars.await_args is not None
    create_statement = scalars.await_args.args[0]
    create_sql = _compiled_sql(create_statement)
    assert (
        "ON CONFLICT (idempotency_key_hash) WHERE idempotency_key_hash IS NOT NULL DO NOTHING"
    ) in create_sql
    assert "RETURNING tasks.id, tasks.name, tasks.description" in create_sql

    parameters = _compiled_parameters(create_statement)
    assert parameters["name"] == "Daily report"
    assert parameters["description"] == "Aggregate daily metrics"
    assert parameters["priority"] is TaskPriority.HIGH
    assert parameters["status"] is TaskStatus.NEW
    assert parameters["idempotency_key_hash"] == b"k" * 32
    assert parameters["request_fingerprint"] == b"f" * 32
    assert parameters["attempt_count"] == 0
    assert parameters["max_attempts"] == 3
    assert parameters["started_at"] is None
    assert parameters["finished_at"] is None
    assert parameters["result"] is None
    assert parameters["error"] is None
    assert parameters["execution_token"] is None
    assert parameters["lease_expires_at"] is None
    assert isinstance(parameters["id"], UUID)
    assert isinstance(parameters["dispatch_token"], UUID)
    assert parameters["id"] != parameters["dispatch_token"]

    add.assert_called_once()
    event = add.call_args.args[0]
    assert isinstance(event, OutboxEventModel)
    assert event.id.version == 4
    assert event.task_id == inserted_task.id
    assert event.event_type == "task.execute.v1"
    assert event.payload == {
        "task_id": str(inserted_task.id),
        "dispatch_token": str(inserted_task.dispatch_token),
    }
    assert event.message_priority == 3
    assert event.published_at is None
    assert event.discarded_at is None
    assert event.publish_attempts == 0
    assert event.publisher_token is None
    assert event.lease_expires_at is None
    assert event.last_error is None
    assert "created_at" not in event.__dict__
    assert "available_at" not in event.__dict__
    flush.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_replay_selects_existing_task_without_writing_outbox(
    inserted_task: TaskModel,
) -> None:
    """A conflicting key is resolved by the following READ COMMITTED query."""

    session, scalars, add, flush = _session_with_scalar_results(
        _ScalarResult(one_or_none=None),
        _ScalarResult(one_or_none=inserted_task),
    )
    repository = TaskRepository(session)

    stored = await repository.create_with_outbox(
        name="Daily report",
        description="Aggregate daily metrics",
        priority=TaskPriority.HIGH,
        max_attempts=3,
        idempotency_key_hash=b"k" * 32,
        request_fingerprint=b"f" * 32,
        event_type="task.execute.v1",
        message_priority=3,
    )

    assert stored.task is inserted_task
    assert stored.created is False
    assert scalars.await_count == 2
    select_statement = scalars.await_args_list[1].args[0]
    select_sql = _compiled_sql(select_statement)
    assert "FROM tasks WHERE tasks.idempotency_key_hash =" in select_sql
    assert _compiled_parameters(select_statement) == {"idempotency_key_hash_1": b"k" * 32}
    add.assert_not_called()
    flush.assert_not_awaited()


@pytest.mark.asyncio
async def test_unkeyed_task_is_created_with_null_idempotency_metadata(
    inserted_task: TaskModel,
) -> None:
    """An unkeyed request always follows the create path and emits one event."""

    inserted_task.idempotency_key_hash = None
    inserted_task.request_fingerprint = None
    session, scalars, add, flush = _session_with_scalar_results(
        _ScalarResult(one_or_none=inserted_task)
    )

    stored = await TaskRepository(session).create_with_outbox(
        name="One-off report",
        description="Run without a replay key",
        priority=TaskPriority.LOW,
        max_attempts=1,
        idempotency_key_hash=None,
        request_fingerprint=None,
        event_type="task.execute.v1",
        message_priority=1,
    )

    assert stored == StoredTaskCreation(task=inserted_task, created=True)
    assert scalars.await_args is not None
    parameters = _compiled_parameters(scalars.await_args.args[0])
    assert parameters["idempotency_key_hash"] is None
    assert parameters["request_fingerprint"] is None
    add.assert_called_once()
    flush.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_unkeyed_insert_without_returning_row_fails_safely() -> None:
    """An impossible empty result never falls back to a broad NULL lookup."""

    session, scalars, add, flush = _session_with_scalar_results(_ScalarResult(one_or_none=None))
    repository = TaskRepository(session)

    with pytest.raises(
        RuntimeError,
        match=r"^task insertion returned no row without an idempotency key$",
    ):
        await repository.create_with_outbox(
            name="Unkeyed task",
            description="Always create a new task",
            priority=TaskPriority.LOW,
            max_attempts=1,
            idempotency_key_hash=None,
            request_fingerprint=None,
            event_type="task.execute.v1",
            message_priority=1,
        )

    scalars.assert_awaited_once()
    add.assert_not_called()
    flush.assert_not_awaited()


@pytest.mark.asyncio
async def test_replay_without_a_visible_winner_fails_with_invariant_error() -> None:
    """A violated READ COMMITTED assumption produces a diagnostic failure."""

    session, scalars, add, flush = _session_with_scalar_results(
        _ScalarResult(one_or_none=None),
        _ScalarResult(one_or_none=None),
    )

    with pytest.raises(
        RuntimeError,
        match=r"^idempotency conflict was not followed by a visible task row$",
    ):
        await TaskRepository(session).create_with_outbox(
            name="Replay",
            description="Missing winner",
            priority=TaskPriority.MEDIUM,
            max_attempts=3,
            idempotency_key_hash=b"k" * 32,
            request_fingerprint=b"f" * 32,
            event_type="task.execute.v1",
            message_priority=2,
        )

    assert scalars.await_count == 2
    add.assert_not_called()
    flush.assert_not_awaited()


@pytest.mark.asyncio
async def test_created_task_without_dispatch_token_fails_before_outbox(
    inserted_task: TaskModel,
) -> None:
    """A corrupt returned row cannot produce a message with a null token."""

    inserted_task.dispatch_token = None
    session, scalars, add, flush = _session_with_scalar_results(
        _ScalarResult(one_or_none=inserted_task)
    )

    with pytest.raises(RuntimeError, match=r"^created task has no dispatch token$"):
        await TaskRepository(session).create_with_outbox(
            name="Daily report",
            description="Aggregate daily metrics",
            priority=TaskPriority.HIGH,
            max_attempts=3,
            idempotency_key_hash=b"k" * 32,
            request_fingerprint=b"f" * 32,
            event_type="task.execute.v1",
            message_priority=3,
        )

    scalars.assert_awaited_once()
    add.assert_not_called()
    flush.assert_not_awaited()


@pytest.mark.asyncio
async def test_repository_propagates_outbox_flush_failure(
    inserted_task: TaskModel,
) -> None:
    """The transaction owner receives persistence failures and can roll back both rows."""

    session, scalars, add, flush = _session_with_scalar_results(
        _ScalarResult(one_or_none=inserted_task)
    )
    flush.side_effect = OSError("storage unavailable")
    repository = TaskRepository(session)

    with pytest.raises(OSError, match="storage unavailable"):
        await repository.create_with_outbox(
            name="Daily report",
            description="Aggregate daily metrics",
            priority=TaskPriority.HIGH,
            max_attempts=3,
            idempotency_key_hash=b"k" * 32,
            request_fingerprint=b"f" * 32,
            event_type="task.execute.v1",
            message_priority=3,
        )

    scalars.assert_awaited_once()
    add.assert_called_once()
    flush.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_repository_never_owns_transaction_boundaries(
    inserted_task: TaskModel,
) -> None:
    """Only the application use case may begin, commit, or roll back a transaction."""

    begin = Mock()
    commit = AsyncMock()
    rollback = AsyncMock()
    scalars = AsyncMock(return_value=_ScalarResult(one_or_none=inserted_task))
    session = cast(
        AsyncSession,
        Mock(
            scalars=scalars,
            add=Mock(),
            flush=AsyncMock(),
            begin=begin,
            commit=commit,
            rollback=rollback,
        ),
    )

    await TaskRepository(session).create_with_outbox(
        name="Daily report",
        description="Aggregate daily metrics",
        priority=TaskPriority.MEDIUM,
        max_attempts=2,
        idempotency_key_hash=None,
        request_fingerprint=None,
        event_type="task.execute.v1",
        message_priority=2,
    )

    assert begin.mock_calls == []
    commit.assert_not_awaited()
    rollback.assert_not_awaited()


def test_stored_creation_is_immutable(inserted_task: TaskModel) -> None:
    """Repository outcome has value semantics and cannot drift after construction."""

    outcome = StoredTaskCreation(task=inserted_task, created=True)

    with pytest.raises(FrozenInstanceError):
        outcome.__setattr__("created", False)
