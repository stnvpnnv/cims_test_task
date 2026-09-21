"""Tests for atomic task execution ownership."""

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from typing import cast
from unittest.mock import AsyncMock, Mock
from uuid import UUID

import pytest
from sqlalchemy.dialects.postgresql.base import PGDialect
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import ClauseElement

from cims_task_service.domain.task import TaskPriority, TaskStatus
from cims_task_service.infrastructure.database import task_execution_repository
from cims_task_service.infrastructure.database.models import JsonObject, OutboxEventModel
from cims_task_service.infrastructure.database.task_execution_repository import (
    ClaimedTaskExecution,
    LockedExpiredTaskExecution,
    TaskExecutionRepository,
)
from cims_task_service.infrastructure.messaging.topology import TASK_ROUTING_KEY

_POSTGRESQL_DIALECT = PGDialect()  # type: ignore[no-untyped-call]
_TASK_ID = UUID("10000000-0000-4000-8000-000000000001")
_DISPATCH_TOKEN = UUID("20000000-0000-4000-8000-000000000002")
_EXECUTION_TOKEN = UUID("30000000-0000-4000-8000-000000000003")
_OUTBOX_EVENT_ID = UUID("50000000-0000-4000-8000-000000000005")
_LEASE_DURATION = timedelta(minutes=1)
_LEASE_EXPIRES_AT = datetime(2026, 9, 19, 2, 1, tzinfo=UTC)
_RESULT: JsonObject = {
    "name_length": 12,
    "description_length": 23,
}

type _ClaimRow = tuple[
    UUID,
    str,
    str,
    TaskPriority,
    int,
    int,
    UUID | None,
    datetime | None,
]
type _ExpiredExecutionRow = tuple[
    UUID,
    TaskPriority,
    int,
    int,
    UUID | None,
    datetime | None,
]


class _RowResult:
    """Minimal row-result double for an execution claim projection."""

    def __init__(self, row: _ClaimRow | None) -> None:
        self._row = row

    def one_or_none(self) -> _ClaimRow | None:
        return self._row


class _RowsResult:
    """Minimal row-result double for an expired execution projection."""

    def __init__(self, rows: tuple[_ExpiredExecutionRow, ...]) -> None:
        self._rows = rows

    def all(self) -> list[_ExpiredExecutionRow]:
        return list(self._rows)


def _session(
    row: _ClaimRow | None,
) -> tuple[AsyncSession, AsyncMock, Mock, AsyncMock, AsyncMock]:
    execute = AsyncMock(return_value=_RowResult(row))
    begin = Mock()
    commit = AsyncMock()
    rollback = AsyncMock()
    session = cast(
        AsyncSession,
        Mock(execute=execute, begin=begin, commit=commit, rollback=rollback),
    )
    return session, execute, begin, commit, rollback


def _compiled_sql(statement: ClauseElement) -> str:
    return " ".join(str(statement.compile(dialect=_POSTGRESQL_DIALECT)).split())


def _compiled_parameters(statement: ClauseElement) -> dict[str, object]:
    return cast(
        dict[str, object],
        statement.compile(dialect=_POSTGRESQL_DIALECT).params,
    )


def _mutation_session(stored_task_id: UUID | None) -> Mock:
    return Mock(
        scalar=AsyncMock(return_value=stored_task_id),
        add=Mock(),
        flush=AsyncMock(),
        begin=Mock(),
        commit=AsyncMock(),
        rollback=AsyncMock(),
    )


def _expired_execution_session(
    rows: tuple[_ExpiredExecutionRow, ...],
) -> tuple[AsyncSession, AsyncMock, Mock, AsyncMock, AsyncMock]:
    execute = AsyncMock(return_value=_RowsResult(rows))
    begin = Mock()
    commit = AsyncMock()
    rollback = AsyncMock()
    session = cast(
        AsyncSession,
        Mock(execute=execute, begin=begin, commit=commit, rollback=rollback),
    )
    return session, execute, begin, commit, rollback


@pytest.mark.asyncio
async def test_expired_batch_locks_oldest_executions_and_returns_frozen_snapshots() -> None:
    """Recovery receives only the projected state while its caller retains row locks."""

    second_task_id = UUID("10000000-0000-4000-8000-000000000002")
    second_execution_token = UUID("30000000-0000-4000-8000-000000000004")
    second_lease_expires_at = _LEASE_EXPIRES_AT + timedelta(seconds=1)
    rows: tuple[_ExpiredExecutionRow, ...] = (
        (_TASK_ID, TaskPriority.HIGH, 1, 3, _EXECUTION_TOKEN, _LEASE_EXPIRES_AT),
        (
            second_task_id,
            TaskPriority.LOW,
            2,
            3,
            second_execution_token,
            second_lease_expires_at,
        ),
    )
    session, execute, begin, commit, rollback = _expired_execution_session(rows)

    locked = await TaskExecutionRepository(session).lock_expired_execution_batch(batch_size=2)

    assert locked == (
        LockedExpiredTaskExecution(
            task_id=_TASK_ID,
            priority=TaskPriority.HIGH,
            attempt_count=1,
            max_attempts=3,
            execution_token=_EXECUTION_TOKEN,
            lease_expires_at=_LEASE_EXPIRES_AT,
        ),
        LockedExpiredTaskExecution(
            task_id=second_task_id,
            priority=TaskPriority.LOW,
            attempt_count=2,
            max_attempts=3,
            execution_token=second_execution_token,
            lease_expires_at=second_lease_expires_at,
        ),
    )
    with pytest.raises(FrozenInstanceError):
        locked[0].attempt_count = 2  # type: ignore[misc]

    execute.assert_awaited_once()
    assert execute.await_args is not None
    statement = execute.await_args.args[0]
    sql = _compiled_sql(statement)
    assert sql.startswith(
        "SELECT tasks.id, tasks.priority, tasks.attempt_count, tasks.max_attempts, "
        "tasks.execution_token, tasks.lease_expires_at FROM tasks"
    )
    assert "tasks.status =" in sql
    assert "tasks.lease_expires_at <= statement_timestamp()" in sql
    assert "ORDER BY tasks.lease_expires_at, tasks.id" in sql
    assert " LIMIT " in sql
    assert sql.endswith("FOR UPDATE OF tasks SKIP LOCKED")

    parameters = _compiled_parameters(statement)
    assert parameters["status_1"] is TaskStatus.IN_PROGRESS
    assert 2 in parameters.values()
    begin.assert_not_called()
    commit.assert_not_awaited()
    rollback.assert_not_awaited()


@pytest.mark.asyncio
async def test_expired_batch_returns_empty_when_no_execution_is_eligible() -> None:
    """An idle recovery pass acquires no task state or transaction ownership."""

    session, execute, begin, commit, rollback = _expired_execution_session(())

    locked = await TaskExecutionRepository(session).lock_expired_execution_batch(batch_size=10)

    assert locked == ()
    execute.assert_awaited_once()
    begin.assert_not_called()
    commit.assert_not_awaited()
    rollback.assert_not_awaited()


@pytest.mark.parametrize("batch_size", [0, -1])
@pytest.mark.asyncio
async def test_expired_batch_rejects_a_non_positive_limit_before_database_access(
    batch_size: int,
) -> None:
    """Recovery cannot accidentally acquire an unbounded set of task locks."""

    session, execute, begin, commit, rollback = _expired_execution_session(())

    with pytest.raises(ValueError, match=r"^batch_size must be at least 1$"):
        await TaskExecutionRepository(session).lock_expired_execution_batch(batch_size=batch_size)

    execute.assert_not_awaited()
    begin.assert_not_called()
    commit.assert_not_awaited()
    rollback.assert_not_awaited()


@pytest.mark.parametrize(
    ("execution_token", "lease_expires_at"),
    [(None, _LEASE_EXPIRES_AT), (_EXECUTION_TOKEN, None)],
)
@pytest.mark.asyncio
async def test_expired_batch_rejects_a_row_without_complete_execution_ownership(
    execution_token: UUID | None,
    lease_expires_at: datetime | None,
) -> None:
    """A broken database invariant aborts the surrounding recovery transaction."""

    row: _ExpiredExecutionRow = (
        _TASK_ID,
        TaskPriority.HIGH,
        1,
        3,
        execution_token,
        lease_expires_at,
    )
    session, execute, _, _, _ = _expired_execution_session((row,))

    with pytest.raises(
        RuntimeError,
        match=r"^expired task execution is missing its lease$",
    ):
        await TaskExecutionRepository(session).lock_expired_execution_batch(batch_size=1)

    execute.assert_awaited_once()


@pytest.mark.parametrize(
    ("stored_task_id", "expected"),
    [(_TASK_ID, True), (None, False)],
)
@pytest.mark.asyncio
async def test_lease_renewal_only_extends_the_current_owners_deadline(
    stored_task_id: UUID | None,
    expected: bool,
) -> None:
    """One fenced update uses database time without changing execution metadata."""

    session = _mutation_session(stored_task_id)

    renewed = await TaskExecutionRepository(cast(AsyncSession, session)).renew_execution_lease(
        _TASK_ID,
        execution_token=_EXECUTION_TOKEN,
        lease_duration=_LEASE_DURATION,
    )

    assert renewed is expected
    session.scalar.assert_awaited_once()
    statement = session.scalar.await_args.args[0]
    sql = _compiled_sql(statement)
    set_sql, where_sql = sql.split(" WHERE ", maxsplit=1)
    assert set_sql.startswith(
        "UPDATE tasks SET lease_expires_at=greatest(tasks.lease_expires_at, clock_timestamp() +"
    )
    for field in (
        "status",
        "started_at",
        "finished_at",
        "attempt_count",
        "max_attempts",
        "dispatch_token",
        "execution_token",
        "result",
        "error",
    ):
        assert f"{field}=" not in set_sql
    assert "tasks.id =" in where_sql
    assert "tasks.status =" in where_sql
    assert "tasks.execution_token =" in where_sql
    assert "lease_expires_at" not in where_sql
    assert "clock_timestamp" not in where_sql
    assert "attempt_count" not in where_sql
    assert "max_attempts" not in where_sql
    assert sql.endswith("RETURNING tasks.id")

    parameters = _compiled_parameters(statement)
    assert parameters["id_1"] == _TASK_ID
    assert parameters["status_1"] is TaskStatus.IN_PROGRESS
    assert parameters["execution_token_1"] == _EXECUTION_TOKEN
    assert _LEASE_DURATION in parameters.values()
    session.add.assert_not_called()
    session.flush.assert_not_awaited()
    session.begin.assert_not_called()
    session.commit.assert_not_awaited()
    session.rollback.assert_not_awaited()


@pytest.mark.parametrize("lease_duration", [timedelta(0), timedelta(microseconds=-1)])
@pytest.mark.asyncio
async def test_lease_renewal_rejects_a_non_positive_duration_before_database_access(
    lease_duration: timedelta,
) -> None:
    """Invalid heartbeat settings cannot alter the existing lease."""

    session = _mutation_session(_TASK_ID)

    with pytest.raises(ValueError, match=r"^lease_duration must be positive$"):
        await TaskExecutionRepository(cast(AsyncSession, session)).renew_execution_lease(
            _TASK_ID,
            execution_token=_EXECUTION_TOKEN,
            lease_duration=lease_duration,
        )

    session.scalar.assert_not_awaited()
    session.begin.assert_not_called()
    session.commit.assert_not_awaited()
    session.rollback.assert_not_awaited()


@pytest.mark.asyncio
async def test_lease_renewal_propagates_database_failure_to_the_caller() -> None:
    """An unavailable database is not reported as lost execution ownership."""

    session = _mutation_session(_TASK_ID)
    session.scalar.side_effect = TimeoutError("database timeout")

    with pytest.raises(TimeoutError, match=r"^database timeout$"):
        await TaskExecutionRepository(cast(AsyncSession, session)).renew_execution_lease(
            _TASK_ID,
            execution_token=_EXECUTION_TOKEN,
            lease_duration=_LEASE_DURATION,
        )

    session.scalar.assert_awaited_once()
    session.begin.assert_not_called()
    session.commit.assert_not_awaited()
    session.rollback.assert_not_awaited()


@pytest.mark.parametrize("retry_delay", [timedelta(0), timedelta(seconds=15)])
@pytest.mark.parametrize(
    ("stored_task_id", "expected"),
    [(_TASK_ID, True), (None, False)],
)
@pytest.mark.asyncio
async def test_retry_fences_the_transition_and_enqueues_only_for_the_winner(
    monkeypatch: pytest.MonkeyPatch,
    retry_delay: timedelta,
    stored_task_id: UUID | None,
    expected: bool,
) -> None:
    """Retry creates a fresh delayed dispatch without consuming another attempt."""

    session = _mutation_session(stored_task_id)
    token_factory = Mock(side_effect=[_DISPATCH_TOKEN, _OUTBOX_EVENT_ID])
    monkeypatch.setattr(task_execution_repository, "uuid4", token_factory)

    scheduled = await TaskExecutionRepository(cast(AsyncSession, session)).schedule_execution_retry(
        _TASK_ID,
        execution_token=_EXECUTION_TOKEN,
        retry_delay=retry_delay,
        event_type=TASK_ROUTING_KEY,
        message_priority=3,
    )

    assert scheduled is expected
    session.scalar.assert_awaited_once()
    statement = session.scalar.await_args.args[0]
    sql = _compiled_sql(statement)
    set_sql, where_sql = sql.split(" WHERE ", maxsplit=1)
    assert set_sql.startswith("UPDATE tasks SET status=")
    assert "attempt_count" not in set_sql
    assert "max_attempts" not in set_sql
    assert "started_at" not in set_sql
    assert "finished_at" not in set_sql
    assert "tasks.id =" in where_sql
    assert "tasks.status =" in where_sql
    assert "tasks.execution_token =" in where_sql
    assert "tasks.attempt_count < tasks.max_attempts" in where_sql
    assert "lease_expires_at" not in where_sql
    assert "clock_timestamp" not in where_sql
    assert sql.endswith("RETURNING tasks.id")

    parameters = _compiled_parameters(statement)
    assert parameters["id_1"] == _TASK_ID
    assert parameters["status_1"] is TaskStatus.IN_PROGRESS
    assert parameters["execution_token_1"] == _EXECUTION_TOKEN
    assert parameters["status"] is TaskStatus.PENDING
    assert parameters["dispatch_token"] == _DISPATCH_TOKEN
    assert parameters["execution_token"] is None
    assert parameters["lease_expires_at"] is None

    if expected:
        session.add.assert_called_once()
        event = session.add.call_args.args[0]
        assert isinstance(event, OutboxEventModel)
        assert event.id == _OUTBOX_EVENT_ID
        assert event.task_id == _TASK_ID
        assert event.event_type == TASK_ROUTING_KEY
        assert event.payload == {
            "task_id": str(_TASK_ID),
            "dispatch_token": str(_DISPATCH_TOKEN),
        }
        assert event.message_priority == 3
        available_at = cast(ClauseElement, event.available_at)
        assert _compiled_sql(available_at).startswith("clock_timestamp() +")
        assert retry_delay in _compiled_parameters(available_at).values()
        assert event.publish_attempts == 0
        assert event.published_at is None
        assert event.discarded_at is None
        assert event.publisher_token is None
        assert event.lease_expires_at is None
        assert event.last_error is None
        session.flush.assert_awaited_once_with()
        assert token_factory.call_count == 2
    else:
        session.add.assert_not_called()
        session.flush.assert_not_awaited()
        token_factory.assert_called_once_with()

    session.begin.assert_not_called()
    session.commit.assert_not_awaited()
    session.rollback.assert_not_awaited()


@pytest.mark.asyncio
async def test_retry_rejects_a_negative_delay_before_database_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalid delay cannot mutate the task or create an outbox event."""

    session = _mutation_session(_TASK_ID)
    token_factory = Mock()
    monkeypatch.setattr(task_execution_repository, "uuid4", token_factory)

    with pytest.raises(ValueError, match=r"^retry_delay must not be negative$"):
        await TaskExecutionRepository(cast(AsyncSession, session)).schedule_execution_retry(
            _TASK_ID,
            execution_token=_EXECUTION_TOKEN,
            retry_delay=timedelta(microseconds=-1),
            event_type=TASK_ROUTING_KEY,
            message_priority=3,
        )

    session.scalar.assert_not_awaited()
    session.add.assert_not_called()
    session.flush.assert_not_awaited()
    token_factory.assert_not_called()


@pytest.mark.asyncio
async def test_retry_propagates_outbox_flush_failure_to_the_transaction_owner() -> None:
    """Outbox persistence failure must abort the caller's transaction."""

    session = _mutation_session(_TASK_ID)
    session.flush.side_effect = OSError("storage unavailable")

    with pytest.raises(OSError, match=r"^storage unavailable$"):
        await TaskExecutionRepository(cast(AsyncSession, session)).schedule_execution_retry(
            _TASK_ID,
            execution_token=_EXECUTION_TOKEN,
            retry_delay=timedelta(seconds=15),
            event_type=TASK_ROUTING_KEY,
            message_priority=3,
        )

    session.scalar.assert_awaited_once()
    session.add.assert_called_once()
    session.flush.assert_awaited_once_with()
    session.begin.assert_not_called()
    session.commit.assert_not_awaited()
    session.rollback.assert_not_awaited()


@pytest.mark.asyncio
async def test_claim_uses_one_fenced_transition_and_returns_a_detached_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the matching pending dispatch can acquire a fresh execution lease."""

    row: _ClaimRow = (
        _TASK_ID,
        "Daily report",
        "Aggregate daily metrics",
        TaskPriority.HIGH,
        1,
        3,
        _EXECUTION_TOKEN,
        _LEASE_EXPIRES_AT,
    )
    session, execute, begin, commit, rollback = _session(row)
    monkeypatch.setattr(task_execution_repository, "uuid4", lambda: _EXECUTION_TOKEN)

    claimed = await TaskExecutionRepository(session).claim_for_execution(
        _TASK_ID,
        dispatch_token=_DISPATCH_TOKEN,
        lease_duration=_LEASE_DURATION,
    )

    assert claimed == ClaimedTaskExecution(
        task_id=_TASK_ID,
        name="Daily report",
        description="Aggregate daily metrics",
        priority=TaskPriority.HIGH,
        attempt_count=1,
        max_attempts=3,
        execution_token=_EXECUTION_TOKEN,
        lease_expires_at=_LEASE_EXPIRES_AT,
    )
    with pytest.raises(FrozenInstanceError):
        claimed.attempt_count = 2  # type: ignore[misc]

    execute.assert_awaited_once()
    assert execute.await_args is not None
    statement = execute.await_args.args[0]
    sql = _compiled_sql(statement)
    assert sql.startswith("UPDATE tasks SET status=")
    assert "started_at=coalesce(tasks.started_at, clock_timestamp())" in sql
    assert "attempt_count=(tasks.attempt_count +" in sql
    assert "dispatch_token=" in sql
    assert "execution_token=" in sql
    assert "lease_expires_at=(clock_timestamp() +" in sql
    assert "tasks.id =" in sql
    assert "tasks.status =" in sql
    assert "tasks.dispatch_token =" in sql
    assert "tasks.attempt_count < tasks.max_attempts" in sql
    assert sql.endswith(
        "RETURNING tasks.id, tasks.name, tasks.description, tasks.priority, "
        "tasks.attempt_count, tasks.max_attempts, tasks.execution_token, "
        "tasks.lease_expires_at"
    )

    parameters = _compiled_parameters(statement)
    assert parameters["status"] is TaskStatus.IN_PROGRESS
    assert parameters["status_1"] is TaskStatus.PENDING
    assert parameters["id_1"] == _TASK_ID
    assert parameters["dispatch_token"] is None
    assert parameters["dispatch_token_1"] == _DISPATCH_TOKEN
    assert parameters["execution_token"] == _EXECUTION_TOKEN
    assert parameters["attempt_count_1"] == 1
    assert _LEASE_DURATION in parameters.values()
    begin.assert_not_called()
    commit.assert_not_awaited()
    rollback.assert_not_awaited()


@pytest.mark.asyncio
async def test_claim_returns_none_when_the_compare_and_set_loses() -> None:
    """A duplicate, stale, cancelled, or terminal delivery acquires no ownership."""

    session, execute, begin, commit, rollback = _session(None)

    claimed = await TaskExecutionRepository(session).claim_for_execution(
        _TASK_ID,
        dispatch_token=_DISPATCH_TOKEN,
        lease_duration=_LEASE_DURATION,
    )

    assert claimed is None
    execute.assert_awaited_once()
    begin.assert_not_called()
    commit.assert_not_awaited()
    rollback.assert_not_awaited()


@pytest.mark.parametrize("lease_duration", [timedelta(0), timedelta(microseconds=-1)])
@pytest.mark.asyncio
async def test_claim_rejects_a_non_positive_lease_before_database_access(
    monkeypatch: pytest.MonkeyPatch,
    lease_duration: timedelta,
) -> None:
    """A configuration bug cannot create an already expired execution claim."""

    session, execute, _, _, _ = _session(None)
    token_factory = Mock()
    monkeypatch.setattr(task_execution_repository, "uuid4", token_factory)

    with pytest.raises(ValueError, match=r"^lease_duration must be positive$"):
        await TaskExecutionRepository(session).claim_for_execution(
            _TASK_ID,
            dispatch_token=_DISPATCH_TOKEN,
            lease_duration=lease_duration,
        )

    execute.assert_not_awaited()
    token_factory.assert_not_called()


@pytest.mark.parametrize(
    ("execution_token", "lease_expires_at"),
    [(None, _LEASE_EXPIRES_AT), (_EXECUTION_TOKEN, None)],
)
@pytest.mark.asyncio
async def test_claim_rejects_a_returned_row_without_a_complete_lease(
    execution_token: UUID | None,
    lease_expires_at: datetime | None,
) -> None:
    """A broken database invariant aborts the surrounding claim transaction."""

    row: _ClaimRow = (
        _TASK_ID,
        "Daily report",
        "Aggregate daily metrics",
        TaskPriority.HIGH,
        1,
        3,
        execution_token,
        lease_expires_at,
    )
    session, execute, _, _, _ = _session(row)

    with pytest.raises(
        RuntimeError,
        match=r"^claimed task execution is missing its lease$",
    ):
        await TaskExecutionRepository(session).claim_for_execution(
            _TASK_ID,
            dispatch_token=_DISPATCH_TOKEN,
            lease_duration=_LEASE_DURATION,
        )

    execute.assert_awaited_once()


@pytest.mark.parametrize(
    ("stored_task_id", "expected"),
    [(_TASK_ID, True), (None, False)],
)
@pytest.mark.asyncio
async def test_complete_uses_a_fenced_terminal_transition(
    stored_task_id: UUID | None,
    expected: bool,
) -> None:
    """Only the current execution owner can persist a successful result."""

    scalar = AsyncMock(return_value=stored_task_id)
    begin = Mock()
    commit = AsyncMock()
    rollback = AsyncMock()
    session = cast(
        AsyncSession,
        Mock(scalar=scalar, begin=begin, commit=commit, rollback=rollback),
    )

    completed = await TaskExecutionRepository(session).complete_execution(
        _TASK_ID,
        execution_token=_EXECUTION_TOKEN,
        result=_RESULT,
    )

    assert completed is expected
    scalar.assert_awaited_once()
    assert scalar.await_args is not None
    statement = scalar.await_args.args[0]
    sql = _compiled_sql(statement)
    assert sql.startswith("UPDATE tasks SET status=")
    assert "finished_at=clock_timestamp()" in sql
    assert "result=" in sql
    assert "error=" in sql
    assert "dispatch_token=" in sql
    assert "execution_token=" in sql
    assert "lease_expires_at=" in sql
    assert "tasks.id =" in sql
    assert "tasks.status =" in sql
    assert "tasks.execution_token =" in sql
    assert sql.endswith("RETURNING tasks.id")
    where_sql = sql.split(" WHERE ", maxsplit=1)[1].split(" RETURNING", maxsplit=1)[0]
    assert "lease_expires_at" not in where_sql
    assert "clock_timestamp" not in where_sql

    parameters = _compiled_parameters(statement)
    assert parameters["status"] is TaskStatus.COMPLETED
    assert parameters["status_1"] is TaskStatus.IN_PROGRESS
    assert parameters["id_1"] == _TASK_ID
    assert parameters["execution_token_1"] == _EXECUTION_TOKEN
    assert parameters["result"] == _RESULT
    assert parameters["error"] is None
    assert parameters["dispatch_token"] is None
    assert parameters["execution_token"] is None
    assert parameters["lease_expires_at"] is None
    begin.assert_not_called()
    commit.assert_not_awaited()
    rollback.assert_not_awaited()


@pytest.mark.parametrize(
    ("stored_task_id", "expected"),
    [(_TASK_ID, True), (None, False)],
)
@pytest.mark.asyncio
async def test_fail_uses_a_fenced_terminal_transition(
    stored_task_id: UUID | None,
    expected: bool,
) -> None:
    """An error is persisted only for the current owner, without starting a retry."""

    error: JsonObject = {"code": "PROCESSING_FAILED", "retryable": False}
    scalar = AsyncMock(return_value=stored_task_id)
    begin = Mock()
    commit = AsyncMock()
    rollback = AsyncMock()
    session = cast(
        AsyncSession,
        Mock(scalar=scalar, begin=begin, commit=commit, rollback=rollback),
    )

    failed = await TaskExecutionRepository(session).fail_execution(
        _TASK_ID,
        execution_token=_EXECUTION_TOKEN,
        error=error,
    )

    assert failed is expected
    scalar.assert_awaited_once()
    assert scalar.await_args is not None
    statement = scalar.await_args.args[0]
    sql = _compiled_sql(statement)
    assert sql.startswith("UPDATE tasks SET status=")
    assert "finished_at=clock_timestamp()" in sql
    assert sql.endswith("RETURNING tasks.id")
    where_sql = sql.split(" WHERE ", maxsplit=1)[1].split(" RETURNING", maxsplit=1)[0]
    assert "tasks.id =" in where_sql
    assert "tasks.status =" in where_sql
    assert "tasks.execution_token =" in where_sql
    assert "lease_expires_at" not in where_sql
    assert "clock_timestamp" not in where_sql
    assert "attempt_count" not in sql
    assert "max_attempts" not in sql
    assert "started_at" not in sql

    parameters = _compiled_parameters(statement)
    assert parameters["status"] is TaskStatus.FAILED
    assert parameters["status_1"] is TaskStatus.IN_PROGRESS
    assert parameters["id_1"] == _TASK_ID
    assert parameters["execution_token_1"] == _EXECUTION_TOKEN
    assert parameters["error"] == error
    for cleared_field in ("result", "dispatch_token", "execution_token", "lease_expires_at"):
        assert parameters[cleared_field] is None
    begin.assert_not_called()
    commit.assert_not_awaited()
    rollback.assert_not_awaited()
