"""Tests for lease-based transactional outbox persistence."""

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
from cims_task_service.infrastructure.database import outbox_repository
from cims_task_service.infrastructure.database.models import (
    OutboxEventModel,
    TaskModel,
)
from cims_task_service.infrastructure.database.outbox_repository import (
    MAX_OUTBOX_ERROR_LENGTH,
    OutboxRepository,
)

_POSTGRESQL_DIALECT = PGDialect()  # type: ignore[no-untyped-call]
_CREATED_AT = datetime(2026, 9, 14, 1, 2, 3, tzinfo=UTC)
_PUBLISHER_TOKEN = UUID("4a2e87ca-b2a0-44a5-8a35-d8ecf9e80689")


class _CandidateResult:
    """Minimal row-result double for joined task and outbox candidates."""

    def __init__(self, rows: tuple[tuple[UUID, TaskStatus, UUID], ...]) -> None:
        self._rows = rows

    def all(self) -> tuple[tuple[UUID, TaskStatus, UUID], ...]:
        return self._rows


class _OutboxScalarResult:
    """Minimal scalar-result double for claimed outbox rows."""

    def __init__(self, events: tuple[OutboxEventModel, ...]) -> None:
        self._events = events

    def all(self) -> tuple[OutboxEventModel, ...]:
        return self._events


def _task(task_id: UUID, *, status: TaskStatus = TaskStatus.NEW) -> TaskModel:
    dispatch_token = UUID(int=task_id.int ^ 1)
    started = status in {TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED, TaskStatus.FAILED}
    finished = status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
    return TaskModel(
        id=task_id,
        name="Daily report",
        description="Aggregate daily metrics",
        priority=TaskPriority.HIGH,
        status=status,
        created_at=_CREATED_AT,
        idempotency_key_hash=None,
        request_fingerprint=None,
        started_at=_CREATED_AT if started else None,
        finished_at=_CREATED_AT + timedelta(seconds=1) if finished else None,
        result={"records": 42} if status is TaskStatus.COMPLETED else None,
        error={"code": "processing_failed"} if status is TaskStatus.FAILED else None,
        attempt_count=1 if started else 0,
        max_attempts=3,
        dispatch_token=dispatch_token if status in {TaskStatus.NEW, TaskStatus.PENDING} else None,
        execution_token=dispatch_token if status is TaskStatus.IN_PROGRESS else None,
        lease_expires_at=(
            _CREATED_AT + timedelta(minutes=1) if status is TaskStatus.IN_PROGRESS else None
        ),
    )


def _event(
    event_id: UUID,
    task: TaskModel,
    *,
    message_priority: int,
    publish_attempts: int = 0,
    publisher_token: UUID | None = None,
    lease_expires_at: datetime | None = None,
) -> OutboxEventModel:
    dispatch_token = task.dispatch_token or UUID(int=task.id.int ^ 1)
    return OutboxEventModel(
        id=event_id,
        task_id=task.id,
        event_type="task.execute.v1",
        payload={
            "task_id": str(task.id),
            "dispatch_token": str(dispatch_token),
        },
        message_priority=message_priority,
        created_at=_CREATED_AT,
        available_at=_CREATED_AT,
        published_at=None,
        discarded_at=None,
        publish_attempts=publish_attempts,
        publisher_token=publisher_token,
        lease_expires_at=lease_expires_at,
        last_error=None,
    )


def _compiled_sql(statement: ClauseElement) -> str:
    return " ".join(str(statement.compile(dialect=_POSTGRESQL_DIALECT)).split())


def _compiled_parameters(statement: ClauseElement) -> dict[str, object]:
    return cast(
        dict[str, object],
        statement.compile(dialect=_POSTGRESQL_DIALECT).params,
    )


def _session(
    *,
    execute: AsyncMock | None = None,
    scalars: AsyncMock | None = None,
    scalar: AsyncMock | None = None,
) -> tuple[AsyncSession, Mock, AsyncMock, AsyncMock]:
    begin = Mock()
    commit = AsyncMock()
    rollback = AsyncMock()
    session = cast(
        AsyncSession,
        Mock(
            execute=execute or AsyncMock(),
            scalars=scalars or AsyncMock(),
            scalar=scalar or AsyncMock(),
            begin=begin,
            commit=commit,
            rollback=rollback,
        ),
    )
    return session, begin, commit, rollback


@pytest.mark.parametrize(
    ("batch_size", "lease_duration", "message"),
    [
        (0, timedelta(seconds=1), "batch_size must be at least 1"),
        (-1, timedelta(seconds=1), "batch_size must be at least 1"),
        (1, timedelta(0), "lease_duration must be positive"),
        (1, timedelta(microseconds=-1), "lease_duration must be positive"),
    ],
)
@pytest.mark.asyncio
async def test_claim_batch_rejects_invalid_reservation_options(
    batch_size: int,
    lease_duration: timedelta,
    message: str,
) -> None:
    """Invalid limits fail before the repository can access PostgreSQL."""

    execute = AsyncMock()
    session, _, _, _ = _session(execute=execute)

    with pytest.raises(ValueError, match=f"^{message}$"):
        await OutboxRepository(session).claim_batch(
            event_type="task.execute.v1",
            batch_size=batch_size,
            lease_duration=lease_duration,
        )

    execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_claim_batch_reserves_in_priority_order_and_moves_new_tasks_to_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reservation locks tasks first and detaches ordered publication data."""

    high_task = _task(UUID("10000000-0000-4000-8000-000000000001"))
    low_task = _task(UUID("20000000-0000-4000-8000-000000000002"))
    high_candidate = _event(
        UUID("30000000-0000-4000-8000-000000000003"),
        high_task,
        message_priority=3,
    )
    low_candidate = _event(
        UUID("40000000-0000-4000-8000-000000000004"),
        low_task,
        message_priority=1,
    )
    lease_expires_at = _CREATED_AT + timedelta(minutes=1)
    high_claimed = _event(
        high_candidate.id,
        high_task,
        message_priority=3,
        publish_attempts=1,
        publisher_token=_PUBLISHER_TOKEN,
        lease_expires_at=lease_expires_at,
    )
    low_claimed = _event(
        low_candidate.id,
        low_task,
        message_priority=1,
        publish_attempts=1,
        publisher_token=_PUBLISHER_TOKEN,
        lease_expires_at=lease_expires_at,
    )
    candidate_result = _CandidateResult(
        (
            (high_task.id, high_task.status, high_candidate.id),
            (low_task.id, low_task.status, low_candidate.id),
        )
    )
    execute = AsyncMock(side_effect=[candidate_result, Mock()])
    scalars = AsyncMock(
        return_value=_OutboxScalarResult((low_claimed, high_claimed)),
    )
    session, begin, commit, rollback = _session(execute=execute, scalars=scalars)
    monkeypatch.setattr(outbox_repository, "uuid4", lambda: _PUBLISHER_TOKEN)

    claimed = await OutboxRepository(session).claim_batch(
        event_type="task.execute.v1",
        batch_size=20,
        lease_duration=timedelta(minutes=1),
    )

    assert [event.id for event in claimed] == [high_candidate.id, low_candidate.id]
    assert claimed[0].task_id == high_task.id
    assert claimed[0].event_type == "task.execute.v1"
    assert claimed[0].payload == high_candidate.payload
    assert claimed[0].payload is not high_claimed.payload
    assert claimed[0].message_priority == 3
    assert claimed[0].created_at == _CREATED_AT
    assert claimed[0].available_at == _CREATED_AT
    assert claimed[0].publish_attempts == 1
    assert claimed[0].publisher_token == _PUBLISHER_TOKEN
    assert claimed[0].lease_expires_at == lease_expires_at
    with pytest.raises(FrozenInstanceError):
        claimed[0].event_type = "changed"  # type: ignore[misc]

    candidate_statement = execute.await_args_list[0].args[0]
    candidate_sql = _compiled_sql(candidate_statement)
    assert candidate_sql.startswith("SELECT tasks.id, tasks.status, outbox_events.id AS id_1 ")
    assert "FROM tasks JOIN outbox_events ON outbox_events.task_id = tasks.id" in candidate_sql
    assert "outbox_events.event_type =" in candidate_sql
    assert "outbox_events.available_at <= statement_timestamp()" in candidate_sql
    assert "outbox_events.published_at IS NULL" in candidate_sql
    assert "outbox_events.discarded_at IS NULL" in candidate_sql
    assert "outbox_events.publisher_token IS NULL" in candidate_sql
    assert "outbox_events.lease_expires_at <= statement_timestamp()" in candidate_sql
    assert "tasks.status !=" in candidate_sql
    assert (
        "ORDER BY outbox_events.message_priority DESC, outbox_events.available_at, "
        "outbox_events.created_at, outbox_events.id" in candidate_sql
    )
    assert candidate_sql.endswith("FOR UPDATE OF tasks SKIP LOCKED")
    candidate_parameters = _compiled_parameters(candidate_statement)
    assert "task.execute.v1" in candidate_parameters.values()
    assert 20 in candidate_parameters.values()
    assert TaskStatus.CANCELLED in candidate_parameters.values()

    assert scalars.await_args is not None
    claim_statement = scalars.await_args.args[0]
    claim_sql = _compiled_sql(claim_statement)
    assert claim_sql.startswith("UPDATE outbox_events SET publish_attempts=")
    assert "outbox_events.publish_attempts +" in claim_sql
    assert "publisher_token=" in claim_sql
    assert "lease_expires_at=(clock_timestamp() +" in claim_sql
    assert "outbox_events.id IN (__[POSTCOMPILE_id_1])" in claim_sql
    assert "outbox_events.available_at <= clock_timestamp()" in claim_sql
    assert "outbox_events.published_at IS NULL" in claim_sql
    assert "outbox_events.discarded_at IS NULL" in claim_sql
    assert "outbox_events.publisher_token IS NULL" in claim_sql
    assert "outbox_events.lease_expires_at <= clock_timestamp()" in claim_sql
    assert "RETURNING outbox_events.id, outbox_events.task_id" in claim_sql
    assert claim_statement.get_execution_options()["populate_existing"] is True
    claim_parameters = _compiled_parameters(claim_statement)
    assert claim_parameters["publisher_token"] == _PUBLISHER_TOKEN
    assert timedelta(minutes=1) in claim_parameters.values()
    assert set(cast(list[UUID], claim_parameters["id_1"])) == {
        high_candidate.id,
        low_candidate.id,
    }

    pending_statement = execute.await_args_list[1].args[0]
    pending_sql = _compiled_sql(pending_statement)
    assert pending_sql.startswith("UPDATE tasks SET status=")
    assert "tasks.id IN (__[POSTCOMPILE_id_1])" in pending_sql
    assert "tasks.status =" in pending_sql
    pending_parameters = _compiled_parameters(pending_statement)
    assert set(cast(list[UUID], pending_parameters["id_1"])) == {
        high_task.id,
        low_task.id,
    }
    assert TaskStatus.NEW in pending_parameters.values()
    assert TaskStatus.PENDING in pending_parameters.values()

    begin.assert_not_called()
    commit.assert_not_awaited()
    rollback.assert_not_awaited()


@pytest.mark.asyncio
async def test_claim_batch_returns_empty_when_no_candidate_is_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An idle poll performs no write and does not allocate a publisher token."""

    execute = AsyncMock(return_value=_CandidateResult(()))
    scalars = AsyncMock()
    session, begin, commit, rollback = _session(execute=execute, scalars=scalars)
    token_factory = Mock()
    monkeypatch.setattr(outbox_repository, "uuid4", token_factory)

    claimed = await OutboxRepository(session).claim_batch(
        event_type="task.execute.v1",
        batch_size=10,
        lease_duration=timedelta(seconds=30),
    )

    assert claimed == ()
    execute.assert_awaited_once()
    scalars.assert_not_awaited()
    token_factory.assert_not_called()
    begin.assert_not_called()
    commit.assert_not_awaited()
    rollback.assert_not_awaited()


@pytest.mark.asyncio
async def test_claim_batch_moves_only_tasks_whose_events_were_claimed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A concurrent outbox change cannot advance an event that lost its CAS."""

    lost_task = _task(UUID("50000000-0000-4000-8000-000000000005"))
    won_task = _task(UUID("60000000-0000-4000-8000-000000000006"))
    lost_event = _event(
        UUID("70000000-0000-4000-8000-000000000007"),
        lost_task,
        message_priority=3,
    )
    won_event = _event(
        UUID("80000000-0000-4000-8000-000000000008"),
        won_task,
        message_priority=2,
    )
    claimed_event = _event(
        won_event.id,
        won_task,
        message_priority=2,
        publish_attempts=1,
        publisher_token=_PUBLISHER_TOKEN,
        lease_expires_at=_CREATED_AT + timedelta(minutes=1),
    )
    execute = AsyncMock(
        side_effect=[
            _CandidateResult(
                (
                    (lost_task.id, lost_task.status, lost_event.id),
                    (won_task.id, won_task.status, won_event.id),
                )
            ),
            Mock(),
        ]
    )
    scalars = AsyncMock(return_value=_OutboxScalarResult((claimed_event,)))
    session, _, _, _ = _session(execute=execute, scalars=scalars)
    monkeypatch.setattr(outbox_repository, "uuid4", lambda: _PUBLISHER_TOKEN)

    claimed = await OutboxRepository(session).claim_batch(
        event_type="task.execute.v1",
        batch_size=2,
        lease_duration=timedelta(minutes=1),
    )

    assert [event.id for event in claimed] == [won_event.id]
    pending_parameters = _compiled_parameters(execute.await_args_list[1].args[0])
    assert tuple(cast(list[UUID], pending_parameters["id_1"])) == (won_task.id,)
    assert lost_task.id not in pending_parameters.values()


@pytest.mark.asyncio
async def test_claim_batch_returns_empty_when_all_candidate_events_lose_the_cas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No task advances when every candidate became unavailable before UPDATE."""

    task = _task(UUID("90000000-0000-4000-8000-000000000009"))
    event = _event(
        UUID("a0000000-0000-4000-8000-00000000000a"),
        task,
        message_priority=2,
    )
    execute = AsyncMock(return_value=_CandidateResult(((task.id, task.status, event.id),)))
    scalars = AsyncMock(return_value=_OutboxScalarResult(()))
    session, _, _, _ = _session(execute=execute, scalars=scalars)
    monkeypatch.setattr(outbox_repository, "uuid4", lambda: _PUBLISHER_TOKEN)

    claimed = await OutboxRepository(session).claim_batch(
        event_type="task.execute.v1",
        batch_size=1,
        lease_duration=timedelta(seconds=30),
    )

    assert claimed == ()
    execute.assert_awaited_once()
    scalars.assert_awaited_once()


@pytest.mark.asyncio
async def test_claim_batch_recovers_non_cancelled_events_without_rewriting_task_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lost confirm remains recoverable after a task has advanced beyond NEW."""

    statuses = (
        TaskStatus.PENDING,
        TaskStatus.IN_PROGRESS,
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
    )
    tasks = tuple(
        _task(UUID(f"{index}0000000-0000-4000-8000-00000000000{index}"), status=status)
        for index, status in enumerate(statuses, start=1)
    )
    candidate_events = tuple(
        _event(
            UUID(f"{index + 4}0000000-0000-4000-8000-00000000000{index + 4}"),
            task,
            message_priority=3,
        )
        for index, task in enumerate(tasks, start=1)
    )
    claimed_events = tuple(
        _event(
            event.id,
            task,
            message_priority=3,
            publish_attempts=2,
            publisher_token=_PUBLISHER_TOKEN,
            lease_expires_at=_CREATED_AT + timedelta(minutes=1),
        )
        for task, event in zip(tasks, candidate_events, strict=True)
    )
    execute = AsyncMock(
        return_value=_CandidateResult(
            tuple(
                (task.id, task.status, event.id)
                for task, event in zip(tasks, candidate_events, strict=True)
            )
        )
    )
    scalars = AsyncMock(return_value=_OutboxScalarResult(claimed_events))
    session, _, _, _ = _session(execute=execute, scalars=scalars)
    monkeypatch.setattr(outbox_repository, "uuid4", lambda: _PUBLISHER_TOKEN)

    claimed = await OutboxRepository(session).claim_batch(
        event_type="task.execute.v1",
        batch_size=4,
        lease_duration=timedelta(minutes=1),
    )

    assert [event.id for event in claimed] == [event.id for event in candidate_events]
    assert all(event.publish_attempts == 2 for event in claimed)
    execute.assert_awaited_once()
    assert execute.await_args is not None
    candidate_parameters = _compiled_parameters(execute.await_args.args[0])
    assert TaskStatus.CANCELLED in candidate_parameters.values()
    assert TaskStatus.IN_PROGRESS not in candidate_parameters.values()
    assert TaskStatus.COMPLETED not in candidate_parameters.values()
    assert TaskStatus.FAILED not in candidate_parameters.values()


@pytest.mark.parametrize(
    ("publisher_token", "lease_expires_at"),
    [
        (None, _CREATED_AT + timedelta(minutes=1)),
        (_PUBLISHER_TOKEN, None),
    ],
)
@pytest.mark.asyncio
async def test_claim_batch_rejects_returned_event_without_complete_lease(
    monkeypatch: pytest.MonkeyPatch,
    publisher_token: UUID | None,
    lease_expires_at: datetime | None,
) -> None:
    """A broken database invariant aborts the reservation transaction."""

    task = _task(
        UUID("90000000-0000-4000-8000-000000000019"),
        status=TaskStatus.PENDING,
    )
    candidate = _event(
        UUID("a0000000-0000-4000-8000-00000000001a"),
        task,
        message_priority=2,
    )
    invalid_claim = _event(
        candidate.id,
        task,
        message_priority=2,
        publish_attempts=1,
        publisher_token=publisher_token,
        lease_expires_at=lease_expires_at,
    )
    execute = AsyncMock(return_value=_CandidateResult(((task.id, task.status, candidate.id),)))
    scalars = AsyncMock(return_value=_OutboxScalarResult((invalid_claim,)))
    session, _, _, _ = _session(execute=execute, scalars=scalars)
    monkeypatch.setattr(outbox_repository, "uuid4", lambda: _PUBLISHER_TOKEN)

    with pytest.raises(
        RuntimeError,
        match=r"^claimed outbox event is missing its publisher lease$",
    ):
        await OutboxRepository(session).claim_batch(
            event_type="task.execute.v1",
            batch_size=1,
            lease_duration=timedelta(minutes=1),
        )

    execute.assert_awaited_once()


@pytest.mark.parametrize(("stored_id", "expected"), [(_PUBLISHER_TOKEN, True), (None, False)])
@pytest.mark.asyncio
async def test_mark_published_uses_fenced_compare_and_set(
    stored_id: UUID | None,
    expected: bool,
) -> None:
    """Only the current lease owner can record a broker confirmation."""

    event_id = UUID("b0000000-0000-4000-8000-00000000000b")
    scalar = AsyncMock(return_value=stored_id)
    session, begin, commit, rollback = _session(scalar=scalar)

    marked = await OutboxRepository(session).mark_published(
        event_id,
        publisher_token=_PUBLISHER_TOKEN,
    )

    assert marked is expected
    assert scalar.await_args is not None
    statement = scalar.await_args.args[0]
    sql = _compiled_sql(statement)
    assert sql.startswith("UPDATE outbox_events SET published_at=clock_timestamp()")
    assert "publisher_token=" in sql
    assert "lease_expires_at=" in sql
    assert "last_error=" in sql
    assert "outbox_events.id =" in sql
    assert "outbox_events.publisher_token =" in sql
    assert "outbox_events.published_at IS NULL" in sql
    assert "outbox_events.discarded_at IS NULL" in sql
    assert sql.endswith("RETURNING outbox_events.id")
    where_sql = sql.split(" WHERE ", maxsplit=1)[1].split(" RETURNING", maxsplit=1)[0]
    assert "lease_expires_at" not in where_sql
    assert "clock_timestamp" not in where_sql
    parameters = _compiled_parameters(statement)
    assert event_id in parameters.values()
    assert _PUBLISHER_TOKEN in parameters.values()
    assert parameters["publisher_token"] is None
    assert parameters["lease_expires_at"] is None
    assert parameters["last_error"] is None
    begin.assert_not_called()
    commit.assert_not_awaited()
    rollback.assert_not_awaited()


@pytest.mark.parametrize(("stored_id", "expected"), [(_PUBLISHER_TOKEN, True), (None, False)])
@pytest.mark.asyncio
async def test_reschedule_uses_fencing_database_time_and_bounded_error(
    stored_id: UUID | None,
    expected: bool,
) -> None:
    """A failed publish is deferred without allowing a stale owner to mutate it."""

    event_id = UUID("c0000000-0000-4000-8000-00000000000c")
    retry_delay = timedelta(seconds=7)
    failure_summary = "x" * (MAX_OUTBOX_ERROR_LENGTH + 50)
    scalar = AsyncMock(return_value=stored_id)
    session, begin, commit, rollback = _session(scalar=scalar)

    rescheduled = await OutboxRepository(session).reschedule(
        event_id,
        publisher_token=_PUBLISHER_TOKEN,
        retry_delay=retry_delay,
        failure_summary=failure_summary,
    )

    assert rescheduled is expected
    assert scalar.await_args is not None
    statement = scalar.await_args.args[0]
    sql = _compiled_sql(statement)
    assert sql.startswith("UPDATE outbox_events SET available_at=(clock_timestamp() +")
    assert "publisher_token=" in sql
    assert "lease_expires_at=" in sql
    assert "last_error=" in sql
    assert "outbox_events.id =" in sql
    assert "outbox_events.publisher_token =" in sql
    assert "outbox_events.published_at IS NULL" in sql
    assert "outbox_events.discarded_at IS NULL" in sql
    assert sql.endswith("RETURNING outbox_events.id")
    where_sql = sql.split(" WHERE ", maxsplit=1)[1].split(" RETURNING", maxsplit=1)[0]
    assert "lease_expires_at" not in where_sql
    assert "clock_timestamp" not in where_sql
    parameters = _compiled_parameters(statement)
    assert retry_delay in parameters.values()
    assert parameters["last_error"] == failure_summary[:MAX_OUTBOX_ERROR_LENGTH]
    assert parameters["publisher_token"] is None
    assert parameters["lease_expires_at"] is None
    begin.assert_not_called()
    commit.assert_not_awaited()
    rollback.assert_not_awaited()


@pytest.mark.asyncio
async def test_reschedule_rejects_negative_retry_delay_before_database_access() -> None:
    """A scheduling bug cannot move the next attempt into the past."""

    scalar = AsyncMock()
    session, _, _, _ = _session(scalar=scalar)

    with pytest.raises(ValueError, match=r"^retry_delay must not be negative$"):
        await OutboxRepository(session).reschedule(
            UUID("d0000000-0000-4000-8000-00000000000d"),
            publisher_token=_PUBLISHER_TOKEN,
            retry_delay=timedelta(microseconds=-1),
            failure_summary="broker unavailable",
        )

    scalar.assert_not_awaited()
