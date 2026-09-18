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
from cims_task_service.infrastructure.database.task_execution_repository import (
    ClaimedTaskExecution,
    TaskExecutionRepository,
)

_POSTGRESQL_DIALECT = PGDialect()  # type: ignore[no-untyped-call]
_TASK_ID = UUID("10000000-0000-4000-8000-000000000001")
_DISPATCH_TOKEN = UUID("20000000-0000-4000-8000-000000000002")
_EXECUTION_TOKEN = UUID("30000000-0000-4000-8000-000000000003")
_LEASE_DURATION = timedelta(minutes=1)
_LEASE_EXPIRES_AT = datetime(2026, 9, 19, 2, 1, tzinfo=UTC)

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


class _RowResult:
    """Minimal row-result double for an execution claim projection."""

    def __init__(self, row: _ClaimRow | None) -> None:
        self._row = row

    def one_or_none(self) -> _ClaimRow | None:
        return self._row


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
