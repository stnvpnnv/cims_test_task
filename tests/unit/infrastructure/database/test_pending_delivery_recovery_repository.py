"""Tests for the fenced outbox replay query used by delivery recovery."""

from datetime import timedelta
from typing import cast
from unittest.mock import AsyncMock, Mock
from uuid import UUID

import pytest
from sqlalchemy.dialects.postgresql.base import PGDialect
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import ClauseElement

from cims_task_service.domain.task import TaskStatus
from cims_task_service.infrastructure.database.outbox_repository import OutboxRepository
from cims_task_service.infrastructure.messaging.topology import TASK_ROUTING_KEY

_POSTGRESQL_DIALECT = PGDialect()  # type: ignore[no-untyped-call]
_DELIVERY_TIMEOUT = timedelta(minutes=5)


def _sql(statement: ClauseElement) -> str:
    return " ".join(str(statement.compile(dialect=_POSTGRESQL_DIALECT)).split())


def _parameters(statement: ClauseElement) -> dict[str, object]:
    return cast(dict[str, object], statement.compile(dialect=_POSTGRESQL_DIALECT).params)


def _session(scalars: AsyncMock) -> AsyncSession:
    return cast(
        AsyncSession,
        Mock(scalars=scalars, begin=Mock(), commit=AsyncMock(), rollback=AsyncMock()),
    )


def _require_current_pending_delivery_predicates(statement: ClauseElement) -> None:
    sql = _sql(statement)
    values = tuple(_parameters(statement).values())
    assert "outbox_events.task_id = tasks.id" in sql
    assert "tasks.status =" in sql
    assert TaskStatus.PENDING in values
    assert "outbox_events.event_type =" in sql
    assert TASK_ROUTING_KEY in values
    assert "outbox_events.published_at IS NOT NULL" in sql
    assert "outbox_events.discarded_at IS NULL" in sql
    assert "outbox_events.published_at <=" in sql
    assert "outbox_events.available_at <=" in sql
    assert "CAST(tasks.id AS VARCHAR)" in sql
    assert "CAST(tasks.dispatch_token AS VARCHAR)" in sql
    assert "task_id" in values
    assert "dispatch_token" in values
    assert _DELIVERY_TIMEOUT in values


@pytest.mark.parametrize(
    ("batch_size", "delivery_timeout", "message"),
    [
        (0, timedelta(seconds=1), "batch_size must be at least 1"),
        (-1, timedelta(seconds=1), "batch_size must be at least 1"),
        (1, timedelta(0), "delivery_timeout must be positive"),
        (1, timedelta(microseconds=-1), "delivery_timeout must be positive"),
        (
            1,
            timedelta(days=7, microseconds=1),
            "delivery_timeout must be at most 7 days",
        ),
        (1, timedelta.max, "delivery_timeout must be at most 7 days"),
    ],
)
@pytest.mark.asyncio
async def test_pending_delivery_repository_rejects_invalid_options_before_queries(
    batch_size: int,
    delivery_timeout: timedelta,
    message: str,
) -> None:
    """Invalid timing and batch size cannot acquire PostgreSQL locks."""

    scalars = AsyncMock()

    with pytest.raises(ValueError, match=f"^{message}$"):
        await OutboxRepository(_session(scalars)).recover_pending_deliveries(
            event_type=TASK_ROUTING_KEY,
            batch_size=batch_size,
            delivery_timeout=delivery_timeout,
        )

    scalars.assert_not_awaited()


@pytest.mark.parametrize("delivery_timeout", [timedelta(microseconds=1), timedelta(days=7)])
@pytest.mark.asyncio
async def test_pending_delivery_repository_accepts_safe_timeout_bounds(
    delivery_timeout: timedelta,
) -> None:
    """The exact upper bound is accepted without risking timestamp underflow."""

    scalars = AsyncMock(return_value=Mock(all=Mock(return_value=[])))

    assert (
        await OutboxRepository(_session(scalars)).recover_pending_deliveries(
            event_type=TASK_ROUTING_KEY,
            batch_size=1,
            delivery_timeout=delivery_timeout,
        )
        == 0
    )

    scalars.assert_awaited_once()
    assert delivery_timeout in _parameters(scalars.await_args_list[0].args[0]).values()


@pytest.mark.asyncio
async def test_pending_delivery_repository_skips_update_for_empty_batch() -> None:
    """An idle watchdog runs only the finite locked-candidate query."""

    scalars = AsyncMock(return_value=Mock(all=Mock(return_value=[])))
    session = _session(scalars)

    assert (
        await OutboxRepository(session).recover_pending_deliveries(
            event_type=TASK_ROUTING_KEY,
            batch_size=7,
            delivery_timeout=_DELIVERY_TIMEOUT,
        )
        == 0
    )

    scalars.assert_awaited_once()
    _require_current_pending_delivery_predicates(scalars.await_args_list[0].args[0])
    session.begin.assert_not_called()  # type: ignore[attr-defined]
    session.commit.assert_not_awaited()  # type: ignore[attr-defined]
    session.rollback.assert_not_awaited()  # type: ignore[attr-defined]


@pytest.mark.parametrize("updated_count", [0, 1, 2])
@pytest.mark.asyncio
async def test_pending_delivery_repository_locks_tasks_then_reopens_current_events(
    updated_count: int,
) -> None:
    """The replay preserves identity and attempts while reporting only changed rows."""

    event_ids = (UUID(int=1), UUID(int=2))
    scalars = AsyncMock(
        side_effect=[
            Mock(all=Mock(return_value=event_ids)),
            Mock(all=Mock(return_value=event_ids[:updated_count])),
        ],
    )
    session = _session(scalars)

    assert (
        await OutboxRepository(session).recover_pending_deliveries(
            event_type=TASK_ROUTING_KEY,
            batch_size=7,
            delivery_timeout=_DELIVERY_TIMEOUT,
        )
        == updated_count
    )

    assert scalars.await_count == 2
    candidates = scalars.await_args_list[0].args[0]
    update = scalars.await_args_list[1].args[0]
    candidate_sql = _sql(candidates)
    _require_current_pending_delivery_predicates(candidates)
    assert "ORDER BY outbox_events.published_at, outbox_events.id" in candidate_sql
    assert "LIMIT" in candidate_sql
    assert 7 in _parameters(candidates).values()
    assert "FOR UPDATE OF tasks SKIP LOCKED" in candidate_sql
    assert "FOR UPDATE OF outbox_events" not in candidate_sql

    update_sql = _sql(update)
    update_parameters = _parameters(update)
    _require_current_pending_delivery_predicates(update)
    assert update_sql.startswith("UPDATE outbox_events SET ")
    assert "FROM tasks" in update_sql
    assert "outbox_events.id IN" in update_sql
    assert "RETURNING outbox_events.id" in update_sql
    assert list(event_ids) in update_parameters.values() or event_ids in update_parameters.values()
    assert "available_at=clock_timestamp()" in update_sql
    assert update_parameters["published_at"] is None
    assert update_parameters["last_error"] == "PENDING_DELIVERY_TIMEOUT"
    for untouched in (
        "id",
        "task_id",
        "payload",
        "message_priority",
        "publish_attempts",
        "publisher_token",
        "lease_expires_at",
        "discarded_at",
        "dispatch_token",
    ):
        assert f"{untouched}=" not in update_sql
    session.begin.assert_not_called()  # type: ignore[attr-defined]
    session.commit.assert_not_awaited()  # type: ignore[attr-defined]
    session.rollback.assert_not_awaited()  # type: ignore[attr-defined]


@pytest.mark.parametrize("query_number", [1, 2])
@pytest.mark.asyncio
async def test_pending_delivery_repository_preserves_query_failures(query_number: int) -> None:
    """The caller-owned transaction receives either selection or update failure."""

    expected_error = RuntimeError("watchdog query failed")
    results: list[object] = [expected_error]
    if query_number == 2:
        results.insert(0, Mock(all=Mock(return_value=[UUID(int=1)])))
    scalars = AsyncMock(side_effect=results)
    session = _session(scalars)

    with pytest.raises(RuntimeError) as error_info:
        await OutboxRepository(session).recover_pending_deliveries(
            event_type=TASK_ROUTING_KEY,
            batch_size=7,
            delivery_timeout=_DELIVERY_TIMEOUT,
        )

    assert error_info.value is expected_error
    assert scalars.await_count == query_number
    session.rollback.assert_not_awaited()  # type: ignore[attr-defined]
