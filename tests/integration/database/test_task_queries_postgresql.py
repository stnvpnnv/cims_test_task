"""Task query behavior exercised against independent PostgreSQL sessions."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, override
from uuid import UUID

import pytest
from sqlalchemy.engine import ScalarResult
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.sql import Executable, Select

from cims_task_service.application.task_creation import CreateTaskCommand, create_task
from cims_task_service.application.task_errors import TaskNotFoundError
from cims_task_service.application.task_queries import (
    ListTasksQuery,
    ListTasksResult,
    get_task,
    get_task_status,
    list_tasks,
)
from cims_task_service.domain.task import TaskPriority, TaskStatus
from cims_task_service.infrastructure.database.models import TaskModel
from cims_task_service.infrastructure.database.session import AsyncSessionFactory
from cims_task_service.infrastructure.database.task_repository import TaskStatusSnapshot

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_INTERLEAVE_PROBE_KEY = "task_list_interleave_probe"


def _task_row(
    task_id: UUID,
    *,
    priority: TaskPriority,
    status: Literal[TaskStatus.NEW, TaskStatus.FAILED],
    created_at: datetime,
) -> TaskModel:
    """Build a deterministic NEW or FAILED row satisfying table invariants."""

    failed = status is TaskStatus.FAILED
    return TaskModel(
        id=task_id,
        name=f"List fixture {task_id.hex[-4:]}",
        description="Persisted directly for a PostgreSQL list-query test",
        priority=priority,
        status=status,
        created_at=created_at,
        idempotency_key_hash=None,
        request_fingerprint=None,
        started_at=created_at + timedelta(seconds=1) if failed else None,
        finished_at=created_at + timedelta(seconds=2) if failed else None,
        result=None,
        error={"code": "PROCESSING_FAILED", "retryable": False} if failed else None,
        attempt_count=1 if failed else 0,
        max_attempts=3,
        dispatch_token=None if failed else UUID(int=task_id.int ^ 1),
        execution_token=None,
        lease_expires_at=None,
    )


async def _persist_tasks(
    session_factory: AsyncSessionFactory,
    *tasks: TaskModel,
) -> None:
    """Commit deterministic task rows before exercising a query."""

    async with session_factory.begin() as session:
        session.add_all(tasks)


def _task_ids(result: ListTasksResult) -> tuple[UUID, ...]:
    return tuple(task.id for task in result.items)


@dataclass(slots=True)
class _InterleaveProbe:
    """Coordinate and record the write inserted between the two list queries."""

    after_count: Callable[[], Awaitable[None]]
    scalar_calls: int = 0
    committed_inserts: int = 0


class _InsertAfterCountSession(AsyncSession):
    """Commit one matching row after the real COUNT query has completed."""

    @override
    async def scalars(
        self,
        statement: Executable,
        *args: Any,
        **kwargs: Any,
    ) -> ScalarResult[Any]:
        result = await super().scalars(statement, *args, **kwargs)
        probe = self.info.get(_INTERLEAVE_PROBE_KEY)
        if not isinstance(probe, _InterleaveProbe):
            message = "interleaving session requires a task-list probe"
            raise RuntimeError(message)

        probe.scalar_calls += 1
        if probe.scalar_calls == 1:
            if not isinstance(statement, Select):
                message = "the first task-list statement must be a SELECT"
                raise AssertionError(message)
            selected_columns = tuple(statement.selected_columns)
            if len(selected_columns) != 1 or selected_columns[0].name != "count":
                message = "the first task-list SELECT must compute COUNT(*)"
                raise AssertionError(message)
            await probe.after_count()
            probe.committed_inserts += 1

        return result


async def test_get_task_hydrates_complete_state_in_a_new_session(
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    """A committed task can be read as a fresh ORM instance with its full state."""

    command = CreateTaskCommand(
        name="Inventory snapshot",
        description="Aggregate the current warehouse quantities",
        priority=TaskPriority.HIGH,
    )
    created = await create_task(
        command,
        session_factory=postgres_session_factory,
        max_attempts=4,
    )

    task = await get_task(
        created.task.id,
        session_factory=postgres_session_factory,
    )

    assert created.created is True
    assert task is not created.task
    assert task.id == created.task.id
    assert task.id.version == 4
    assert task.name == command.name
    assert task.description == command.description
    assert task.priority is TaskPriority.HIGH
    assert task.status is TaskStatus.NEW
    assert task.created_at == created.task.created_at
    assert task.created_at.tzinfo is not None
    assert task.created_at.utcoffset() == timedelta(0)
    assert task.idempotency_key_hash is task.request_fingerprint is None
    assert task.started_at is task.finished_at is None
    assert task.result is task.error is None
    assert task.attempt_count == 0
    assert task.max_attempts == 4
    assert task.dispatch_token == created.task.dispatch_token
    assert task.dispatch_token is not None
    assert task.dispatch_token.version == 4
    assert task.execution_token is None
    assert task.lease_expires_at is None


async def test_get_task_raises_for_an_unknown_identifier(
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    """A primary key absent from PostgreSQL remains attached to the query error."""

    task_id = UUID("ba21a692-6c11-47ac-bc71-392f27f03416")

    with pytest.raises(TaskNotFoundError) as error_info:
        await get_task(
            task_id,
            session_factory=postgres_session_factory,
        )

    assert error_info.value.task_id == task_id


async def test_get_task_status_returns_the_persisted_projection(
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    """A committed task status is read as the exact lightweight snapshot."""

    created = await create_task(
        CreateTaskCommand(
            name="Refresh materialized view",
            description="Rebuild the reporting projection",
            priority=TaskPriority.MEDIUM,
        ),
        session_factory=postgres_session_factory,
        max_attempts=3,
    )

    snapshot = await get_task_status(
        created.task.id,
        session_factory=postgres_session_factory,
    )

    assert created.created is True
    assert snapshot == TaskStatusSnapshot(id=created.task.id, status=TaskStatus.NEW)
    assert snapshot.status is TaskStatus.NEW


async def test_get_task_status_raises_for_an_unknown_identifier(
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    """A missing status projection reports the identifier requested by the caller."""

    task_id = UUID("0a6757ba-0dc8-4d87-b31c-cbfb78d308e4")

    with pytest.raises(TaskNotFoundError) as error_info:
        await get_task_status(
            task_id,
            session_factory=postgres_session_factory,
        )

    assert error_info.value.task_id == task_id


async def test_list_tasks_applies_status_priority_and_combined_filters(
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    """PostgreSQL applies each optional filter and combines both with AND."""

    created_at = datetime(2026, 9, 10, 3, 0, tzinfo=UTC)
    new_high = _task_row(
        UUID("10000000-0000-4000-8000-000000000001"),
        priority=TaskPriority.HIGH,
        status=TaskStatus.NEW,
        created_at=created_at + timedelta(seconds=2),
    )
    new_low = _task_row(
        UUID("20000000-0000-4000-8000-000000000002"),
        priority=TaskPriority.LOW,
        status=TaskStatus.NEW,
        created_at=created_at + timedelta(seconds=1),
    )
    failed_high = _task_row(
        UUID("30000000-0000-4000-8000-000000000003"),
        priority=TaskPriority.HIGH,
        status=TaskStatus.FAILED,
        created_at=created_at,
    )
    await _persist_tasks(postgres_session_factory, new_high, new_low, failed_high)

    status_page = await list_tasks(
        ListTasksQuery(status=TaskStatus.NEW, size=10),
        session_factory=postgres_session_factory,
    )
    priority_page = await list_tasks(
        ListTasksQuery(priority=TaskPriority.HIGH, size=10),
        session_factory=postgres_session_factory,
    )
    combined_page = await list_tasks(
        ListTasksQuery(
            status=TaskStatus.NEW,
            priority=TaskPriority.HIGH,
            size=10,
        ),
        session_factory=postgres_session_factory,
    )

    assert (_task_ids(status_page), status_page.total) == (
        (new_high.id, new_low.id),
        2,
    )
    assert (_task_ids(priority_page), priority_page.total) == (
        (new_high.id, failed_high.id),
        2,
    )
    assert (_task_ids(combined_page), combined_page.total) == ((new_high.id,), 1)


async def test_list_tasks_paginates_equal_timestamps_by_descending_identifier(
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    """The UUID tie-breaker keeps adjacent offset pages deterministic and disjoint."""

    created_at = datetime(2026, 9, 10, 4, 0, tzinfo=UTC)
    task_ids = tuple(
        UUID(value)
        for value in (
            "10000000-0000-4000-8000-000000000001",
            "20000000-0000-4000-8000-000000000002",
            "30000000-0000-4000-8000-000000000003",
            "40000000-0000-4000-8000-000000000004",
            "50000000-0000-4000-8000-000000000005",
        )
    )
    tasks = tuple(
        _task_row(
            task_id,
            priority=TaskPriority.MEDIUM,
            status=TaskStatus.NEW,
            created_at=created_at,
        )
        for task_id in task_ids
    )
    await _persist_tasks(postgres_session_factory, *tasks)

    first_page = await list_tasks(
        ListTasksQuery(page=1, size=2),
        session_factory=postgres_session_factory,
    )
    second_page = await list_tasks(
        ListTasksQuery(page=2, size=2),
        session_factory=postgres_session_factory,
    )
    out_of_range = await list_tasks(
        ListTasksQuery(page=4, size=2),
        session_factory=postgres_session_factory,
    )

    first_ids = _task_ids(first_page)
    second_ids = _task_ids(second_page)
    assert (first_ids, first_page.total) == ((task_ids[4], task_ids[3]), 5)
    assert (second_ids, second_page.total) == ((task_ids[2], task_ids[1]), 5)
    assert set(first_ids).isdisjoint(second_ids)
    assert (_task_ids(out_of_range), out_of_range.total) == ((), 5)


async def test_list_tasks_reads_count_and_page_from_one_repeatable_read_snapshot(
    postgres_engine: AsyncEngine,
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    """A matching commit between COUNT and SELECT is excluded from both snapshot reads."""

    original_task = _task_row(
        UUID("61000000-0000-4000-8000-000000000006"),
        priority=TaskPriority.HIGH,
        status=TaskStatus.NEW,
        created_at=datetime(2026, 9, 10, 5, 0, tzinfo=UTC),
    )
    late_task = _task_row(
        UUID("62000000-0000-4000-8000-000000000007"),
        priority=TaskPriority.HIGH,
        status=TaskStatus.NEW,
        created_at=datetime(2026, 9, 10, 5, 1, tzinfo=UTC),
    )
    await _persist_tasks(postgres_session_factory, original_task)

    async def insert_late_task() -> None:
        async with postgres_session_factory.begin() as writer:
            writer.add(late_task)

    probe = _InterleaveProbe(after_count=insert_late_task)
    interleaved_session_factory: AsyncSessionFactory = async_sessionmaker[AsyncSession](
        bind=postgres_engine,
        class_=_InsertAfterCountSession,
        autoflush=False,
        expire_on_commit=False,
        info={_INTERLEAVE_PROBE_KEY: probe},
    )

    query = ListTasksQuery(
        status=TaskStatus.NEW,
        priority=TaskPriority.HIGH,
        size=20,
    )
    result = await list_tasks(
        query,
        session_factory=interleaved_session_factory,
    )

    assert probe.scalar_calls == 2
    assert probe.committed_inserts == 1
    assert (result.total, _task_ids(result)) == (1, (original_task.id,))

    visible_after_snapshot = await list_tasks(
        query,
        session_factory=postgres_session_factory,
    )
    assert (visible_after_snapshot.total, _task_ids(visible_after_snapshot)) == (
        2,
        (late_task.id, original_task.id),
    )
