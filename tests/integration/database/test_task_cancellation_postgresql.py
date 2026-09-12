"""Task cancellation guarantees exercised against PostgreSQL."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
)

from cims_task_service.application.task_cancellation import cancel_task
from cims_task_service.application.task_creation import CreateTaskCommand, create_task
from cims_task_service.application.task_errors import (
    TaskNotCancellableError,
    TaskNotFoundError,
)
from cims_task_service.domain.task import TaskPriority, TaskStatus
from cims_task_service.infrastructure.database.models import OutboxEventModel, TaskModel
from cims_task_service.infrastructure.database.session import AsyncSessionFactory
from cims_task_service.infrastructure.database.task_repository import TaskRepository
from cims_task_service.infrastructure.messaging.topology import TASK_ROUTING_KEY

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_COMMAND = CreateTaskCommand(
    name="Daily report",
    description="Aggregate daily metrics",
    priority=TaskPriority.HIGH,
)


async def _create_task(session_factory: AsyncSessionFactory) -> TaskModel:
    result = await create_task(
        _COMMAND,
        session_factory=session_factory,
        max_attempts=3,
    )
    assert result.created is True
    return result.task


async def _read_task_and_events(
    session_factory: AsyncSessionFactory,
    task_id: UUID,
) -> tuple[TaskModel, list[OutboxEventModel]]:
    async with session_factory() as session:
        task = await session.get(TaskModel, task_id)
        events = list(
            (
                await session.scalars(
                    select(OutboxEventModel)
                    .where(OutboxEventModel.task_id == task_id)
                    .order_by(OutboxEventModel.id)
                )
            ).all()
        )
    assert task is not None
    return task, events


async def _prepare_task_status(
    session_factory: AsyncSessionFactory,
    task_id: UUID,
    status: TaskStatus,
) -> None:
    if status is TaskStatus.NEW:
        return

    async with session_factory.begin() as session:
        database_time = await session.scalar(select(func.clock_timestamp()))
        assert isinstance(database_time, datetime)
        statement = update(TaskModel).where(TaskModel.id == task_id)

        if status is TaskStatus.PENDING:
            statement = statement.values(status=status)
        elif status is TaskStatus.IN_PROGRESS:
            statement = statement.values(
                status=status,
                started_at=database_time,
                attempt_count=1,
                dispatch_token=None,
                execution_token=uuid4(),
                lease_expires_at=database_time + timedelta(minutes=1),
            )
        elif status is TaskStatus.COMPLETED:
            statement = statement.values(
                status=status,
                started_at=database_time,
                finished_at=database_time,
                result={"records_processed": 42},
                error=None,
                attempt_count=1,
                dispatch_token=None,
                execution_token=None,
                lease_expires_at=None,
            )
        elif status is TaskStatus.FAILED:
            statement = statement.values(
                status=status,
                started_at=database_time,
                finished_at=database_time,
                result=None,
                error={"type": "processing_failed"},
                attempt_count=1,
                dispatch_token=None,
                execution_token=None,
                lease_expires_at=None,
            )
        else:
            message = f"unsupported cancellation fixture status: {status.value}"
            raise ValueError(message)

        updated_id = await session.scalar(statement.returning(TaskModel.id))
        assert updated_id == task_id


def _outbox_event(
    event_id: UUID,
    task_id: UUID,
    *,
    created_at: datetime,
    event_type: str = TASK_ROUTING_KEY,
    published_at: datetime | None = None,
    discarded_at: datetime | None = None,
    publisher_token: UUID | None = None,
    lease_expires_at: datetime | None = None,
) -> OutboxEventModel:
    return OutboxEventModel(
        id=event_id,
        task_id=task_id,
        event_type=event_type,
        payload={"task_id": str(task_id)},
        message_priority=3,
        created_at=created_at,
        available_at=created_at,
        published_at=published_at,
        discarded_at=discarded_at,
        publish_attempts=0,
        publisher_token=publisher_token,
        lease_expires_at=lease_expires_at,
        last_error=None,
    )


def _task_snapshot(task: TaskModel) -> tuple[object, ...]:
    return (
        task.id,
        task.name,
        task.description,
        task.priority,
        task.status,
        task.created_at,
        task.idempotency_key_hash,
        task.request_fingerprint,
        task.started_at,
        task.finished_at,
        task.result,
        task.error,
        task.attempt_count,
        task.max_attempts,
        task.dispatch_token,
        task.execution_token,
        task.lease_expires_at,
    )


def _outbox_snapshot(event: OutboxEventModel) -> tuple[object, ...]:
    return (
        event.id,
        event.task_id,
        event.event_type,
        event.payload,
        event.message_priority,
        event.created_at,
        event.available_at,
        event.published_at,
        event.discarded_at,
        event.publish_attempts,
        event.publisher_token,
        event.lease_expires_at,
        event.last_error,
    )


async def _wait_until_blocked_by(
    observer: AsyncConnection,
    *,
    holder_pid: int,
    waiter_pid: int,
    waiter: asyncio.Task[TaskModel],
) -> None:
    async with asyncio.timeout(10):
        while True:
            if waiter.done():
                pytest.fail("Cancellation completed before PostgreSQL reported the row lock")
            blocked = await observer.scalar(
                text("SELECT :holder_pid = ANY(pg_blocking_pids(:waiter_pid))"),
                {"holder_pid": holder_pid, "waiter_pid": waiter_pid},
            )
            if blocked:
                return


@asynccontextmanager
async def _blocked_cancellation(
    waiting_connection: AsyncConnection,
    observer: AsyncConnection,
    *,
    holder_pid: int,
    task_id: UUID,
) -> AsyncIterator[asyncio.Task[TaskModel]]:
    """Start cancellation and yield only after it waits for the holder's task row."""

    waiter_pid = await waiting_connection.scalar(text("SELECT pg_backend_pid()"))
    observer_pid = await observer.scalar(text("SELECT pg_backend_pid()"))
    assert isinstance(waiter_pid, int)
    assert isinstance(observer_pid, int)
    assert len({holder_pid, waiter_pid, observer_pid}) == 3
    await waiting_connection.commit()
    waiting_factory: AsyncSessionFactory = async_sessionmaker[AsyncSession](
        bind=waiting_connection,
        autoflush=False,
        expire_on_commit=False,
    )
    waiter = asyncio.create_task(cancel_task(task_id, session_factory=waiting_factory))
    try:
        await _wait_until_blocked_by(
            observer,
            holder_pid=holder_pid,
            waiter_pid=waiter_pid,
            waiter=waiter,
        )
        yield waiter
    finally:
        if not waiter.done():
            waiter.cancel()
        async with asyncio.timeout(10):
            await asyncio.gather(waiter, return_exceptions=True)


@pytest.mark.parametrize(
    "initial_status",
    [TaskStatus.NEW, TaskStatus.PENDING, TaskStatus.IN_PROGRESS],
)
async def test_cancel_active_task_persists_valid_terminal_state_and_discards_outbox(
    postgres_session_factory: AsyncSessionFactory,
    initial_status: TaskStatus,
) -> None:
    created = await _create_task(postgres_session_factory)
    await _prepare_task_status(postgres_session_factory, created.id, initial_status)
    before, before_events = await _read_task_and_events(
        postgres_session_factory,
        created.id,
    )

    cancelled = await cancel_task(
        created.id,
        session_factory=postgres_session_factory,
    )

    stored, stored_events = await _read_task_and_events(
        postgres_session_factory,
        created.id,
    )
    assert before.status is initial_status
    assert len(before_events) == len(stored_events) == 1
    assert cancelled.status is stored.status is TaskStatus.CANCELLED
    assert cancelled.finished_at == stored.finished_at
    assert stored.finished_at is not None
    assert stored.finished_at.tzinfo is not None
    assert stored.finished_at >= stored.created_at
    if stored.started_at is not None:
        assert stored.finished_at >= stored.started_at
    assert stored.started_at == before.started_at
    assert stored.attempt_count == before.attempt_count
    assert stored.result is stored.error is None
    assert stored.dispatch_token is stored.execution_token is None
    assert stored.lease_expires_at is None

    event = stored_events[0]
    assert event.event_type == TASK_ROUTING_KEY
    assert event.published_at is None
    assert event.discarded_at is not None
    assert event.publisher_token is None
    assert event.lease_expires_at is None


async def test_concurrent_cancellation_is_an_idempotent_serialized_replay(
    postgres_engine: AsyncEngine,
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    created = await _create_task(postgres_session_factory)

    async with (
        postgres_session_factory() as holder,
        postgres_engine.connect() as waiting_connection,
        postgres_engine.connect() as observer,
        holder.begin() as transaction,
    ):
        holder_pid = await holder.scalar(text("SELECT pg_backend_pid()"))
        assert isinstance(holder_pid, int)
        first = await TaskRepository(holder).cancel_with_outbox(
            created.id,
            event_type=TASK_ROUTING_KEY,
        )
        assert first is not None
        assert first.changed is True
        assert first.task.finished_at is not None
        first_finished_at = first.task.finished_at
        first_discarded_at = await holder.scalar(
            select(OutboxEventModel.discarded_at).where(OutboxEventModel.task_id == created.id)
        )
        assert first_discarded_at is not None

        async with _blocked_cancellation(
            waiting_connection,
            observer,
            holder_pid=holder_pid,
            task_id=created.id,
        ) as waiter:
            await transaction.commit()
            async with asyncio.timeout(10):
                repeated = await waiter

    stored, events = await _read_task_and_events(postgres_session_factory, created.id)
    assert repeated.status is stored.status is TaskStatus.CANCELLED
    assert repeated.finished_at == stored.finished_at == first_finished_at
    assert len(events) == 1
    assert events[0].discarded_at == first_discarded_at


async def test_waiting_cancellation_rechecks_active_status_and_uses_late_clock_time(
    postgres_engine: AsyncEngine,
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    created = await _create_task(postgres_session_factory)
    await _prepare_task_status(
        postgres_session_factory,
        created.id,
        TaskStatus.PENDING,
    )

    async with (
        postgres_session_factory() as holder,
        postgres_engine.connect() as waiting_connection,
        postgres_engine.connect() as observer,
        holder.begin() as transaction,
    ):
        holder_pid = await holder.scalar(text("SELECT pg_backend_pid()"))
        assert isinstance(holder_pid, int)
        locked_id = await holder.scalar(
            select(TaskModel.id).where(TaskModel.id == created.id).with_for_update()
        )
        assert locked_id == created.id

        async with _blocked_cancellation(
            waiting_connection,
            observer,
            holder_pid=holder_pid,
            task_id=created.id,
        ) as waiter:
            started_at = await holder.scalar(select(func.clock_timestamp()))
            assert isinstance(started_at, datetime)
            execution_token = uuid4()
            claimed_id = await holder.scalar(
                update(TaskModel)
                .where(TaskModel.id == created.id, TaskModel.status == TaskStatus.PENDING)
                .values(
                    status=TaskStatus.IN_PROGRESS,
                    started_at=started_at,
                    attempt_count=1,
                    dispatch_token=None,
                    execution_token=execution_token,
                    lease_expires_at=started_at + timedelta(minutes=1),
                )
                .returning(TaskModel.id)
            )
            assert claimed_id == created.id
            await transaction.commit()
            async with asyncio.timeout(10):
                cancelled = await waiter

    stored, events = await _read_task_and_events(postgres_session_factory, created.id)
    assert cancelled.status is stored.status is TaskStatus.CANCELLED
    assert cancelled.started_at == stored.started_at == started_at
    assert cancelled.finished_at == stored.finished_at
    assert stored.finished_at is not None
    assert stored.finished_at >= started_at
    assert stored.attempt_count == 1
    assert stored.dispatch_token is stored.execution_token is None
    assert stored.lease_expires_at is None
    assert len(events) == 1
    assert events[0].discarded_at is not None


async def test_completed_task_wins_race_against_waiting_cancellation(
    postgres_engine: AsyncEngine,
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    created = await _create_task(postgres_session_factory)
    await _prepare_task_status(
        postgres_session_factory,
        created.id,
        TaskStatus.IN_PROGRESS,
    )

    async with (
        postgres_session_factory() as holder,
        postgres_engine.connect() as waiting_connection,
        postgres_engine.connect() as observer,
        holder.begin() as transaction,
    ):
        holder_pid = await holder.scalar(text("SELECT pg_backend_pid()"))
        assert isinstance(holder_pid, int)
        completed_at = await holder.scalar(select(func.clock_timestamp()))
        assert isinstance(completed_at, datetime)
        completed_id = await holder.scalar(
            update(TaskModel)
            .where(TaskModel.id == created.id, TaskModel.status == TaskStatus.IN_PROGRESS)
            .values(
                status=TaskStatus.COMPLETED,
                finished_at=completed_at,
                result={"records_processed": 42},
                error=None,
                dispatch_token=None,
                execution_token=None,
                lease_expires_at=None,
            )
            .returning(TaskModel.id)
        )
        assert completed_id == created.id

        async with _blocked_cancellation(
            waiting_connection,
            observer,
            holder_pid=holder_pid,
            task_id=created.id,
        ) as waiter:
            await transaction.commit()
            with pytest.raises(TaskNotCancellableError) as error_info:
                async with asyncio.timeout(10):
                    await waiter

    stored, events = await _read_task_and_events(postgres_session_factory, created.id)
    assert error_info.value.task_id == created.id
    assert error_info.value.current_status is TaskStatus.COMPLETED
    assert stored.status is TaskStatus.COMPLETED
    assert stored.finished_at == completed_at
    assert stored.result == {"records_processed": 42}
    assert stored.error is None
    assert len(events) == 1
    assert events[0].published_at is events[0].discarded_at is None


async def test_repeated_cancellation_preserves_finished_and_discard_timestamps(
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    created = await _create_task(postgres_session_factory)
    await cancel_task(created.id, session_factory=postgres_session_factory)
    after_first, first_events = await _read_task_and_events(
        postgres_session_factory,
        created.id,
    )

    repeated = await cancel_task(created.id, session_factory=postgres_session_factory)

    after_second, second_events = await _read_task_and_events(
        postgres_session_factory,
        created.id,
    )
    assert after_first.finished_at is not None
    assert repeated.status is after_second.status is TaskStatus.CANCELLED
    assert repeated.finished_at == after_first.finished_at == after_second.finished_at
    assert len(first_events) == len(second_events) == 1
    assert first_events[0].discarded_at is not None
    assert second_events[0].discarded_at == first_events[0].discarded_at


async def test_cancellation_discards_only_unpublished_execution_events(
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    created = await _create_task(postgres_session_factory)
    original_task, original_events = await _read_task_and_events(
        postgres_session_factory,
        created.id,
    )
    assert len(original_events) == 1
    original_event_id = original_events[0].id

    leased_id = UUID("10000000-0000-4000-8000-000000000001")
    published_id = UUID("20000000-0000-4000-8000-000000000002")
    discarded_id = UUID("30000000-0000-4000-8000-000000000003")
    unrelated_id = UUID("40000000-0000-4000-8000-000000000004")
    async with postgres_session_factory.begin() as session:
        database_time = await session.scalar(select(func.clock_timestamp()))
        assert isinstance(database_time, datetime)
        existing_discarded_at = database_time
        published_at = database_time
        session.add_all(
            [
                _outbox_event(
                    leased_id,
                    created.id,
                    created_at=database_time,
                    publisher_token=uuid4(),
                    lease_expires_at=database_time + timedelta(minutes=1),
                ),
                _outbox_event(
                    published_id,
                    created.id,
                    created_at=database_time,
                    published_at=published_at,
                ),
                _outbox_event(
                    discarded_id,
                    created.id,
                    created_at=database_time,
                    discarded_at=existing_discarded_at,
                ),
                _outbox_event(
                    unrelated_id,
                    created.id,
                    created_at=database_time,
                    event_type="task.audit.v1",
                ),
            ]
        )

    await cancel_task(created.id, session_factory=postgres_session_factory)

    stored_task, stored_events = await _read_task_and_events(
        postgres_session_factory,
        created.id,
    )
    events_by_id = {event.id: event for event in stored_events}
    assert stored_task.status is TaskStatus.CANCELLED
    assert original_task.status is TaskStatus.NEW
    assert len(events_by_id) == 5

    for event_id in (original_event_id, leased_id):
        event = events_by_id[event_id]
        assert event.discarded_at is not None
        assert event.published_at is None
        assert event.publisher_token is None
        assert event.lease_expires_at is None

    published_event = events_by_id[published_id]
    assert published_event.published_at == published_at
    assert published_event.discarded_at is None

    discarded_event = events_by_id[discarded_id]
    assert discarded_event.discarded_at == existing_discarded_at
    assert discarded_event.published_at is None

    unrelated_event = events_by_id[unrelated_id]
    assert unrelated_event.event_type == "task.audit.v1"
    assert unrelated_event.published_at is unrelated_event.discarded_at is None


@pytest.mark.parametrize("terminal_status", [TaskStatus.COMPLETED, TaskStatus.FAILED])
async def test_terminal_task_rejects_cancellation_without_changing_task_or_outbox(
    postgres_session_factory: AsyncSessionFactory,
    terminal_status: TaskStatus,
) -> None:
    created = await _create_task(postgres_session_factory)
    await _prepare_task_status(postgres_session_factory, created.id, terminal_status)
    before, before_events = await _read_task_and_events(
        postgres_session_factory,
        created.id,
    )
    before_task_snapshot = _task_snapshot(before)
    before_event_snapshots = tuple(_outbox_snapshot(event) for event in before_events)

    with pytest.raises(TaskNotCancellableError) as error_info:
        await cancel_task(created.id, session_factory=postgres_session_factory)

    after, after_events = await _read_task_and_events(
        postgres_session_factory,
        created.id,
    )
    assert error_info.value.task_id == created.id
    assert error_info.value.current_status is terminal_status
    assert after.status is terminal_status
    assert _task_snapshot(after) == before_task_snapshot
    assert tuple(_outbox_snapshot(event) for event in after_events) == before_event_snapshots


async def test_unknown_task_cancellation_raises_not_found(
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    task_id = UUID("ba21a692-6c11-47ac-bc71-392f27f03416")

    with pytest.raises(TaskNotFoundError) as error_info:
        await cancel_task(task_id, session_factory=postgres_session_factory)

    assert error_info.value.task_id == task_id


async def test_outbox_failure_rolls_back_task_cancellation(
    postgres_engine: AsyncEngine,
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    created = await _create_task(postgres_session_factory)
    async with postgres_engine.begin() as connection:
        await connection.execute(
            text(
                "ALTER TABLE outbox_events "
                "ADD CONSTRAINT ck_test_reject_discard CHECK (discarded_at IS NULL)"
            )
        )

    try:
        with pytest.raises(IntegrityError) as error_info:
            await cancel_task(created.id, session_factory=postgres_session_factory)
        assert "ck_test_reject_discard" in str(error_info.value.orig)

        task, events = await _read_task_and_events(postgres_session_factory, created.id)
        assert task.status is TaskStatus.NEW
        assert task.finished_at is None
        assert task.dispatch_token == created.dispatch_token
        assert task.dispatch_token is not None
        assert len(events) == 1
        assert events[0].published_at is events[0].discarded_at is None
    finally:
        async with postgres_engine.begin() as connection:
            await connection.execute(
                text("ALTER TABLE outbox_events DROP CONSTRAINT ck_test_reject_discard")
            )

    recovered = await cancel_task(created.id, session_factory=postgres_session_factory)
    assert recovered.status is TaskStatus.CANCELLED
