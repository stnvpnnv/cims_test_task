"""Pending-delivery watchdog invariants exercised against PostgreSQL."""

import asyncio
from datetime import datetime, timedelta
from typing import Literal
from uuid import UUID

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from cims_task_service.application.pending_delivery_recovery import PendingTaskDeliveryRecovery
from cims_task_service.domain.task import TaskPriority, TaskStatus
from cims_task_service.infrastructure.database.models import OutboxEventModel, TaskModel
from cims_task_service.infrastructure.database.outbox_repository import OutboxRepository
from cims_task_service.infrastructure.database.session import AsyncSessionFactory
from cims_task_service.infrastructure.database.task_execution_repository import (
    TaskExecutionRepository,
)
from cims_task_service.infrastructure.database.task_repository import TaskRepository
from cims_task_service.infrastructure.messaging.topology import TASK_ROUTING_KEY

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_DELIVERY_TIMEOUT = timedelta(minutes=5)
_PUBLISH_LEASE = timedelta(minutes=1)
_EXECUTION_LEASE = timedelta(minutes=1)
_FAILURE_SUMMARY = "PENDING_DELIVERY_TIMEOUT"

type _Rows = dict[UUID, dict[str, object]]


async def _database_now(session_factory: AsyncSessionFactory) -> datetime:
    async with session_factory() as session:
        current_time = await session.scalar(select(func.clock_timestamp()))
    assert isinstance(current_time, datetime)
    return current_time


def _delivery_pair(
    number: int,
    *,
    now: datetime,
    published_at: datetime | None = None,
    status: TaskStatus = TaskStatus.PENDING,
) -> tuple[TaskModel, OutboxEventModel]:
    created_at = now - timedelta(hours=1)
    task_id = UUID(int=number)
    dispatch_token = UUID(int=number + 10_000)
    task = TaskModel(
        id=task_id,
        name=f"Pending delivery {number}",
        description="The broker may have lost this confirmed delivery",
        priority=TaskPriority.HIGH,
        status=status,
        created_at=created_at,
        attempt_count=0,
        max_attempts=3,
        dispatch_token=dispatch_token,
    )
    if status is TaskStatus.IN_PROGRESS:
        task.attempt_count = 1
        task.started_at = created_at + timedelta(minutes=1)
        task.dispatch_token = None
        task.execution_token = UUID(int=number + 20_000)
        task.lease_expires_at = now + _EXECUTION_LEASE
    elif status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
        task.dispatch_token = None
        task.finished_at = now - timedelta(minutes=1)
        if status is not TaskStatus.CANCELLED:
            task.attempt_count = 1
            task.started_at = created_at + timedelta(minutes=1)
    event = OutboxEventModel(
        id=UUID(int=number + 30_000),
        task_id=task_id,
        event_type=TASK_ROUTING_KEY,
        payload={"task_id": str(task_id), "dispatch_token": str(dispatch_token)},
        message_priority=3,
        created_at=created_at,
        available_at=created_at,
        published_at=published_at,
        publish_attempts=2,
    )
    return task, event


async def _seed_pairs(
    session_factory: AsyncSessionFactory,
    *pairs: tuple[TaskModel, OutboxEventModel],
) -> None:
    async with session_factory.begin() as session:
        # Insert parents explicitly before children; the models have no relationship.
        session.add_all(task for task, _event in pairs)
        await session.flush()
        session.add_all(event for _task, event in pairs)


def _snapshot(row: TaskModel | OutboxEventModel) -> dict[str, object]:
    return {column.key: getattr(row, column.key) for column in row.__mapper__.columns}


async def _read_rows(session_factory: AsyncSessionFactory) -> tuple[_Rows, _Rows]:
    async with session_factory() as session:
        tasks = {task.id: _snapshot(task) for task in await session.scalars(select(TaskModel))}
        events = {
            event.id: _snapshot(event) for event in await session.scalars(select(OutboxEventModel))
        }
    return tasks, events


def _watchdog(
    session_factory: AsyncSessionFactory,
    *,
    batch_size: int = 10,
) -> PendingTaskDeliveryRecovery:
    return PendingTaskDeliveryRecovery(
        session_factory,
        batch_size=batch_size,
        delivery_timeout=_DELIVERY_TIMEOUT,
    )


async def _recover_in_transaction(session: AsyncSession, *, batch_size: int = 10) -> int:
    return await OutboxRepository(session).recover_pending_deliveries(
        event_type=TASK_ROUTING_KEY,
        batch_size=batch_size,
        delivery_timeout=_DELIVERY_TIMEOUT,
    )


async def test_watchdog_reopens_confirmation_without_changing_task_or_message_identity(
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    now = await _database_now(postgres_session_factory)
    task, event = _delivery_pair(1, now=now, published_at=now - timedelta(minutes=10))
    event.last_error = "obsolete publication error"
    await _seed_pairs(postgres_session_factory, (task, event))
    before_tasks, before_events = await _read_rows(postgres_session_factory)
    before_recovery = await _database_now(postgres_session_factory)

    assert await _watchdog(postgres_session_factory).recover_once() == 1

    after_recovery = await _database_now(postgres_session_factory)
    after_tasks, after_events = await _read_rows(postgres_session_factory)
    assert after_tasks == before_tasks
    available_at = after_events[event.id]["available_at"]
    assert isinstance(available_at, datetime)
    assert before_recovery <= available_at <= after_recovery
    assert after_events == {
        event.id: {
            **before_events[event.id],
            "published_at": None,
            "available_at": available_at,
            "last_error": _FAILURE_SUMMARY,
        }
    }
    # An already reopened event cannot be reopened repeatedly before publication.
    assert await _watchdog(postgres_session_factory).recover_once() == 0


async def test_watchdog_excludes_unsafe_or_unrelated_deliveries(
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    now = await _database_now(postgres_session_factory)
    aged = now - timedelta(minutes=10)
    pairs = [_delivery_pair(number, now=now, published_at=aged) for number in range(1, 14)]
    pairs[1][1].published_at = now
    pairs[2][1].published_at = None
    pairs[3][1].published_at = None
    pairs[3][1].discarded_at = now
    pairs[4][1].available_at = now + timedelta(hours=1)
    pairs[5][1].event_type = "task.unrelated.v1"
    pairs[6][1].payload["dispatch_token"] = str(UUID(int=999))
    pairs[7][1].payload["task_id"] = str(pairs[0][0].id)
    del pairs[8][1].payload["dispatch_token"]
    del pairs[9][1].payload["task_id"]
    pairs[10][1].payload["dispatch_token"] = ["not a UUID"]
    pairs[11][1].payload["task_id"] = {"not": "a task identifier"}
    pairs[12][1].published_at = None
    pairs[12][1].publisher_token = UUID(int=999)
    pairs[12][1].lease_expires_at = now + _PUBLISH_LEASE
    for number, status in enumerate(
        (
            TaskStatus.NEW,
            TaskStatus.IN_PROGRESS,
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        ),
        start=14,
    ):
        pairs.append(_delivery_pair(number, now=now, published_at=aged, status=status))
    await _seed_pairs(postgres_session_factory, *pairs)
    before_tasks, before_events = await _read_rows(postgres_session_factory)

    assert await _watchdog(postgres_session_factory, batch_size=100).recover_once() == 1

    after_tasks, after_events = await _read_rows(postgres_session_factory)
    assert after_tasks == before_tasks
    recovered_id = pairs[0][1].id
    assert after_events[recovered_id]["published_at"] is None
    assert {
        event_id: values for event_id, values in after_events.items() if event_id != recovered_id
    } == {
        event_id: values for event_id, values in before_events.items() if event_id != recovered_id
    }


async def test_maximum_delivery_timeout_is_valid_for_real_postgresql_timestamps(
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    now = await _database_now(postgres_session_factory)
    task, event = _delivery_pair(1, now=now, published_at=now - timedelta(days=8))
    task.created_at = now - timedelta(days=9)
    event.created_at = task.created_at
    event.available_at = task.created_at
    fresh_pair = _delivery_pair(2, now=now, published_at=now)
    await _seed_pairs(postgres_session_factory, (task, event), fresh_pair)
    before_tasks, before_events = await _read_rows(postgres_session_factory)
    watchdog = PendingTaskDeliveryRecovery(
        postgres_session_factory,
        batch_size=10,
        delivery_timeout=timedelta(days=7),
    )

    assert await watchdog.recover_once() == 1

    after_tasks, after_events = await _read_rows(postgres_session_factory)
    assert after_tasks == before_tasks
    assert set(after_events) == set(before_events)
    assert after_events[event.id]["published_at"] is None
    assert after_events[event.id]["payload"] == before_events[event.id]["payload"]
    assert after_events[event.id]["publish_attempts"] == before_events[event.id]["publish_attempts"]
    assert after_events[fresh_pair[1].id] == before_events[fresh_pair[1].id]
    assert await watchdog.recover_once() == 0


async def test_watchdog_orders_and_bounds_each_batch_by_confirmation_age_and_identifier(
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    now = await _database_now(postgres_session_factory)
    pairs = (
        _delivery_pair(1, now=now, published_at=now - timedelta(minutes=10)),
        _delivery_pair(2, now=now, published_at=now - timedelta(minutes=20)),
        _delivery_pair(3, now=now, published_at=now - timedelta(minutes=20)),
    )
    await _seed_pairs(postgres_session_factory, *pairs)
    watchdog = _watchdog(postgres_session_factory, batch_size=1)

    for expected_number in (2, 3, 1):
        _before_tasks, before_events = await _read_rows(postgres_session_factory)
        assert await watchdog.recover_once() == 1
        _after_tasks, after_events = await _read_rows(postgres_session_factory)
        changed_ids = {
            event_id
            for event_id, values in after_events.items()
            if values != before_events[event_id]
        }
        assert changed_ids == {UUID(int=expected_number + 30_000)}
    assert await watchdog.recover_once() == 0


async def test_rolling_back_watchdog_restores_the_confirmed_event(
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    now = await _database_now(postgres_session_factory)
    pair = _delivery_pair(1, now=now, published_at=now - timedelta(minutes=10))
    await _seed_pairs(postgres_session_factory, pair)
    original = await _read_rows(postgres_session_factory)

    async with postgres_session_factory() as session, session.begin() as transaction:
        assert await _recover_in_transaction(session) == 1
        assert (
            await session.scalar(
                select(OutboxEventModel.published_at).where(OutboxEventModel.id == pair[1].id)
            )
            is None
        )
        await transaction.rollback()

    assert await _read_rows(postgres_session_factory) == original
    assert await _watchdog(postgres_session_factory).recover_once() == 1


async def test_update_failure_rolls_back_the_watchdog_batch(
    postgres_engine: AsyncEngine,
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    now = await _database_now(postgres_session_factory)
    await _seed_pairs(
        postgres_session_factory,
        *(
            _delivery_pair(number, now=now, published_at=now - timedelta(minutes=10))
            for number in (1, 2)
        ),
    )
    original = await _read_rows(postgres_session_factory)
    async with postgres_engine.begin() as connection:
        await connection.execute(
            text(
                "ALTER TABLE outbox_events ADD CONSTRAINT "
                "ck_test_keep_confirmation CHECK (published_at IS NOT NULL)"
            )
        )
    try:
        with pytest.raises(IntegrityError, match="ck_test_keep_confirmation"):
            await _watchdog(postgres_session_factory).recover_once()
    finally:
        async with postgres_engine.begin() as connection:
            await connection.execute(
                text("ALTER TABLE outbox_events DROP CONSTRAINT ck_test_keep_confirmation")
            )
    assert await _read_rows(postgres_session_factory) == original


async def test_watchdogs_skip_held_task_locks_and_recover_disjoint_batches(
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    now = await _database_now(postgres_session_factory)
    pairs = tuple(
        _delivery_pair(number, now=now, published_at=now - timedelta(minutes=10))
        for number in (1, 2)
    )
    await _seed_pairs(postgres_session_factory, *pairs)
    async with (
        postgres_session_factory() as first,
        postgres_session_factory() as second,
        first.begin(),
        second.begin(),
    ):
        assert await _recover_in_transaction(first, batch_size=1) == 1
        async with asyncio.timeout(5):
            assert await _recover_in_transaction(second, batch_size=1) == 1
        assert await _recover_in_transaction(first, batch_size=1) == 0
        assert await _recover_in_transaction(second, batch_size=1) == 0
    tasks, events = await _read_rows(postgres_session_factory)
    assert {values["status"] for values in tasks.values()} == {TaskStatus.PENDING}
    assert all(values["published_at"] is None for values in events.values())
    assert await _watchdog(postgres_session_factory).recover_once() == 0


@pytest.mark.parametrize("competitor", ["claim", "cancellation"])
async def test_claim_or_cancellation_held_first_fences_watchdog(
    postgres_session_factory: AsyncSessionFactory,
    competitor: Literal["claim", "cancellation"],
) -> None:
    now = await _database_now(postgres_session_factory)
    task, event = _delivery_pair(1, now=now, published_at=now - timedelta(minutes=10))
    assert task.dispatch_token is not None
    await _seed_pairs(postgres_session_factory, (task, event))
    async with postgres_session_factory() as holder, holder.begin():
        if competitor == "claim":
            assert (
                await TaskExecutionRepository(holder).claim_for_execution(
                    task.id,
                    dispatch_token=task.dispatch_token,
                    lease_duration=_EXECUTION_LEASE,
                )
                is not None
            )
        else:
            cancelled = await TaskRepository(holder).cancel_with_outbox(
                task.id,
                event_type=TASK_ROUTING_KEY,
            )
            assert cancelled is not None
            assert cancelled.changed
        async with asyncio.timeout(5):
            assert await _watchdog(postgres_session_factory).recover_once() == 0
    assert await _watchdog(postgres_session_factory).recover_once() == 0
    _tasks, events = await _read_rows(postgres_session_factory)
    assert events[event.id]["published_at"] == event.published_at


@pytest.mark.parametrize("competitor", ["claim", "cancellation"])
async def test_claim_or_cancellation_waits_for_watchdog_then_observes_its_commit(
    postgres_engine: AsyncEngine,
    postgres_session_factory: AsyncSessionFactory,
    competitor: Literal["claim", "cancellation"],
) -> None:
    now = await _database_now(postgres_session_factory)
    task, event = _delivery_pair(1, now=now, published_at=now - timedelta(minutes=10))
    dispatch_token = task.dispatch_token
    assert dispatch_token is not None
    await _seed_pairs(postgres_session_factory, (task, event))

    async def run_competitor(factory: AsyncSessionFactory) -> None:
        async with factory.begin() as session:
            if competitor == "claim":
                execution = await TaskExecutionRepository(session).claim_for_execution(
                    task.id,
                    dispatch_token=dispatch_token,
                    lease_duration=_EXECUTION_LEASE,
                )
                assert execution is not None
                assert execution.attempt_count == 1
            else:
                cancelled = await TaskRepository(session).cancel_with_outbox(
                    task.id,
                    event_type=TASK_ROUTING_KEY,
                )
                assert cancelled is not None
                assert cancelled.changed

    async with (
        postgres_session_factory() as holder,
        postgres_engine.connect() as waiting_connection,
        postgres_engine.connect() as observer,
        holder.begin() as holder_transaction,
    ):
        holder_pid = await holder.scalar(text("SELECT pg_backend_pid()"))
        assert isinstance(holder_pid, int)
        assert await _recover_in_transaction(holder) == 1
        waiter_pid = await waiting_connection.scalar(text("SELECT pg_backend_pid()"))
        assert isinstance(waiter_pid, int)
        assert waiter_pid != holder_pid
        await waiting_connection.commit()
        factory: AsyncSessionFactory = async_sessionmaker[AsyncSession](
            bind=waiting_connection,
            autoflush=False,
            expire_on_commit=False,
        )
        waiter = asyncio.create_task(run_competitor(factory))
        try:
            async with asyncio.timeout(5):
                while True:
                    if waiter.done():
                        await waiter
                        pytest.fail("The competing operation did not wait for the task lock")
                    if await observer.scalar(
                        text("SELECT :holder_pid = ANY(pg_blocking_pids(:waiter_pid))"),
                        {"holder_pid": holder_pid, "waiter_pid": waiter_pid},
                    ):
                        break
            await holder_transaction.commit()
            async with asyncio.timeout(5):
                await waiter
        finally:
            if not waiter.done():
                waiter.cancel()
            async with asyncio.timeout(5):
                await asyncio.gather(waiter, return_exceptions=True)

    tasks, events = await _read_rows(postgres_session_factory)
    assert events[event.id]["published_at"] is None
    if competitor == "claim":
        assert tasks[task.id]["status"] is TaskStatus.IN_PROGRESS
        assert tasks[task.id]["attempt_count"] == 1
        assert events[event.id]["discarded_at"] is None
        async with postgres_session_factory.begin() as session:
            assert (
                await TaskExecutionRepository(session).claim_for_execution(
                    task.id,
                    dispatch_token=dispatch_token,
                    lease_duration=_EXECUTION_LEASE,
                )
                is None
            )
    else:
        assert tasks[task.id]["status"] is TaskStatus.CANCELLED
        assert tasks[task.id]["attempt_count"] == 0
        assert events[event.id]["discarded_at"] is not None
    assert await _watchdog(postgres_session_factory).recover_once() == 0


async def test_reopened_event_is_publishable_and_fresh_confirmation_is_not_replayed(
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    now = await _database_now(postgres_session_factory)
    task, event = _delivery_pair(1, now=now, published_at=now - timedelta(minutes=10))
    await _seed_pairs(postgres_session_factory, (task, event))
    watchdog = _watchdog(postgres_session_factory)
    assert await watchdog.recover_once() == 1

    async with postgres_session_factory.begin() as session:
        claimed = await OutboxRepository(session).claim_batch(
            event_type=TASK_ROUTING_KEY,
            batch_size=1,
            lease_duration=_PUBLISH_LEASE,
        )
        assert len(claimed) == 1
        assert claimed[0].id == event.id
        assert claimed[0].payload == event.payload
        assert claimed[0].publish_attempts == 3
    assert await watchdog.recover_once() == 0
    async with postgres_session_factory.begin() as session:
        assert await OutboxRepository(session).mark_published(
            event.id,
            publisher_token=claimed[0].publisher_token,
        )

    assert await watchdog.recover_once() == 0
    tasks, events = await _read_rows(postgres_session_factory)
    assert tasks[task.id]["status"] is TaskStatus.PENDING
    assert tasks[task.id]["attempt_count"] == 0
    assert tasks[task.id]["dispatch_token"] == task.dispatch_token
    assert events[event.id]["published_at"] is not None
    assert events[event.id]["last_error"] is None
    assert events[event.id]["publish_attempts"] == 3
