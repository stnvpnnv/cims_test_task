"""Transactional outbox reservation guarantees exercised against PostgreSQL."""

import asyncio
from datetime import datetime, timedelta
from uuid import UUID

import pytest
from sqlalchemy import func, select, text, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from cims_task_service.application.task_cancellation import cancel_task
from cims_task_service.application.task_creation import CreateTaskCommand, create_task
from cims_task_service.domain.task import TaskPriority, TaskStatus
from cims_task_service.infrastructure.database.models import OutboxEventModel, TaskModel
from cims_task_service.infrastructure.database.outbox_repository import (
    ClaimedOutboxEvent,
    OutboxRepository,
)
from cims_task_service.infrastructure.database.session import AsyncSessionFactory
from cims_task_service.infrastructure.messaging.topology import TASK_ROUTING_KEY

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_PUBLISH_LEASE = timedelta(minutes=1)
_LIVE_PUBLISHER_TOKEN = UUID("10000000-0000-4000-8000-000000000001")


async def _create_task(
    session_factory: AsyncSessionFactory,
    *,
    name: str,
    priority: TaskPriority,
) -> TaskModel:
    result = await create_task(
        CreateTaskCommand(
            name=name,
            description=f"Integration fixture for {name}",
            priority=priority,
        ),
        session_factory=session_factory,
        max_attempts=3,
    )
    assert result.created is True
    return result.task


async def _claim_and_commit(
    session_factory: AsyncSessionFactory,
    *,
    batch_size: int,
) -> tuple[ClaimedOutboxEvent, ...]:
    async with session_factory.begin() as session:
        return await OutboxRepository(session).claim_batch(
            event_type=TASK_ROUTING_KEY,
            batch_size=batch_size,
            lease_duration=_PUBLISH_LEASE,
        )


async def _read_state(
    session_factory: AsyncSessionFactory,
) -> tuple[dict[UUID, TaskModel], dict[UUID, OutboxEventModel]]:
    async with session_factory() as session:
        tasks = {task.id: task for task in (await session.scalars(select(TaskModel))).all()}
        event_rows = tuple((await session.scalars(select(OutboxEventModel))).all())
        events = {event.task_id: event for event in event_rows}
        assert len(events) == len(event_rows)
    return tasks, events


async def test_claim_orders_ready_events_and_commits_pending_atomically(
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    """Only ready events are leased, in priority order, with their task transition."""

    ready_low = await _create_task(
        postgres_session_factory,
        name="Ready low",
        priority=TaskPriority.LOW,
    )
    ready_high = await _create_task(
        postgres_session_factory,
        name="Ready high",
        priority=TaskPriority.HIGH,
    )
    future_high = await _create_task(
        postgres_session_factory,
        name="Future high",
        priority=TaskPriority.HIGH,
    )
    leased_high = await _create_task(
        postgres_session_factory,
        name="Leased high",
        priority=TaskPriority.HIGH,
    )
    async with postgres_session_factory.begin() as session:
        database_time = await session.scalar(select(func.clock_timestamp()))
        assert isinstance(database_time, datetime)
        await session.execute(
            update(TaskModel)
            .where(TaskModel.id == leased_high.id)
            .values(status=TaskStatus.PENDING)
        )
        await session.execute(
            update(OutboxEventModel)
            .where(OutboxEventModel.task_id == leased_high.id)
            .values(
                publish_attempts=1,
                publisher_token=_LIVE_PUBLISHER_TOKEN,
                lease_expires_at=database_time + _PUBLISH_LEASE,
            )
        )
        await session.execute(
            update(OutboxEventModel)
            .where(OutboxEventModel.task_id == future_high.id)
            .values(available_at=database_time + timedelta(hours=1))
        )

    claimed = await _claim_and_commit(postgres_session_factory, batch_size=10)

    assert tuple(event.task_id for event in claimed) == (ready_high.id, ready_low.id)
    assert all(event.publish_attempts == 1 for event in claimed)
    claimed_by_task = {event.task_id: event for event in claimed}
    tasks, events = await _read_state(postgres_session_factory)
    assert tasks[ready_high.id].status is TaskStatus.PENDING
    assert tasks[ready_low.id].status is TaskStatus.PENDING
    assert tasks[future_high.id].status is TaskStatus.NEW
    assert tasks[leased_high.id].status is TaskStatus.PENDING
    assert events[ready_high.id].publisher_token == claimed_by_task[ready_high.id].publisher_token
    assert events[ready_low.id].publisher_token == claimed_by_task[ready_low.id].publisher_token
    assert events[future_high.id].publisher_token is None
    assert events[future_high.id].publish_attempts == 0
    assert events[leased_high.id].publisher_token == _LIVE_PUBLISHER_TOKEN
    assert events[leased_high.id].lease_expires_at == database_time + _PUBLISH_LEASE
    assert events[leased_high.id].publish_attempts == 1


async def test_concurrent_claimers_skip_locked_tasks_and_get_disjoint_batches(
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    """A second dispatcher keeps making progress while the first transaction is open."""

    first_task = await _create_task(
        postgres_session_factory,
        name="Concurrent first",
        priority=TaskPriority.HIGH,
    )
    second_task = await _create_task(
        postgres_session_factory,
        name="Concurrent second",
        priority=TaskPriority.HIGH,
    )

    async with (
        postgres_session_factory() as first_session,
        postgres_session_factory() as second_session,
        first_session.begin(),
        second_session.begin(),
    ):
        first_claim = await OutboxRepository(first_session).claim_batch(
            event_type=TASK_ROUTING_KEY,
            batch_size=1,
            lease_duration=_PUBLISH_LEASE,
        )
        async with asyncio.timeout(10):
            second_claim = await OutboxRepository(second_session).claim_batch(
                event_type=TASK_ROUTING_KEY,
                batch_size=1,
                lease_duration=_PUBLISH_LEASE,
            )

        assert len(first_claim) == len(second_claim) == 1
        assert first_claim[0].task_id != second_claim[0].task_id
        assert {first_claim[0].task_id, second_claim[0].task_id} == {
            first_task.id,
            second_task.id,
        }
        assert first_claim[0].publisher_token != second_claim[0].publisher_token

    tasks, events = await _read_state(postgres_session_factory)
    assert {task.status for task in tasks.values()} == {TaskStatus.PENDING}
    assert {event.publish_attempts for event in events.values()} == {1}
    assert all(event.publisher_token is not None for event in events.values())


async def test_rolling_back_claim_restores_task_and_event_together(
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    """A failed reservation transaction cannot leave a PENDING task without its lease."""

    created = await _create_task(
        postgres_session_factory,
        name="Rollback reservation",
        priority=TaskPriority.MEDIUM,
    )
    async with postgres_session_factory() as session, session.begin() as transaction:
        claimed = await OutboxRepository(session).claim_batch(
            event_type=TASK_ROUTING_KEY,
            batch_size=1,
            lease_duration=_PUBLISH_LEASE,
        )
        pending_status = await session.scalar(
            select(TaskModel.status).where(TaskModel.id == created.id)
        )
        assert len(claimed) == 1
        assert pending_status is TaskStatus.PENDING
        await transaction.rollback()

    tasks, events = await _read_state(postgres_session_factory)
    assert tasks[created.id].status is TaskStatus.NEW
    assert events[created.id].publish_attempts == 0
    assert events[created.id].publisher_token is None
    assert events[created.id].lease_expires_at is None

    recovered = await _claim_and_commit(postgres_session_factory, batch_size=1)
    assert len(recovered) == 1
    assert recovered[0].task_id == created.id
    assert recovered[0].publish_attempts == 1


async def test_expired_lease_is_reclaimed_and_fences_the_previous_owner(
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    """Reclaim changes the token, so a delayed publisher cannot finalize the event."""

    created = await _create_task(
        postgres_session_factory,
        name="Expired publication lease",
        priority=TaskPriority.MEDIUM,
    )
    original = await _claim_and_commit(postgres_session_factory, batch_size=1)
    assert len(original) == 1
    async with postgres_session_factory.begin() as session:
        await session.execute(
            update(OutboxEventModel)
            .where(OutboxEventModel.task_id == created.id)
            .values(lease_expires_at=func.clock_timestamp() - timedelta(seconds=1))
        )

    reclaimed = await _claim_and_commit(postgres_session_factory, batch_size=1)

    assert len(reclaimed) == 1
    assert reclaimed[0].id == original[0].id
    assert reclaimed[0].publisher_token != original[0].publisher_token
    assert reclaimed[0].publish_attempts == 2
    async with postgres_session_factory.begin() as session:
        repository = OutboxRepository(session)
        assert (
            await repository.mark_published(
                original[0].id,
                publisher_token=original[0].publisher_token,
            )
            is False
        )
        assert (
            await repository.mark_published(
                reclaimed[0].id,
                publisher_token=reclaimed[0].publisher_token,
            )
            is True
        )

    _, events = await _read_state(postgres_session_factory)
    stored = events[created.id]
    assert stored.published_at is not None
    assert stored.publisher_token is None
    assert stored.lease_expires_at is None
    assert stored.last_error is None


async def test_reschedule_defers_retry_and_invalidates_the_released_token(
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    """A failed publish becomes unavailable until its database-backed retry time."""

    created = await _create_task(
        postgres_session_factory,
        name="Deferred publication retry",
        priority=TaskPriority.LOW,
    )
    claimed = await _claim_and_commit(postgres_session_factory, batch_size=1)
    assert len(claimed) == 1
    retry_delay = timedelta(hours=1)
    async with postgres_session_factory.begin() as session:
        repository = OutboxRepository(session)
        database_before = await session.scalar(select(func.clock_timestamp()))
        assert isinstance(database_before, datetime)
        assert (
            await repository.reschedule(
                claimed[0].id,
                publisher_token=claimed[0].publisher_token,
                retry_delay=retry_delay,
                failure_summary="publisher confirm timed out",
            )
            is True
        )
        database_after = await session.scalar(select(func.clock_timestamp()))
        assert isinstance(database_after, datetime)
        assert (
            await repository.mark_published(
                claimed[0].id,
                publisher_token=claimed[0].publisher_token,
            )
            is False
        )

    assert await _claim_and_commit(postgres_session_factory, batch_size=1) == ()
    tasks, events = await _read_state(postgres_session_factory)
    stored = events[created.id]
    assert tasks[created.id].status is TaskStatus.PENDING
    assert stored.publish_attempts == 1
    assert stored.publisher_token is None
    assert stored.lease_expires_at is None
    assert stored.last_error == "publisher confirm timed out"
    assert database_before + retry_delay <= stored.available_at <= database_after + retry_delay


async def test_cancellation_waits_for_claim_then_discards_the_reserved_event(
    postgres_engine: AsyncEngine,
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    """Cancellation serializes behind a held claim and fences its publisher."""

    created = await _create_task(
        postgres_session_factory,
        name="Cancellation during publication",
        priority=TaskPriority.HIGH,
    )
    assert created.dispatch_token is not None
    async with (
        postgres_session_factory() as holder,
        postgres_engine.connect() as waiting_connection,
        postgres_engine.connect() as observer,
        holder.begin() as holder_transaction,
    ):
        holder_pid = await holder.scalar(text("SELECT pg_backend_pid()"))
        claimed = await OutboxRepository(holder).claim_batch(
            event_type=TASK_ROUTING_KEY,
            batch_size=1,
            lease_duration=_PUBLISH_LEASE,
        )
        assert isinstance(holder_pid, int)
        assert len(claimed) == 1
        claimed_event = claimed[0]
        assert claimed_event.task_id == created.id

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
        waiter = asyncio.create_task(cancel_task(created.id, session_factory=waiting_factory))
        try:
            async with asyncio.timeout(10):
                while True:
                    if waiter.done():
                        await waiter
                        pytest.fail(
                            "Cancellation completed before PostgreSQL reported the task lock"
                        )
                    blocked = await observer.scalar(
                        text("SELECT :holder_pid = ANY(pg_blocking_pids(:waiter_pid))"),
                        {"holder_pid": holder_pid, "waiter_pid": waiter_pid},
                    )
                    if blocked:
                        break

            await holder_transaction.commit()
            async with asyncio.timeout(10):
                cancelled = await waiter
            assert cancelled.status is TaskStatus.CANCELLED
        finally:
            if not waiter.done():
                waiter.cancel()
            async with asyncio.timeout(10):
                await asyncio.gather(waiter, return_exceptions=True)

    async with postgres_session_factory.begin() as session:
        assert (
            await OutboxRepository(session).mark_published(
                claimed_event.id,
                publisher_token=claimed_event.publisher_token,
            )
            is False
        )

    tasks, events = await _read_state(postgres_session_factory)
    stored_task = tasks[created.id]
    stored_event = events[created.id]
    assert stored_task.status is TaskStatus.CANCELLED
    assert stored_task.finished_at is not None
    assert stored_task.dispatch_token is None
    assert stored_task.execution_token is None
    assert stored_task.lease_expires_at is None
    assert stored_event.publish_attempts == 1
    assert stored_event.published_at is None
    assert stored_event.discarded_at is not None
    assert stored_event.publisher_token is None
    assert stored_event.lease_expires_at is None
    assert stored_event.last_error is None
