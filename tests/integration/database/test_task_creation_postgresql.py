"""Task creation guarantees exercised against independent PostgreSQL transactions."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from uuid import UUID

import pytest
from sqlalchemy import select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSessionTransaction, async_sessionmaker

from cims_task_service.application.idempotency import (
    fingerprint_task_creation_request,
    hash_task_creation_idempotency_key,
)
from cims_task_service.application.task_creation import (
    CreateTaskCommand,
    CreateTaskResult,
    IdempotencyKeyConflictError,
    create_task,
)
from cims_task_service.domain.task import TaskPriority, TaskStatus
from cims_task_service.infrastructure.database.models import OutboxEventModel, TaskModel
from cims_task_service.infrastructure.database.session import AsyncSessionFactory
from cims_task_service.infrastructure.database.task_repository import TaskRepository
from cims_task_service.infrastructure.messaging.topology import TASK_ROUTING_KEY

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_KEY = "60c856f4-1704-4848-95d3-354c382cd6a4"
_COMMAND = CreateTaskCommand(
    name="Daily report",
    description="Aggregate yesterday's records",
    priority=TaskPriority.MEDIUM,
    idempotency_key=_KEY,
)


def _fingerprint(command: CreateTaskCommand) -> bytes:
    return fingerprint_task_creation_request(
        name=command.name,
        description=command.description,
        priority=command.priority,
    )


async def _read_rows(
    session_factory: AsyncSessionFactory,
) -> tuple[list[TaskModel], list[OutboxEventModel]]:
    async with session_factory() as session:
        tasks = list((await session.scalars(select(TaskModel))).all())
        events = list((await session.scalars(select(OutboxEventModel))).all())
    return tasks, events


@dataclass(frozen=True, slots=True)
class _BlockedCreation:
    transaction: AsyncSessionTransaction
    task_id: UUID
    dispatch_token: UUID | None
    waiter: asyncio.Task[CreateTaskResult]


@asynccontextmanager
async def _blocked_creation(
    engine: AsyncEngine,
    session_factory: AsyncSessionFactory,
    contender: CreateTaskCommand,
) -> AsyncIterator[_BlockedCreation]:
    """Yield only after PostgreSQL confirms the contender is blocked by our insert."""

    async with (
        session_factory() as holder,
        engine.connect() as waiting_connection,
        engine.connect() as observer,
        holder.begin() as transaction,
    ):
        holder_pid = await holder.scalar(text("SELECT pg_backend_pid()"))
        waiter_pid = await waiting_connection.scalar(text("SELECT pg_backend_pid()"))
        observer_pid = await observer.scalar(text("SELECT pg_backend_pid()"))
        assert len({holder_pid, waiter_pid, observer_pid}) == 3
        # The application must start its own transaction on the waiting connection.
        await waiting_connection.commit()
        waiting_factory = async_sessionmaker(
            bind=waiting_connection, autoflush=False, expire_on_commit=False
        )
        stored = await TaskRepository(holder).create_with_outbox(
            name=_COMMAND.name,
            description=_COMMAND.description,
            priority=_COMMAND.priority,
            max_attempts=3,
            idempotency_key_hash=hash_task_creation_idempotency_key(_KEY),
            request_fingerprint=_fingerprint(_COMMAND),
            event_type=TASK_ROUTING_KEY,
            message_priority=2,
        )
        assert stored.created is True
        # Rollback expires ORM attributes, so capture these before releasing the holder.
        task_id, dispatch_token = stored.task.id, stored.task.dispatch_token
        waiter = asyncio.create_task(
            create_task(contender, session_factory=waiting_factory, max_attempts=3)
        )
        try:
            async with asyncio.timeout(10):
                while True:
                    if waiter.done():
                        await waiter
                        pytest.fail("Concurrent creation completed before the holder was released")
                    blocked = await observer.scalar(
                        text("SELECT :holder_pid = ANY(pg_blocking_pids(:waiter_pid))"),
                        {"holder_pid": holder_pid, "waiter_pid": waiter_pid},
                    )
                    if blocked:
                        break
                yield _BlockedCreation(transaction, task_id, dispatch_token, waiter)
        finally:
            if not waiter.done():
                waiter.cancel()
            async with asyncio.timeout(10):
                await asyncio.gather(waiter, return_exceptions=True)


@pytest.mark.parametrize(
    ("priority", "message_priority"),
    [(TaskPriority.LOW, 1), (TaskPriority.MEDIUM, 2), (TaskPriority.HIGH, 3)],
)
async def test_creation_commits_task_and_matching_outbox(
    postgres_session_factory: AsyncSessionFactory,
    priority: TaskPriority,
    message_priority: int,
) -> None:
    command = replace(_COMMAND, priority=priority)
    result = await create_task(command, session_factory=postgres_session_factory, max_attempts=4)

    tasks, events = await _read_rows(postgres_session_factory)
    assert result.created is True
    assert len(tasks) == len(events) == 1
    task, event = tasks[0], events[0]
    assert task.id == result.task.id
    assert (task.name, task.description, task.priority) == (
        command.name,
        command.description,
        command.priority,
    )
    assert task.status is TaskStatus.NEW
    assert task.idempotency_key_hash == hash_task_creation_idempotency_key(_KEY)
    assert task.request_fingerprint == _fingerprint(command)
    assert task.attempt_count == 0
    assert task.max_attempts == 4
    assert task.dispatch_token is not None
    assert task.started_at is task.finished_at is None
    assert task.result is task.error is None
    assert task.execution_token is None
    assert task.lease_expires_at is None
    assert task.created_at.tzinfo is not None
    assert event.task_id == task.id
    assert event.event_type == TASK_ROUTING_KEY
    assert event.payload == {
        "task_id": str(task.id),
        "dispatch_token": str(task.dispatch_token),
    }
    assert event.message_priority == message_priority
    assert event.publish_attempts == 0
    assert event.published_at is event.discarded_at is None
    assert event.publisher_token is None
    assert event.lease_expires_at is None
    assert event.last_error is None
    assert event.available_at == event.created_at == task.created_at


async def test_requests_without_key_create_independent_tasks(
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    command = replace(_COMMAND, idempotency_key=None)
    first = await create_task(command, session_factory=postgres_session_factory, max_attempts=3)
    second = await create_task(command, session_factory=postgres_session_factory, max_attempts=3)

    tasks, events = await _read_rows(postgres_session_factory)
    assert first.created is second.created is True
    assert first.task.id != second.task.id
    assert len(tasks) == len(events) == 2
    assert {task.id for task in tasks} == {first.task.id, second.task.id}
    assert {event.task_id for event in events} == {first.task.id, second.task.id}
    assert all(task.idempotency_key_hash is task.request_fingerprint is None for task in tasks)


async def test_replay_returns_current_resource_without_overwriting_original_metadata(
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    first = await create_task(_COMMAND, session_factory=postgres_session_factory, max_attempts=3)
    async with postgres_session_factory.begin() as session:
        await session.execute(
            update(TaskModel)
            .where(TaskModel.id == first.task.id)
            .values(priority=TaskPriority.HIGH)
        )

    replay = await create_task(_COMMAND, session_factory=postgres_session_factory, max_attempts=7)

    tasks, events = await _read_rows(postgres_session_factory)
    assert replay.created is False
    assert replay.task.id == first.task.id
    assert replay.task.priority is TaskPriority.HIGH
    assert replay.task.max_attempts == 3
    assert replay.task.request_fingerprint == _fingerprint(_COMMAND)
    assert replay.task.dispatch_token == first.task.dispatch_token
    assert replay.task.created_at == first.task.created_at
    assert len(tasks) == len(events) == 1
    assert tasks[0].priority is TaskPriority.HIGH
    assert events[0].task_id == first.task.id
    assert events[0].message_priority == 2


async def test_reused_key_with_changed_payload_preserves_original_task(
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    first = await create_task(_COMMAND, session_factory=postgres_session_factory, max_attempts=3)

    with pytest.raises(IdempotencyKeyConflictError) as error_info:
        await create_task(
            replace(_COMMAND, description="A different report"),
            session_factory=postgres_session_factory,
            max_attempts=3,
        )

    tasks, events = await _read_rows(postgres_session_factory)
    assert error_info.value.task_id == first.task.id
    assert len(tasks) == len(events) == 1
    assert tasks[0].id == events[0].task_id == first.task.id
    assert tasks[0].description == _COMMAND.description
    assert tasks[0].request_fingerprint == _fingerprint(_COMMAND)


async def test_concurrent_matching_request_replays_after_first_transaction_commits(
    postgres_engine: AsyncEngine,
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    async with _blocked_creation(postgres_engine, postgres_session_factory, _COMMAND) as race:
        await race.transaction.commit()
        replay = await race.waiter
        assert replay.created is False
        assert replay.task.id == race.task_id
        assert replay.task.dispatch_token == race.dispatch_token

    tasks, events = await _read_rows(postgres_session_factory)
    assert len(tasks) == len(events) == 1
    assert tasks[0].id == events[0].task_id == replay.task.id


async def test_concurrent_changed_request_conflicts_after_first_transaction_commits(
    postgres_engine: AsyncEngine,
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    contender = replace(_COMMAND, name="Another report")
    async with _blocked_creation(postgres_engine, postgres_session_factory, contender) as race:
        await race.transaction.commit()
        with pytest.raises(IdempotencyKeyConflictError) as error_info:
            await race.waiter
        assert error_info.value.task_id == race.task_id

    tasks, events = await _read_rows(postgres_session_factory)
    assert len(tasks) == len(events) == 1
    assert tasks[0].id == events[0].task_id == race.task_id
    assert tasks[0].name == _COMMAND.name
    assert tasks[0].request_fingerprint == _fingerprint(_COMMAND)


async def test_concurrent_request_creates_task_after_first_transaction_rolls_back(
    postgres_engine: AsyncEngine,
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    async with _blocked_creation(postgres_engine, postgres_session_factory, _COMMAND) as race:
        await race.transaction.rollback()
        result = await race.waiter
        assert result.created is True
        assert result.task.id != race.task_id

    tasks, events = await _read_rows(postgres_session_factory)
    assert len(tasks) == len(events) == 1
    assert tasks[0].id == events[0].task_id == result.task.id
    assert events[0].payload == {
        "task_id": str(result.task.id),
        "dispatch_token": str(result.task.dispatch_token),
    }


async def test_outbox_constraint_failure_rolls_back_task_and_releases_idempotency_key(
    postgres_engine: AsyncEngine,
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    async with postgres_engine.begin() as connection:
        await connection.execute(
            text("ALTER TABLE outbox_events ADD CONSTRAINT ck_test_reject_outbox CHECK (false)")
        )
    try:
        with pytest.raises(IntegrityError) as error_info:
            await create_task(_COMMAND, session_factory=postgres_session_factory, max_attempts=3)
        assert "ck_test_reject_outbox" in str(error_info.value.orig)
        tasks, events = await _read_rows(postgres_session_factory)
        assert tasks == []
        assert events == []
    finally:
        async with postgres_engine.begin() as connection:
            await connection.execute(
                text("ALTER TABLE outbox_events DROP CONSTRAINT ck_test_reject_outbox")
            )

    result = await create_task(_COMMAND, session_factory=postgres_session_factory, max_attempts=3)

    tasks, events = await _read_rows(postgres_session_factory)
    assert result.created is True
    assert len(tasks) == len(events) == 1
    assert tasks[0].id == events[0].task_id == result.task.id
