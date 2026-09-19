"""Task execution ownership guarantees exercised against PostgreSQL."""

import asyncio
from datetime import timedelta
from uuid import UUID

import pytest
from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from cims_task_service.application.task_creation import CreateTaskCommand, create_task
from cims_task_service.domain.task import TaskPriority, TaskStatus
from cims_task_service.infrastructure.database.models import JsonObject, TaskModel
from cims_task_service.infrastructure.database.session import AsyncSessionFactory
from cims_task_service.infrastructure.database.task_execution_repository import (
    ClaimedTaskExecution,
    TaskExecutionRepository,
)
from cims_task_service.infrastructure.database.task_repository import TaskRepository
from cims_task_service.infrastructure.messaging.topology import TASK_ROUTING_KEY

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_EXECUTION_LEASE = timedelta(minutes=1)
_STALE_EXECUTION_TOKEN = UUID("40000000-0000-4000-8000-000000000004")
_RESULT: JsonObject = {
    "name_length": 20,
    "description_length": 37,
}


async def _create_pending_task(
    session_factory: AsyncSessionFactory,
) -> tuple[UUID, UUID]:
    created = await create_task(
        CreateTaskCommand(
            name="Concurrent execution",
            description="Only one worker may acquire this task",
            priority=TaskPriority.HIGH,
        ),
        session_factory=session_factory,
        max_attempts=3,
    )
    dispatch_token = created.task.dispatch_token
    assert dispatch_token is not None

    async with session_factory.begin() as session:
        await session.execute(
            update(TaskModel)
            .where(
                TaskModel.id == created.task.id,
                TaskModel.status == TaskStatus.NEW,
            )
            .values(status=TaskStatus.PENDING)
        )

    return created.task.id, dispatch_token


async def _claim_and_commit(
    session_factory: AsyncSessionFactory,
    *,
    task_id: UUID,
    dispatch_token: UUID,
) -> ClaimedTaskExecution | None:
    async with session_factory.begin() as session:
        return await TaskExecutionRepository(session).claim_for_execution(
            task_id,
            dispatch_token=dispatch_token,
            lease_duration=_EXECUTION_LEASE,
        )


async def _complete_and_commit(
    session_factory: AsyncSessionFactory,
    *,
    task_id: UUID,
    execution_token: UUID,
) -> bool:
    async with session_factory.begin() as session:
        return await TaskExecutionRepository(session).complete_execution(
            task_id,
            execution_token=execution_token,
            result=_RESULT,
        )


async def test_concurrent_claims_grant_exactly_one_execution_lease(
    postgres_engine: AsyncEngine,
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    """A waiting delivery rechecks the fenced predicate after its competitor commits."""

    task_id, dispatch_token = await _create_pending_task(postgres_session_factory)

    async with (
        postgres_session_factory() as holder,
        postgres_engine.connect() as waiting_connection,
        postgres_engine.connect() as observer,
        holder.begin() as holder_transaction,
    ):
        holder_pid = await holder.scalar(text("SELECT pg_backend_pid()"))
        waiter_pid = await waiting_connection.scalar(text("SELECT pg_backend_pid()"))
        observer_pid = await observer.scalar(text("SELECT pg_backend_pid()"))
        assert isinstance(holder_pid, int)
        assert isinstance(waiter_pid, int)
        assert isinstance(observer_pid, int)
        assert len({holder_pid, waiter_pid, observer_pid}) == 3
        await waiting_connection.commit()

        first_claim = await TaskExecutionRepository(holder).claim_for_execution(
            task_id,
            dispatch_token=dispatch_token,
            lease_duration=_EXECUTION_LEASE,
        )
        assert first_claim is not None

        waiting_factory: AsyncSessionFactory = async_sessionmaker[AsyncSession](
            bind=waiting_connection,
            autoflush=False,
            expire_on_commit=False,
        )
        waiter = asyncio.create_task(
            _claim_and_commit(
                waiting_factory,
                task_id=task_id,
                dispatch_token=dispatch_token,
            )
        )
        try:
            async with asyncio.timeout(10):
                while True:
                    if waiter.done():
                        await waiter
                        pytest.fail(
                            "Second claim completed before PostgreSQL reported the row lock"
                        )
                    blocked = await observer.scalar(
                        text("SELECT :holder_pid = ANY(pg_blocking_pids(:waiter_pid))"),
                        {"holder_pid": holder_pid, "waiter_pid": waiter_pid},
                    )
                    if blocked:
                        break

            await holder_transaction.commit()
            async with asyncio.timeout(10):
                second_claim = await waiter
        finally:
            if not waiter.done():
                waiter.cancel()
            async with asyncio.timeout(10):
                await asyncio.gather(waiter, return_exceptions=True)

    assert second_claim is None
    async with postgres_session_factory() as session:
        stored = await session.scalar(select(TaskModel).where(TaskModel.id == task_id))

    assert stored is not None
    assert stored.status is TaskStatus.IN_PROGRESS
    assert stored.attempt_count == first_claim.attempt_count == 1
    assert stored.max_attempts == first_claim.max_attempts == 3
    assert stored.dispatch_token is None
    assert stored.execution_token == first_claim.execution_token
    assert stored.lease_expires_at == first_claim.lease_expires_at
    assert stored.started_at is not None
    assert stored.lease_expires_at is not None
    assert stored.lease_expires_at > stored.started_at
    assert stored.finished_at is None
    assert stored.result is None
    assert stored.error is None


async def test_current_execution_owner_persists_a_successful_result(
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    """Completion commits a valid terminal state and releases its execution lease."""

    task_id, dispatch_token = await _create_pending_task(postgres_session_factory)
    claimed = await _claim_and_commit(
        postgres_session_factory,
        task_id=task_id,
        dispatch_token=dispatch_token,
    )
    assert claimed is not None

    async with postgres_session_factory.begin() as session:
        completed = await TaskExecutionRepository(session).complete_execution(
            task_id,
            execution_token=claimed.execution_token,
            result=_RESULT,
        )

    assert completed is True
    async with postgres_session_factory() as session:
        stored = await session.scalar(select(TaskModel).where(TaskModel.id == task_id))

    assert stored is not None
    assert stored.status is TaskStatus.COMPLETED
    assert stored.attempt_count == 1
    assert stored.started_at is not None
    assert stored.finished_at is not None
    assert stored.finished_at >= stored.started_at
    assert stored.result == _RESULT
    assert stored.error is None
    assert stored.dispatch_token is None
    assert stored.execution_token is None
    assert stored.lease_expires_at is None


async def test_stale_execution_token_cannot_persist_a_result(
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    """A worker without the current fencing token leaves the active attempt unchanged."""

    task_id, dispatch_token = await _create_pending_task(postgres_session_factory)
    claimed = await _claim_and_commit(
        postgres_session_factory,
        task_id=task_id,
        dispatch_token=dispatch_token,
    )
    assert claimed is not None

    async with postgres_session_factory.begin() as session:
        completed = await TaskExecutionRepository(session).complete_execution(
            task_id,
            execution_token=_STALE_EXECUTION_TOKEN,
            result=_RESULT,
        )

    assert completed is False
    async with postgres_session_factory() as session:
        stored = await session.scalar(select(TaskModel).where(TaskModel.id == task_id))

    assert stored is not None
    assert stored.status is TaskStatus.IN_PROGRESS
    assert stored.attempt_count == 1
    assert stored.result is None
    assert stored.finished_at is None
    assert stored.execution_token == claimed.execution_token
    assert stored.lease_expires_at == claimed.lease_expires_at


async def test_cancellation_fences_a_late_successful_result(
    postgres_engine: AsyncEngine,
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    """A completion waiting on cancellation rechecks ownership after the lock releases."""

    task_id, dispatch_token = await _create_pending_task(postgres_session_factory)
    claimed = await _claim_and_commit(
        postgres_session_factory,
        task_id=task_id,
        dispatch_token=dispatch_token,
    )
    assert claimed is not None

    async with (
        postgres_session_factory() as holder,
        postgres_engine.connect() as waiting_connection,
        postgres_engine.connect() as observer,
        holder.begin() as holder_transaction,
    ):
        holder_pid = await holder.scalar(text("SELECT pg_backend_pid()"))
        waiter_pid = await waiting_connection.scalar(text("SELECT pg_backend_pid()"))
        observer_pid = await observer.scalar(text("SELECT pg_backend_pid()"))
        assert isinstance(holder_pid, int)
        assert isinstance(waiter_pid, int)
        assert isinstance(observer_pid, int)
        assert len({holder_pid, waiter_pid, observer_pid}) == 3
        await waiting_connection.commit()

        cancelled = await TaskRepository(holder).cancel_with_outbox(
            task_id,
            event_type=TASK_ROUTING_KEY,
        )
        assert cancelled is not None
        assert cancelled.changed is True
        assert cancelled.task.status is TaskStatus.CANCELLED
        assert cancelled.task.finished_at is not None
        cancelled_at = cancelled.task.finished_at

        waiting_factory: AsyncSessionFactory = async_sessionmaker[AsyncSession](
            bind=waiting_connection,
            autoflush=False,
            expire_on_commit=False,
        )
        waiter = asyncio.create_task(
            _complete_and_commit(
                waiting_factory,
                task_id=task_id,
                execution_token=claimed.execution_token,
            )
        )
        try:
            async with asyncio.timeout(10):
                while True:
                    if waiter.done():
                        await waiter
                        pytest.fail(
                            "Completion finished before PostgreSQL reported the cancellation lock"
                        )
                    blocked = await observer.scalar(
                        text("SELECT :holder_pid = ANY(pg_blocking_pids(:waiter_pid))"),
                        {"holder_pid": holder_pid, "waiter_pid": waiter_pid},
                    )
                    if blocked:
                        break

            await holder_transaction.commit()
            async with asyncio.timeout(10):
                completed = await waiter
        finally:
            if not waiter.done():
                waiter.cancel()
            async with asyncio.timeout(10):
                await asyncio.gather(waiter, return_exceptions=True)

    assert completed is False
    async with postgres_session_factory() as session:
        stored = await session.scalar(select(TaskModel).where(TaskModel.id == task_id))

    assert stored is not None
    assert stored.status is TaskStatus.CANCELLED
    assert stored.finished_at == cancelled_at
    assert stored.result is None
    assert stored.error is None
    assert stored.execution_token is None
    assert stored.lease_expires_at is None
