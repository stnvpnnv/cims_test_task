"""Task execution ownership guarantees exercised against PostgreSQL."""

import asyncio
from datetime import timedelta
from uuid import UUID

import pytest
from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from cims_task_service.application.task_creation import CreateTaskCommand, create_task
from cims_task_service.domain.task import TaskPriority, TaskStatus
from cims_task_service.infrastructure.database.models import TaskModel
from cims_task_service.infrastructure.database.session import AsyncSessionFactory
from cims_task_service.infrastructure.database.task_execution_repository import (
    ClaimedTaskExecution,
    TaskExecutionRepository,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_EXECUTION_LEASE = timedelta(minutes=1)


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
