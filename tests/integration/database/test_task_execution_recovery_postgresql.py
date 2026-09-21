"""Task execution recovery guarantees exercised against PostgreSQL."""

import asyncio
from datetime import datetime, timedelta
from uuid import UUID

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from cims_task_service.application.task_execution_recovery import (
    RecoveryBatchResult,
    TaskExecutionRecovery,
)
from cims_task_service.domain.task import TaskPriority, TaskStatus
from cims_task_service.infrastructure.database.models import OutboxEventModel, TaskModel
from cims_task_service.infrastructure.database.session import AsyncSessionFactory
from cims_task_service.infrastructure.messaging.topology import TASK_ROUTING_KEY

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_EXPECTED_ERROR = {"code": "EXECUTION_LEASE_EXPIRED", "retryable": False}


def _expired_execution(
    *,
    task_number: int,
    lease_expires_at: datetime,
    priority: TaskPriority,
    attempt_count: int,
    max_attempts: int = 3,
) -> TaskModel:
    return TaskModel(
        id=UUID(int=task_number),
        name=f"Recovery candidate {task_number}",
        description="Expired worker execution",
        priority=priority,
        status=TaskStatus.IN_PROGRESS,
        created_at=lease_expires_at - timedelta(hours=2),
        started_at=lease_expires_at - timedelta(hours=1),
        attempt_count=attempt_count,
        max_attempts=max_attempts,
        execution_token=UUID(int=task_number + 100),
        lease_expires_at=lease_expires_at,
    )


async def _database_now(session_factory: AsyncSessionFactory) -> datetime:
    async with session_factory.begin() as session:
        current_time = await session.scalar(select(func.clock_timestamp()))
    assert isinstance(current_time, datetime)
    return current_time


async def _read_rows(
    session_factory: AsyncSessionFactory,
) -> tuple[list[TaskModel], list[OutboxEventModel]]:
    async with session_factory() as session:
        tasks = list(await session.scalars(select(TaskModel).order_by(TaskModel.id)))
        events = list(await session.scalars(select(OutboxEventModel).order_by(OutboxEventModel.id)))
    return tasks, events


async def test_recovery_commits_retries_and_terminal_failures_atomically(
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    now = await _database_now(postgres_session_factory)
    expired_at = now - timedelta(minutes=1)
    retry_delays = {1: timedelta(seconds=11), 2: timedelta(seconds=23)}
    observed_attempts: list[int] = []
    candidates = (
        _expired_execution(
            task_number=1,
            lease_expires_at=expired_at,
            priority=TaskPriority.LOW,
            attempt_count=1,
        ),
        _expired_execution(
            task_number=2,
            lease_expires_at=expired_at,
            priority=TaskPriority.HIGH,
            attempt_count=2,
        ),
        _expired_execution(
            task_number=3,
            lease_expires_at=expired_at,
            priority=TaskPriority.MEDIUM,
            attempt_count=3,
        ),
    )
    original_started_at = {candidate.id: candidate.started_at for candidate in candidates}
    async with postgres_session_factory.begin() as session:
        session.add_all(candidates)

    def retry_delay_for_attempt(attempt_count: int) -> timedelta:
        observed_attempts.append(attempt_count)
        return retry_delays[attempt_count]

    recovery = TaskExecutionRecovery(
        postgres_session_factory,
        batch_size=10,
        retry_delay_for_attempt=retry_delay_for_attempt,
    )
    before_recovery = await _database_now(postgres_session_factory)
    result = await recovery.recover_once()
    after_recovery = await _database_now(postgres_session_factory)

    assert result == RecoveryBatchResult(locked=3, retried=2, failed=1)
    assert observed_attempts == [1, 2]
    tasks, events = await _read_rows(postgres_session_factory)
    assert len(tasks) == 3
    assert len(events) == 2

    events_by_task_id = {event.task_id: event for event in events}
    expected_message_priorities = {
        UUID(int=1): 1,
        UUID(int=2): 3,
    }
    expected_attempt_counts = {
        UUID(int=1): 1,
        UUID(int=2): 2,
    }
    for task in tasks[:2]:
        assert task.status is TaskStatus.PENDING
        assert task.attempt_count == expected_attempt_counts[task.id]
        assert task.max_attempts == 3
        assert task.started_at == original_started_at[task.id]
        assert task.finished_at is None
        assert task.result is None
        assert task.error is None
        assert task.execution_token is None
        assert task.lease_expires_at is None
        assert task.dispatch_token is not None

        event = events_by_task_id[task.id]
        retry_delay = retry_delays[task.attempt_count]
        assert event.event_type == TASK_ROUTING_KEY
        assert event.payload == {
            "task_id": str(task.id),
            "dispatch_token": str(task.dispatch_token),
        }
        assert event.message_priority == expected_message_priorities[task.id]
        assert before_recovery + retry_delay <= event.available_at <= after_recovery + retry_delay
        assert event.available_at >= event.created_at
        assert event.published_at is None
        assert event.discarded_at is None
        assert event.publish_attempts == 0
        assert event.publisher_token is None
        assert event.lease_expires_at is None
        assert event.last_error is None

    exhausted = tasks[2]
    assert exhausted.id == UUID(int=3)
    assert exhausted.status is TaskStatus.FAILED
    assert exhausted.attempt_count == exhausted.max_attempts == 3
    assert exhausted.started_at == original_started_at[exhausted.id]
    assert exhausted.finished_at is not None
    assert before_recovery <= exhausted.finished_at <= after_recovery
    assert exhausted.result is None
    assert exhausted.error == _EXPECTED_ERROR
    assert exhausted.dispatch_token is None
    assert exhausted.execution_token is None
    assert exhausted.lease_expires_at is None


async def test_recovery_respects_batch_limit_and_returns_an_empty_result(
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    now = await _database_now(postgres_session_factory)
    async with postgres_session_factory.begin() as session:
        session.add_all(
            _expired_execution(
                task_number=task_number,
                lease_expires_at=now - timedelta(minutes=1),
                priority=TaskPriority.MEDIUM,
                attempt_count=1,
                max_attempts=1,
            )
            for task_number in (1, 2, 3)
        )

    def unexpected_retry(_attempt_count: int) -> timedelta:
        pytest.fail("An exhausted execution must not be retried")

    recovery = TaskExecutionRecovery(
        postgres_session_factory,
        batch_size=2,
        retry_delay_for_attempt=unexpected_retry,
    )

    assert await recovery.recover_once() == RecoveryBatchResult(locked=2, retried=0, failed=2)
    tasks, _events = await _read_rows(postgres_session_factory)
    assert [task.status for task in tasks] == [
        TaskStatus.FAILED,
        TaskStatus.FAILED,
        TaskStatus.IN_PROGRESS,
    ]
    assert await recovery.recover_once() == RecoveryBatchResult(locked=1, retried=0, failed=1)
    assert await recovery.recover_once() == RecoveryBatchResult(locked=0, retried=0, failed=0)


async def test_concurrent_recovery_passes_retry_an_execution_exactly_once(
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    now = await _database_now(postgres_session_factory)
    candidate = _expired_execution(
        task_number=1,
        lease_expires_at=now - timedelta(minutes=1),
        priority=TaskPriority.HIGH,
        attempt_count=1,
    )
    async with postgres_session_factory.begin() as session:
        session.add(candidate)

    recoveries = tuple(
        TaskExecutionRecovery(
            postgres_session_factory,
            batch_size=1,
            retry_delay_for_attempt=lambda _attempt_count: timedelta(seconds=5),
        )
        for _ in range(2)
    )
    results = await asyncio.gather(*(recovery.recover_once() for recovery in recoveries))

    assert sum(result.locked for result in results) == 1
    assert sum(result.retried for result in results) == 1
    assert sum(result.failed for result in results) == 0
    tasks, events = await _read_rows(postgres_session_factory)
    assert len(tasks) == len(events) == 1
    stored = tasks[0]
    assert stored.status is TaskStatus.PENDING
    assert stored.dispatch_token is not None
    assert stored.execution_token is None
    assert stored.lease_expires_at is None
    assert events[0].payload["dispatch_token"] == str(stored.dispatch_token)


async def test_outbox_failure_rolls_back_the_entire_mixed_recovery_batch(
    postgres_engine: AsyncEngine,
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    now = await _database_now(postgres_session_factory)
    candidates = (
        _expired_execution(
            task_number=1,
            lease_expires_at=now - timedelta(minutes=1),
            priority=TaskPriority.LOW,
            attempt_count=3,
        ),
        _expired_execution(
            task_number=2,
            lease_expires_at=now - timedelta(minutes=1),
            priority=TaskPriority.HIGH,
            attempt_count=1,
        ),
    )
    original_state = {
        candidate.id: (
            candidate.status,
            candidate.execution_token,
            candidate.lease_expires_at,
            candidate.started_at,
            candidate.attempt_count,
        )
        for candidate in candidates
    }
    async with postgres_session_factory.begin() as session:
        session.add_all(candidates)
    async with postgres_engine.begin() as connection:
        await connection.execute(
            text(
                "ALTER TABLE outbox_events ADD CONSTRAINT "
                "ck_test_reject_recovery_outbox CHECK (false)"
            )
        )

    try:
        recovery = TaskExecutionRecovery(
            postgres_session_factory,
            batch_size=2,
            retry_delay_for_attempt=lambda _attempt_count: timedelta(seconds=5),
        )
        with pytest.raises(IntegrityError, match="ck_test_reject_recovery_outbox"):
            await recovery.recover_once()
    finally:
        async with postgres_engine.begin() as connection:
            await connection.execute(
                text("ALTER TABLE outbox_events DROP CONSTRAINT ck_test_reject_recovery_outbox")
            )

    tasks, events = await _read_rows(postgres_session_factory)
    assert events == []
    for task in tasks:
        assert (
            task.status,
            task.execution_token,
            task.lease_expires_at,
            task.started_at,
            task.attempt_count,
        ) == original_state[task.id]
        assert task.dispatch_token is None
        assert task.finished_at is None
        assert task.result is None
        assert task.error is None
