"""Persistence operations for task execution ownership."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from cims_task_service.domain.task import TaskPriority, TaskStatus
from cims_task_service.infrastructure.database.models import (
    JsonObject,
    OutboxEventModel,
    TaskModel,
)


@dataclass(frozen=True, slots=True)
class ClaimedTaskExecution:
    """Detached execution input protected by a unique fencing token."""

    task_id: UUID
    name: str
    description: str
    priority: TaskPriority
    attempt_count: int
    max_attempts: int
    execution_token: UUID
    lease_expires_at: datetime


@dataclass(frozen=True, slots=True)
class LockedExpiredTaskExecution:
    """Expired execution whose task row stays locked by the caller's transaction."""

    task_id: UUID
    priority: TaskPriority
    attempt_count: int
    max_attempts: int
    execution_token: UUID
    lease_expires_at: datetime


class TaskExecutionRepository:
    """Manage task execution ownership without owning the transaction."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def claim_for_execution(
        self,
        task_id: UUID,
        *,
        dispatch_token: UUID,
        lease_duration: timedelta,
    ) -> ClaimedTaskExecution | None:
        """Atomically start one pending task for its current dispatch token."""

        if lease_duration <= timedelta(0):
            raise ValueError("lease_duration must be positive")

        execution_token = uuid4()
        statement = (
            update(TaskModel)
            .where(
                TaskModel.id == task_id,
                TaskModel.status == TaskStatus.PENDING,
                TaskModel.dispatch_token == dispatch_token,
                TaskModel.attempt_count < TaskModel.max_attempts,
            )
            .values(
                status=TaskStatus.IN_PROGRESS,
                started_at=func.coalesce(TaskModel.started_at, func.clock_timestamp()),
                attempt_count=TaskModel.attempt_count + 1,
                dispatch_token=None,
                execution_token=execution_token,
                lease_expires_at=func.clock_timestamp() + lease_duration,
            )
            .returning(
                TaskModel.id,
                TaskModel.name,
                TaskModel.description,
                TaskModel.priority,
                TaskModel.attempt_count,
                TaskModel.max_attempts,
                TaskModel.execution_token,
                TaskModel.lease_expires_at,
            )
        )
        row = (await self._session.execute(statement)).one_or_none()
        if row is None:
            return None

        (
            stored_task_id,
            name,
            description,
            priority,
            attempt_count,
            max_attempts,
            stored_execution_token,
            lease_expires_at,
        ) = row
        if stored_execution_token is None or lease_expires_at is None:
            raise RuntimeError("claimed task execution is missing its lease")

        return ClaimedTaskExecution(
            task_id=stored_task_id,
            name=name,
            description=description,
            priority=priority,
            attempt_count=attempt_count,
            max_attempts=max_attempts,
            execution_token=stored_execution_token,
            lease_expires_at=lease_expires_at,
        )

    async def lock_expired_execution_batch(
        self,
        *,
        batch_size: int,
    ) -> tuple[LockedExpiredTaskExecution, ...]:
        """Lock the oldest expired executions for processing in the same transaction."""

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        statement = (
            select(
                TaskModel.id,
                TaskModel.priority,
                TaskModel.attempt_count,
                TaskModel.max_attempts,
                TaskModel.execution_token,
                TaskModel.lease_expires_at,
            )
            .where(
                TaskModel.status == TaskStatus.IN_PROGRESS,
                TaskModel.lease_expires_at <= func.statement_timestamp(),
            )
            .order_by(TaskModel.lease_expires_at, TaskModel.id)
            .limit(batch_size)
            .with_for_update(of=TaskModel, skip_locked=True)
        )
        rows = (await self._session.execute(statement)).all()

        locked_executions: list[LockedExpiredTaskExecution] = []
        for row in rows:
            (
                task_id,
                priority,
                attempt_count,
                max_attempts,
                execution_token,
                lease_expires_at,
            ) = row
            if execution_token is None or lease_expires_at is None:
                raise RuntimeError("expired task execution is missing its lease")

            locked_executions.append(
                LockedExpiredTaskExecution(
                    task_id=task_id,
                    priority=priority,
                    attempt_count=attempt_count,
                    max_attempts=max_attempts,
                    execution_token=execution_token,
                    lease_expires_at=lease_expires_at,
                )
            )

        return tuple(locked_executions)

    async def renew_execution_lease(
        self,
        task_id: UUID,
        *,
        execution_token: UUID,
        lease_duration: timedelta,
    ) -> bool:
        """Extend the current owner's lease; expiry alone does not revoke ownership."""

        if lease_duration <= timedelta(0):
            raise ValueError("lease_duration must be positive")

        statement = (
            update(TaskModel)
            .where(
                TaskModel.id == task_id,
                TaskModel.status == TaskStatus.IN_PROGRESS,
                TaskModel.execution_token == execution_token,
            )
            .values(
                lease_expires_at=func.greatest(
                    TaskModel.lease_expires_at,
                    func.clock_timestamp() + lease_duration,
                ),
            )
            .returning(TaskModel.id)
        )
        return await self._session.scalar(statement) is not None

    async def schedule_execution_retry(
        self,
        task_id: UUID,
        *,
        execution_token: UUID,
        retry_delay: timedelta,
        event_type: str,
        message_priority: int,
    ) -> bool:
        """Release the current execution and enqueue its retry in the caller's transaction."""

        if retry_delay < timedelta(0):
            raise ValueError("retry_delay must not be negative")

        dispatch_token = uuid4()
        statement = (
            update(TaskModel)
            .where(
                TaskModel.id == task_id,
                TaskModel.status == TaskStatus.IN_PROGRESS,
                TaskModel.execution_token == execution_token,
                TaskModel.attempt_count < TaskModel.max_attempts,
            )
            .values(
                status=TaskStatus.PENDING,
                dispatch_token=dispatch_token,
                execution_token=None,
                lease_expires_at=None,
            )
            .returning(TaskModel.id)
        )
        if await self._session.scalar(statement) is None:
            return False

        outbox_event = OutboxEventModel(
            id=uuid4(),
            task_id=task_id,
            event_type=event_type,
            payload={"task_id": str(task_id), "dispatch_token": str(dispatch_token)},
            message_priority=message_priority,
            available_at=func.clock_timestamp() + retry_delay,
            published_at=None,
            discarded_at=None,
            publish_attempts=0,
            publisher_token=None,
            lease_expires_at=None,
            last_error=None,
        )
        self._session.add(outbox_event)
        await self._session.flush()
        return True

    async def complete_execution(
        self,
        task_id: UUID,
        *,
        execution_token: UUID,
        result: JsonObject,
    ) -> bool:
        """Persist success only while the caller still owns the execution."""

        statement = (
            update(TaskModel)
            .where(
                TaskModel.id == task_id,
                TaskModel.status == TaskStatus.IN_PROGRESS,
                TaskModel.execution_token == execution_token,
            )
            .values(
                status=TaskStatus.COMPLETED,
                finished_at=func.clock_timestamp(),
                result=result,
                error=None,
                dispatch_token=None,
                execution_token=None,
                lease_expires_at=None,
            )
            .returning(TaskModel.id)
        )
        return await self._session.scalar(statement) is not None

    async def fail_execution(
        self,
        task_id: UUID,
        *,
        execution_token: UUID,
        error: JsonObject,
    ) -> bool:
        """Persist a terminal error only for the current execution owner."""

        statement = (
            update(TaskModel)
            .where(
                TaskModel.id == task_id,
                TaskModel.status == TaskStatus.IN_PROGRESS,
                TaskModel.execution_token == execution_token,
            )
            .values(
                status=TaskStatus.FAILED,
                finished_at=func.clock_timestamp(),
                result=None,
                error=error,
                dispatch_token=None,
                execution_token=None,
                lease_expires_at=None,
            )
            .returning(TaskModel.id)
        )
        return await self._session.scalar(statement) is not None
