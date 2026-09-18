"""Persistence operations for task execution ownership."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import func, update
from sqlalchemy.ext.asyncio import AsyncSession

from cims_task_service.domain.task import TaskPriority, TaskStatus
from cims_task_service.infrastructure.database.models import TaskModel


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


class TaskExecutionRepository:
    """Acquire task execution ownership without owning the transaction."""

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
