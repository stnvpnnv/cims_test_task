"""Transactional task cancellation application service."""

from uuid import UUID

from cims_task_service.application.task_errors import (
    TaskNotCancellableError,
    TaskNotFoundError,
)
from cims_task_service.domain.task import TaskStatus
from cims_task_service.domain.task_lifecycle import is_terminal
from cims_task_service.infrastructure.database.models import TaskModel
from cims_task_service.infrastructure.database.session import AsyncSessionFactory
from cims_task_service.infrastructure.database.task_repository import TaskRepository
from cims_task_service.infrastructure.messaging.topology import TASK_ROUTING_KEY


async def cancel_task(
    task_id: UUID,
    *,
    session_factory: AsyncSessionFactory,
) -> TaskModel:
    """Cancel an active task or return its existing cancellation idempotently."""

    async with session_factory.begin() as session:
        stored = await TaskRepository(session).cancel_with_outbox(
            task_id,
            event_type=TASK_ROUTING_KEY,
        )
        if stored is None:
            raise TaskNotFoundError(task_id)

        task = stored.task
        if task.status is TaskStatus.CANCELLED:
            result = task
        elif is_terminal(task.status):
            raise TaskNotCancellableError(task_id, task.status)
        else:
            message = "task cancellation compare-and-set returned an active task"
            raise RuntimeError(message)

    return result
