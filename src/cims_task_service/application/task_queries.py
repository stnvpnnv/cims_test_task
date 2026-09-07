"""Application queries for persisted tasks."""

from uuid import UUID

from cims_task_service.infrastructure.database.models import TaskModel
from cims_task_service.infrastructure.database.session import AsyncSessionFactory
from cims_task_service.infrastructure.database.task_repository import (
    TaskRepository,
    TaskStatusSnapshot,
)


class TaskNotFoundError(LookupError):
    """Raised when a requested task does not exist."""

    def __init__(self, task_id: UUID) -> None:
        self.task_id = task_id
        super().__init__(f"Task {task_id} was not found")


async def get_task(
    task_id: UUID,
    *,
    session_factory: AsyncSessionFactory,
) -> TaskModel:
    """Return one task while owning the read session lifecycle."""

    async with session_factory() as session:
        task = await TaskRepository(session).get_by_id(task_id)
        if task is None:
            raise TaskNotFoundError(task_id)

    return task


async def get_task_status(
    task_id: UUID,
    *,
    session_factory: AsyncSessionFactory,
) -> TaskStatusSnapshot:
    """Return one task status while owning the read session lifecycle."""

    async with session_factory() as session:
        snapshot = await TaskRepository(session).get_status_by_id(task_id)
        if snapshot is None:
            raise TaskNotFoundError(task_id)

    return snapshot
