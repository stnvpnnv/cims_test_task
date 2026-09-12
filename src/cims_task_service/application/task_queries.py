"""Application queries for persisted tasks."""

from dataclasses import dataclass
from typing import Final
from uuid import UUID

from cims_task_service.application.task_errors import TaskNotFoundError
from cims_task_service.domain.task import TaskPriority, TaskStatus
from cims_task_service.infrastructure.database.models import TaskModel
from cims_task_service.infrastructure.database.session import AsyncSessionFactory
from cims_task_service.infrastructure.database.task_repository import (
    StoredTaskPage,
    TaskRepository,
    TaskStatusSnapshot,
)

DEFAULT_TASK_PAGE: Final = 1
DEFAULT_TASK_PAGE_SIZE: Final = 20
MAX_TASK_PAGE_SIZE: Final = 100


@dataclass(frozen=True, slots=True)
class ListTasksQuery:
    """Validated filtering and offset-pagination options for task listing."""

    status: TaskStatus | None = None
    priority: TaskPriority | None = None
    page: int = DEFAULT_TASK_PAGE
    size: int = DEFAULT_TASK_PAGE_SIZE

    def __post_init__(self) -> None:
        if self.page < 1:
            message = "page must be at least 1"
            raise ValueError(message)
        if not 1 <= self.size <= MAX_TASK_PAGE_SIZE:
            message = f"size must be between 1 and {MAX_TASK_PAGE_SIZE}"
            raise ValueError(message)

    @property
    def offset(self) -> int:
        """Return the zero-based row offset for this one-based page."""

        return (self.page - 1) * self.size


@dataclass(frozen=True, slots=True)
class ListTasksResult:
    """One task page together with filtered collection metadata."""

    items: tuple[TaskModel, ...]
    total: int
    page: int
    size: int


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


async def list_tasks(
    query: ListTasksQuery,
    *,
    session_factory: AsyncSessionFactory,
) -> ListTasksResult:
    """Return one filtered task page from a consistent database snapshot."""

    async with session_factory() as session:
        await session.connection(execution_options={"isolation_level": "REPEATABLE READ"})
        stored_page: StoredTaskPage = await TaskRepository(session).list_page(
            status=query.status,
            priority=query.priority,
            offset=query.offset,
            limit=query.size,
        )
        result = ListTasksResult(
            items=stored_page.items,
            total=stored_page.total,
            page=query.page,
            size=query.size,
        )

    return result
