"""FastAPI dependencies backed by application-lifetime resources."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, cast
from uuid import UUID

from fastapi import Depends, Request

from cims_task_service.application.task_creation import (
    CreateTaskCommand,
    CreateTaskResult,
    create_task,
)
from cims_task_service.application.task_queries import (
    ListTasksQuery,
    ListTasksResult,
    get_task,
    get_task_status,
    list_tasks,
)
from cims_task_service.infrastructure.database.models import TaskModel
from cims_task_service.infrastructure.database.session import AsyncSessionFactory
from cims_task_service.infrastructure.database.task_repository import TaskStatusSnapshot

type TaskCreator = Callable[[CreateTaskCommand], Awaitable[CreateTaskResult]]
type TaskReader = Callable[[UUID], Awaitable[TaskModel]]
type TaskStatusReader = Callable[[UUID], Awaitable[TaskStatusSnapshot]]
type TaskListReader = Callable[[ListTasksQuery], Awaitable[ListTasksResult]]


@dataclass(frozen=True, slots=True)
class ApplicationResources:
    """Resources shared by requests during one application lifespan."""

    session_factory: AsyncSessionFactory
    task_max_attempts: int


def get_application_resources(request: Request) -> ApplicationResources:
    """Read initialized resources from the current application instance."""

    return cast(ApplicationResources, request.app.state.resources)


def get_task_creator(
    resources: Annotated[ApplicationResources, Depends(get_application_resources)],
) -> TaskCreator:
    """Bind the task creation use case to this application's resources."""

    async def create(command: CreateTaskCommand) -> CreateTaskResult:
        return await create_task(
            command,
            session_factory=resources.session_factory,
            max_attempts=resources.task_max_attempts,
        )

    return create


def get_task_reader(
    resources: Annotated[ApplicationResources, Depends(get_application_resources)],
) -> TaskReader:
    """Bind the task query use case to this application's resources."""

    async def read(task_id: UUID) -> TaskModel:
        return await get_task(
            task_id,
            session_factory=resources.session_factory,
        )

    return read


def get_task_status_reader(
    resources: Annotated[ApplicationResources, Depends(get_application_resources)],
) -> TaskStatusReader:
    """Bind the task status query to this application's resources."""

    async def read_status(task_id: UUID) -> TaskStatusSnapshot:
        return await get_task_status(
            task_id,
            session_factory=resources.session_factory,
        )

    return read_status


def get_task_list_reader(
    resources: Annotated[ApplicationResources, Depends(get_application_resources)],
) -> TaskListReader:
    """Bind the paginated task query to this application's resources."""

    async def read_list(query: ListTasksQuery) -> ListTasksResult:
        return await list_tasks(
            query,
            session_factory=resources.session_factory,
        )

    return read_list
