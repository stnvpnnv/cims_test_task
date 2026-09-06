"""FastAPI dependencies backed by application-lifetime resources."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, cast

from fastapi import Depends, Request

from cims_task_service.application.task_creation import (
    CreateTaskCommand,
    CreateTaskResult,
    create_task,
)
from cims_task_service.infrastructure.database.session import AsyncSessionFactory

type TaskCreator = Callable[[CreateTaskCommand], Awaitable[CreateTaskResult]]


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
