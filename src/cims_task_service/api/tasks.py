"""HTTP operations for task resources."""

from typing import Annotated, Final

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import UUID4

from cims_task_service.api.dependencies import (
    TaskCreator,
    TaskListReader,
    TaskReader,
    TaskStatusReader,
    get_task_creator,
    get_task_list_reader,
    get_task_reader,
    get_task_status_reader,
)
from cims_task_service.api.schemas.problem import ProblemDetails
from cims_task_service.api.schemas.task import (
    CreateTaskRequest,
    TaskListParameters,
    TaskListResponse,
    TaskResponse,
    TaskStatusResponse,
)
from cims_task_service.application.task_creation import (
    CreateTaskCommand,
    IdempotencyKeyConflictError,
)
from cims_task_service.application.task_queries import ListTasksQuery, TaskNotFoundError

router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])

_IDEMPOTENCY_KEY_PATTERN: Final = r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$"
_IDEMPOTENCY_CONFLICT = ProblemDetails(
    type="urn:cims-task-service:problem:idempotency-key-reused",
    title="Idempotency-Key is already used",
    status=status.HTTP_422_UNPROCESSABLE_CONTENT,
    detail="The Idempotency-Key header was already used with a different request body.",
)
_TASK_NOT_FOUND = ProblemDetails(
    type="urn:cims-task-service:problem:task-not-found",
    title="Task not found",
    status=status.HTTP_404_NOT_FOUND,
    detail="The requested task does not exist.",
)
_LOCATION_HEADER: Final[dict[str, object]] = {
    "description": "Relative URI of the created or replayed task resource.",
    "schema": {"type": "string", "example": "/api/v1/tasks/<task-id>"},
}


def _get_idempotency_key(
    request: Request,
    idempotency_key: Annotated[
        str | None,
        Header(
            alias="Idempotency-Key",
            min_length=1,
            max_length=255,
            pattern=_IDEMPOTENCY_KEY_PATTERN,
            description=(
                "Optional case-sensitive retry key using 1-255 HTTP token characters; "
                "a random UUID is recommended. Send this header at most once."
            ),
        ),
    ] = None,
) -> str | None:
    """Validate that the optional key is represented by one header field."""

    if len(request.headers.getlist("idempotency-key")) > 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=[
                {
                    "type": "multiple_header_values",
                    "loc": ["header", "idempotency-key"],
                    "msg": "Idempotency-Key must be sent at most once",
                }
            ],
        )
    return idempotency_key


def _problem_response(problem: ProblemDetails) -> JSONResponse:
    """Serialize stable Problem Details without exposing internal state."""

    return JSONResponse(
        status_code=problem.status,
        content=problem.model_dump(mode="json"),
        media_type="application/problem+json",
    )


@router.post(
    "",
    name="create_task",
    response_model=TaskResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an asynchronous task",
    responses={
        status.HTTP_200_OK: {
            "model": TaskResponse,
            "description": "The matching idempotent request already created this task.",
            "headers": {"Location": _LOCATION_HEADER},
        },
        status.HTTP_201_CREATED: {
            "description": "The task and its publication event were stored atomically.",
            "headers": {"Location": _LOCATION_HEADER},
        },
    },
    openapi_extra={
        "responses": {
            "422": {
                "description": "Request validation error or reuse of a key with another body.",
                "content": {
                    "application/problem+json": {
                        "schema": ProblemDetails.model_json_schema(),
                    }
                },
            }
        }
    },
)
async def create_task(
    payload: CreateTaskRequest,
    response: Response,
    task_creator: Annotated[TaskCreator, Depends(get_task_creator)],
    idempotency_key: Annotated[str | None, Depends(_get_idempotency_key)],
) -> TaskResponse | JSONResponse:
    """Persist a new task or return the resource created by a matching retry."""

    command = CreateTaskCommand(
        name=payload.name,
        description=payload.description,
        priority=payload.priority,
        idempotency_key=idempotency_key,
    )
    try:
        result = await task_creator(command)
    except IdempotencyKeyConflictError:
        return _problem_response(_IDEMPOTENCY_CONFLICT)

    response.status_code = status.HTTP_201_CREATED if result.created else status.HTTP_200_OK
    response.headers["Location"] = f"/api/v1/tasks/{result.task.id}"
    return TaskResponse.model_validate(result.task, from_attributes=True)


@router.get(
    "",
    name="list_tasks",
    response_model=TaskListResponse,
    summary="List tasks",
    description=(
        "Status and priority filters are combined with AND. "
        "Results are ordered by created_at DESC, then id DESC."
    ),
)
async def list_tasks(
    parameters: Annotated[TaskListParameters, Query()],
    task_list_reader: Annotated[TaskListReader, Depends(get_task_list_reader)],
) -> TaskListResponse:
    """Return a filtered, deterministically ordered page of tasks."""

    result = await task_list_reader(
        ListTasksQuery(
            status=parameters.status,
            priority=parameters.priority,
            page=parameters.page,
            size=parameters.size,
        )
    )
    return TaskListResponse(
        items=[TaskResponse.model_validate(item, from_attributes=True) for item in result.items],
        total=result.total,
        page=result.page,
        size=result.size,
    )


@router.get(
    "/{task_id}",
    name="get_task",
    response_model=TaskResponse,
    summary="Get a task",
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "No task exists with the supplied identifier.",
            "content": {
                "application/problem+json": {
                    "schema": ProblemDetails.model_json_schema(),
                }
            },
        }
    },
)
async def read_task(
    task_id: UUID4,
    task_reader: Annotated[TaskReader, Depends(get_task_reader)],
) -> TaskResponse | JSONResponse:
    """Return the current representation of one task."""

    try:
        task = await task_reader(task_id)
    except TaskNotFoundError:
        return _problem_response(_TASK_NOT_FOUND)

    return TaskResponse.model_validate(task, from_attributes=True)


@router.get(
    "/{task_id}/status",
    name="get_task_status",
    response_model=TaskStatusResponse,
    summary="Get a task status",
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "No task exists with the supplied identifier.",
            "content": {
                "application/problem+json": {
                    "schema": ProblemDetails.model_json_schema(),
                }
            },
        }
    },
)
async def read_task_status(
    task_id: UUID4,
    task_status_reader: Annotated[TaskStatusReader, Depends(get_task_status_reader)],
) -> TaskStatusResponse | JSONResponse:
    """Return the current status of one task."""

    try:
        snapshot = await task_status_reader(task_id)
    except TaskNotFoundError:
        return _problem_response(_TASK_NOT_FOUND)

    return TaskStatusResponse.model_validate(snapshot, from_attributes=True)
