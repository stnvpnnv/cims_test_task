"""Task request and response contracts."""

from datetime import UTC, datetime
from typing import Annotated, Self

from pydantic import (
    UUID4,
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    model_validator,
)

from cims_task_service.application.task_queries import (
    DEFAULT_TASK_PAGE,
    DEFAULT_TASK_PAGE_SIZE,
    MAX_TASK_PAGE_SIZE,
)
from cims_task_service.domain.task import TaskPriority, TaskStatus

type JsonObject = dict[str, JsonValue]


def _normalize_to_utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


type TaskName = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
type UtcDatetime = Annotated[AwareDatetime, AfterValidator(_normalize_to_utc)]


class _ApiSchema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )


class CreateTaskRequest(_ApiSchema):
    """Input accepted when a client creates a task."""

    name: TaskName
    description: str
    priority: TaskPriority


class TaskResponse(_ApiSchema):
    """Complete task data without lifecycle transition rules."""

    model_config = ConfigDict(
        json_schema_extra={
            "oneOf": [
                {
                    "properties": {
                        "result": {"type": "object"},
                        "error": {"type": "null"},
                    }
                },
                {
                    "properties": {
                        "result": {"type": "null"},
                        "error": {"type": "object"},
                    }
                },
                {
                    "properties": {
                        "result": {"type": "null"},
                        "error": {"type": "null"},
                    }
                },
            ]
        }
    )

    id: UUID4
    name: TaskName
    description: str
    priority: TaskPriority
    status: TaskStatus
    created_at: UtcDatetime = Field(description="UTC time when the task was created.")
    started_at: UtcDatetime | None = Field(
        description="UTC time of the first transition to IN_PROGRESS."
    )
    finished_at: UtcDatetime | None = Field(
        description="UTC time of the terminal transition, including cancellation."
    )
    result: JsonObject | None = Field(
        description="Task result as a JSON object; mutually exclusive with error."
    )
    error: JsonObject | None = Field(
        description="Task error information as a JSON object; mutually exclusive with result."
    )

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        """Reject incompatible fields and timestamp ordering."""

        if self.result is not None and self.error is not None:
            raise ValueError("Task result and error are mutually exclusive")
        if self.started_at is not None and self.started_at < self.created_at:
            raise ValueError("Task start time cannot precede creation time")
        if self.finished_at is not None and self.finished_at < self.created_at:
            raise ValueError("Task finish time cannot precede creation time")
        if (
            self.started_at is not None
            and self.finished_at is not None
            and self.finished_at < self.started_at
        ):
            raise ValueError("Task finish time cannot precede start time")
        return self


class TaskStatusResponse(_ApiSchema):
    """Compact response for the dedicated task status endpoint."""

    id: UUID4
    status: TaskStatus


class TaskListParameters(_ApiSchema):
    """Filtering and pagination accepted by the task list endpoint."""

    status: TaskStatus | None = Field(default=None, description="Filter by exact task status.")
    priority: TaskPriority | None = Field(
        default=None,
        description="Filter by exact task priority.",
    )
    page: int = Field(
        default=DEFAULT_TASK_PAGE,
        ge=1,
        description="One-based page number.",
    )
    size: int = Field(
        default=DEFAULT_TASK_PAGE_SIZE,
        ge=1,
        le=MAX_TASK_PAGE_SIZE,
        description="Maximum number of tasks per page.",
    )


class TaskListResponse(_ApiSchema):
    """Paginated task collection."""

    items: list[TaskResponse]
    total: int = Field(ge=0, description="Total number of matching tasks.")
    page: int = Field(ge=1, description="One-based page number.")
    size: int = Field(ge=1, description="Requested page size.")
