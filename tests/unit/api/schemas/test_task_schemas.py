"""Contract tests for task API schemas."""

from datetime import UTC, datetime, timedelta, timezone
from uuid import uuid1, uuid4

import pytest
from pydantic import ValidationError

from cims_task_service.api.schemas.task import (
    CreateTaskRequest,
    TaskListResponse,
    TaskResponse,
    TaskStatusResponse,
)
from cims_task_service.domain.task import TaskPriority, TaskStatus


def _task_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": str(uuid4()),
        "name": "Generate statistics",
        "description": "Count words in the supplied text",
        "priority": "MEDIUM",
        "status": "PENDING",
        "created_at": "2026-07-31T00:00:00Z",
        "started_at": None,
        "finished_at": None,
        "result": None,
        "error": None,
    }
    payload.update(overrides)
    return payload


def test_create_task_request_accepts_the_public_contract() -> None:
    """Create input keeps meaningful whitespace and typed priority."""

    request = CreateTaskRequest.model_validate(
        {
            "name": "  Generate statistics  ",
            "description": "",
            "priority": "HIGH",
        }
    )

    assert request.name == "  Generate statistics  "
    assert request.description == ""
    assert request.priority is TaskPriority.HIGH
    assert request.model_dump(mode="json") == {
        "name": "  Generate statistics  ",
        "description": "",
        "priority": "HIGH",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "Task", "description": "Description"},
        {"name": "Task", "priority": "LOW"},
        {"name": "   ", "description": "Description", "priority": "LOW"},
        {"name": "Task", "description": "Description", "priority": "URGENT"},
        {
            "name": "Task",
            "description": "Description",
            "priority": "LOW",
            "unexpected": True,
        },
    ],
)
def test_create_task_request_rejects_invalid_input(payload: dict[str, object]) -> None:
    """Required fields, known enum values, and exact shape are enforced."""

    with pytest.raises(ValidationError):
        CreateTaskRequest.model_validate(payload)


def test_task_response_normalizes_utc_and_serializes_wire_types() -> None:
    """UUID, enum, timestamps, and JSON values have a stable wire form."""

    identifier = uuid4()
    local_timezone = timezone(timedelta(hours=10))
    created_at = datetime(2026, 7, 31, 12, 0, tzinfo=local_timezone)
    started_at = created_at + timedelta(seconds=5)
    finished_at = started_at + timedelta(seconds=5)
    response = TaskResponse.model_validate(
        _task_payload(
            id=str(identifier),
            priority="HIGH",
            status="COMPLETED",
            created_at=created_at.isoformat(),
            started_at=started_at.isoformat(),
            finished_at=finished_at.isoformat(),
            result={"counts": {"words": 4}, "labels": ["demo", None]},
        )
    )

    assert response.id == identifier
    assert response.priority is TaskPriority.HIGH
    assert response.status is TaskStatus.COMPLETED
    assert response.created_at == datetime(2026, 7, 31, 2, 0, tzinfo=UTC)
    assert response.started_at == datetime(2026, 7, 31, 2, 0, 5, tzinfo=UTC)

    payload = response.model_dump(mode="json")
    assert payload["id"] == str(identifier)
    assert payload["priority"] == "HIGH"
    assert payload["status"] == "COMPLETED"
    assert payload["created_at"] == "2026-07-31T02:00:00Z"
    assert payload["started_at"] == "2026-07-31T02:00:05Z"
    assert payload["finished_at"] == "2026-07-31T02:00:10Z"
    assert payload["error"] is None


@pytest.mark.parametrize("identifier", [str(uuid1()), "not-a-uuid"])
def test_task_response_requires_uuid_v4(identifier: str) -> None:
    """Identifiers with another UUID version or malformed text are rejected."""

    with pytest.raises(ValidationError):
        TaskResponse.model_validate(_task_payload(id=identifier))


def test_task_response_requires_timezone_aware_timestamps() -> None:
    """Naive timestamps cannot silently enter a UTC contract."""

    with pytest.raises(ValidationError):
        TaskResponse.model_validate(_task_payload(created_at="2026-07-31T00:00:00"))


@pytest.mark.parametrize(
    "timestamps",
    [
        {
            "started_at": "2026-07-30T23:59:59Z",
        },
        {
            "finished_at": "2026-07-30T23:59:59Z",
        },
        {
            "started_at": "2026-07-31T00:00:10Z",
            "finished_at": "2026-07-31T00:00:05Z",
        },
    ],
)
def test_task_response_rejects_invalid_timestamp_order(
    timestamps: dict[str, object],
) -> None:
    """Task timestamps cannot move backwards."""

    with pytest.raises(ValidationError):
        TaskResponse.model_validate(_task_payload(**timestamps))


def test_task_response_rejects_result_and_error_together() -> None:
    """A task cannot expose a successful result and an error simultaneously."""

    with pytest.raises(ValidationError, match="mutually exclusive"):
        TaskResponse.model_validate(
            _task_payload(
                result={"value": 42},
                error={"message": "processing failed"},
            )
        )


def test_task_response_accepts_error_without_result() -> None:
    """A failed task may expose structured error information."""

    response = TaskResponse.model_validate(
        _task_payload(
            status="FAILED",
            finished_at="2026-07-31T00:00:10Z",
            error={"code": "PROCESSING_FAILED", "retryable": False},
        )
    )

    assert response.result is None
    assert response.error == {
        "code": "PROCESSING_FAILED",
        "retryable": False,
    }


@pytest.mark.parametrize(
    "result",
    [
        "completed",
        [1, 2, 3],
        {"score": float("nan")},
        {"score": float("inf")},
        {"score": float("-inf")},
        {"created_at": datetime(2026, 7, 31, tzinfo=UTC)},
        {"id": uuid4()},
    ],
)
def test_task_response_requires_a_finite_json_result_object(result: object) -> None:
    """Result data must be a JSON object with finite numeric values."""

    with pytest.raises(ValidationError):
        TaskResponse.model_validate(_task_payload(result=result))


def test_json_schema_exposes_public_task_constraints() -> None:
    """Generated schemas describe required fields and structural rules."""

    create_schema = CreateTaskRequest.model_json_schema()
    task_schema = TaskResponse.model_json_schema()

    assert create_schema["additionalProperties"] is False
    assert create_schema["properties"]["name"] == {"$ref": "#/$defs/TaskName"}
    assert create_schema["$defs"]["TaskName"]["minLength"] == 1
    assert create_schema["$defs"]["TaskName"]["pattern"] == r"\S"
    assert task_schema["additionalProperties"] is False
    assert set(task_schema["required"]) == {
        "id",
        "name",
        "description",
        "priority",
        "status",
        "created_at",
        "started_at",
        "finished_at",
        "result",
        "error",
    }
    assert task_schema["oneOf"] == [
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


def test_status_and_list_responses_preserve_typed_values() -> None:
    """Dedicated status and paginated responses keep domain types."""

    task = TaskResponse.model_validate(_task_payload())
    status = TaskStatusResponse.model_validate(
        {
            "id": str(task.id),
            "status": "PENDING",
        }
    )
    task_list = TaskListResponse(
        items=[task],
        total=1,
        page=1,
        size=20,
    )

    assert status.id == task.id
    assert status.status is TaskStatus.PENDING
    assert status.model_dump(mode="json")["status"] == "PENDING"
    assert task_list.items == [task]
    assert task_list.total == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"total": -1},
        {"page": 0},
        {"size": 0},
        {"unexpected": True},
    ],
)
def test_task_list_response_rejects_invalid_pagination(
    overrides: dict[str, object],
) -> None:
    """Pagination metadata is positive and the response shape is exact."""

    payload: dict[str, object] = {
        "items": [],
        "total": 0,
        "page": 1,
        "size": 20,
    }
    payload.update(overrides)

    with pytest.raises(ValidationError):
        TaskListResponse.model_validate(payload)
