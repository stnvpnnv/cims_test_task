"""HTTP contract tests for task creation."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cims_task_service.api import dependencies as api_dependencies
from cims_task_service.application.task_creation import (
    CreateTaskCommand,
    CreateTaskResult,
    IdempotencyKeyConflictError,
)
from cims_task_service.config import Settings
from cims_task_service.domain.task import TaskPriority, TaskStatus
from cims_task_service.infrastructure.database.models import TaskModel
from cims_task_service.main import create_app


def _task(*, completed: bool = False) -> TaskModel:
    identifier = UUID("7683d9ce-4218-48d4-a41f-bb05df8e806f")
    created_at = datetime(2026, 9, 6, 1, 2, 3, tzinfo=UTC)
    started_at = created_at + timedelta(seconds=1) if completed else None
    finished_at = created_at + timedelta(seconds=2) if completed else None
    return TaskModel(
        id=identifier,
        name="Daily report",
        description="Aggregate yesterday's records",
        priority=TaskPriority.HIGH,
        status=TaskStatus.COMPLETED if completed else TaskStatus.NEW,
        created_at=created_at,
        idempotency_key_hash=b"k" * 32,
        request_fingerprint=b"f" * 32,
        started_at=started_at,
        finished_at=finished_at,
        result={"records": 42} if completed else None,
        error=None,
        attempt_count=1 if completed else 0,
        max_attempts=5,
        dispatch_token=None if completed else uuid4(),
        execution_token=None,
        lease_expires_at=None,
    )


def _request_body() -> dict[str, str]:
    return {
        "name": "Daily report",
        "description": "Aggregate yesterday's records",
        "priority": "HIGH",
    }


def _application_with_task_creator(task_creator: AsyncMock) -> FastAPI:
    application = create_app(Settings())
    application.dependency_overrides[api_dependencies.get_task_creator] = lambda: task_creator
    return application


def test_create_task_returns_201_location_and_public_resource(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new keyed request forwards the opaque key and hides persistence metadata."""

    task = _task()
    creator = AsyncMock(return_value=CreateTaskResult(task=task, created=True))
    monkeypatch.setattr(api_dependencies, "create_task", creator)
    application = create_app(Settings(task_max_attempts=5))
    idempotency_key = "8e03978e-40d5-43e8-bc93-6894a57f9324"

    with TestClient(application) as client:
        response = client.post(
            "/api/v1/tasks",
            json=_request_body(),
            headers={"Idempotency-Key": idempotency_key},
        )
        session_factory = application.state.resources.session_factory

    assert response.status_code == 201
    assert response.headers["location"] == f"/api/v1/tasks/{task.id}"
    assert response.json() == {
        "id": str(task.id),
        "name": task.name,
        "description": task.description,
        "priority": "HIGH",
        "status": "NEW",
        "created_at": "2026-09-06T01:02:03Z",
        "started_at": None,
        "finished_at": None,
        "result": None,
        "error": None,
    }
    creator.assert_awaited_once()
    call = creator.await_args
    assert call is not None
    assert call.args == (
        CreateTaskCommand(
            name=task.name,
            description=task.description,
            priority=TaskPriority.HIGH,
            idempotency_key=idempotency_key,
        ),
    )
    assert call.kwargs["max_attempts"] == 5
    assert call.kwargs["session_factory"] is session_factory
    assert idempotency_key not in response.text
    assert "idempotency_key_hash" not in response.text
    assert "request_fingerprint" not in response.text


def test_create_task_without_key_remains_non_idempotent() -> None:
    """The PDF-compatible request works when the extension header is absent."""

    task = _task()
    creator = AsyncMock(return_value=CreateTaskResult(task=task, created=True))
    application = _application_with_task_creator(creator)

    with TestClient(application) as client:
        response = client.post("/api/v1/tasks", json=_request_body())

    assert response.status_code == 201
    call = creator.await_args
    assert call is not None
    command = call.args[0]
    assert command.idempotency_key is None


def test_matching_replay_returns_200_and_current_task_state() -> None:
    """A retry receives the resource's current representation and stable location."""

    task = _task(completed=True)
    creator = AsyncMock(return_value=CreateTaskResult(task=task, created=False))
    application = _application_with_task_creator(creator)

    with TestClient(application) as client:
        response = client.post(
            "/api/v1/tasks",
            json=_request_body(),
            headers={"Idempotency-Key": "case-sensitive-key"},
        )

    assert response.status_code == 200
    assert response.headers["location"] == f"/api/v1/tasks/{task.id}"
    assert response.json()["status"] == "COMPLETED"
    assert response.json()["result"] == {"records": 42}
    assert response.json()["started_at"] == "2026-09-06T01:02:04Z"
    assert response.json()["finished_at"] == "2026-09-06T01:02:05Z"


def test_reused_key_with_another_body_returns_problem_details() -> None:
    """A semantic key conflict uses Problem Details without disclosing stored data."""

    existing_task_id = uuid4()
    creator = AsyncMock(side_effect=IdempotencyKeyConflictError(existing_task_id))
    application = _application_with_task_creator(creator)
    idempotency_key = "conflicting-client-key"

    with TestClient(application) as client:
        response = client.post(
            "/api/v1/tasks",
            json=_request_body(),
            headers={"Idempotency-Key": idempotency_key},
        )

    assert response.status_code == 422
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json() == {
        "type": "urn:cims-task-service:problem:idempotency-key-reused",
        "title": "Idempotency-Key is already used",
        "status": 422,
        "detail": "The Idempotency-Key header was already used with a different request body.",
    }
    assert idempotency_key not in response.text
    assert str(existing_task_id) not in response.text


@pytest.mark.parametrize(
    ("body", "headers"),
    [
        ({"name": "Task", "description": "Missing priority"}, {}),
        (_request_body(), {"Idempotency-Key": ""}),
        (_request_body(), {"Idempotency-Key": "contains whitespace"}),
        (_request_body(), {"Idempotency-Key": "x" * 256}),
        (
            _request_body(),
            [("Idempotency-Key", "first-key"), ("Idempotency-Key", "second-key")],
        ),
    ],
)
def test_invalid_request_is_rejected_before_task_creation(
    body: dict[str, str],
    headers: dict[str, str] | list[tuple[str, str]],
) -> None:
    """Body and header validation retain FastAPI's JSON error contract."""

    creator = AsyncMock()
    application = _application_with_task_creator(creator)

    with TestClient(application) as client:
        response = client.post("/api/v1/tasks", json=body, headers=headers)

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")
    assert isinstance(response.json()["detail"], list)
    creator.assert_not_awaited()


def test_openapi_documents_the_complete_task_creation_contract() -> None:
    """The canonical PDF route exposes both success and 422 response formats."""

    with TestClient(create_app(Settings())) as client:
        schema = client.get("/openapi.json").json()

    assert "/api/v1/tasks" in schema["paths"]
    assert "/tasks" not in schema["paths"]
    operation = schema["paths"]["/api/v1/tasks"]["post"]
    header = next(
        parameter for parameter in operation["parameters"] if parameter["name"] == "Idempotency-Key"
    )
    assert header["in"] == "header"
    assert header["required"] is False
    header_schema = header["schema"]
    string_schema = next(
        option for option in header_schema["anyOf"] if option.get("type") == "string"
    )
    assert string_schema["minLength"] == 1
    assert string_schema["maxLength"] == 255
    assert string_schema["pattern"] == r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$"
    assert set(operation["responses"]) >= {"200", "201", "422"}
    assert "Location" in operation["responses"]["200"]["headers"]
    assert "Location" in operation["responses"]["201"]["headers"]
    error_content = operation["responses"]["422"]["content"]
    assert set(error_content) == {"application/json", "application/problem+json"}
    assert error_content["application/json"]["schema"] == {
        "$ref": "#/components/schemas/HTTPValidationError"
    }
    problem_schema = error_content["application/problem+json"]["schema"]
    assert set(problem_schema["required"]) == {"type", "title", "status", "detail"}
