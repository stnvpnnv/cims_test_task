"""HTTP API contract tests for task cancellation."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, call
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cims_task_service.api import dependencies as api_dependencies
from cims_task_service.application.task_errors import (
    TaskNotCancellableError,
    TaskNotFoundError,
)
from cims_task_service.config import Settings
from cims_task_service.domain.task import TaskPriority, TaskStatus
from cims_task_service.infrastructure.database.models import TaskModel
from cims_task_service.main import create_app

_TASK_ID = UUID("7683d9ce-4218-48d4-a41f-bb05df8e806f")
_UUID_V1 = UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
_CREATED_AT = datetime(2026, 9, 6, 1, 2, 3, tzinfo=UTC)
_STARTED_AT = _CREATED_AT + timedelta(seconds=1)
_FINISHED_AT = _CREATED_AT + timedelta(seconds=2)


def _cancelled_task() -> TaskModel:
    return TaskModel(
        id=_TASK_ID,
        name="Daily report",
        description="Aggregate yesterday's records",
        priority=TaskPriority.HIGH,
        status=TaskStatus.CANCELLED,
        created_at=_CREATED_AT,
        idempotency_key_hash=b"k" * 32,
        request_fingerprint=b"f" * 32,
        started_at=_STARTED_AT,
        finished_at=_FINISHED_AT,
        result=None,
        error=None,
        attempt_count=1,
        max_attempts=5,
        dispatch_token=None,
        execution_token=None,
        lease_expires_at=None,
    )


def _application_with_task_canceller(task_canceller: AsyncMock) -> FastAPI:
    application = create_app(Settings())
    application.dependency_overrides[api_dependencies.get_task_canceller] = lambda: task_canceller
    return application


def test_cancel_task_returns_the_complete_current_public_resource() -> None:
    """A successful cancellation returns its terminal representation."""

    task = _cancelled_task()
    canceller = AsyncMock(return_value=task)
    application = _application_with_task_canceller(canceller)

    with TestClient(application) as client:
        response = client.delete(f"/api/v1/tasks/{task.id}")

    assert response.status_code == 200
    assert response.json() == {
        "id": str(task.id),
        "name": task.name,
        "description": task.description,
        "priority": "HIGH",
        "status": "CANCELLED",
        "created_at": "2026-09-06T01:02:03Z",
        "started_at": "2026-09-06T01:02:04Z",
        "finished_at": "2026-09-06T01:02:05Z",
        "result": None,
        "error": None,
    }
    canceller.assert_awaited_once_with(task.id)
    assert "idempotency_key_hash" not in response.text
    assert "request_fingerprint" not in response.text
    assert "attempt_count" not in response.text
    assert "execution_token" not in response.text


def test_repeated_cancellation_returns_the_same_terminal_representation() -> None:
    """A replay remains successful and preserves the original finish time."""

    task = _cancelled_task()
    canceller = AsyncMock(return_value=task)
    application = _application_with_task_canceller(canceller)

    with TestClient(application) as client:
        first_response = client.delete(f"/api/v1/tasks/{task.id}")
        second_response = client.delete(f"/api/v1/tasks/{task.id}")

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert second_response.json() == first_response.json()
    assert second_response.json()["finished_at"] == "2026-09-06T01:02:05Z"
    assert canceller.await_args_list == [call(task.id), call(task.id)]


def test_task_cancellation_dependency_binds_the_application_session_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production dependency forwards the UUID and lifespan resource."""

    task = _cancelled_task()
    cancel_task = AsyncMock(return_value=task)
    monkeypatch.setattr(api_dependencies, "cancel_task", cancel_task)
    application = create_app(Settings())

    with TestClient(application) as client:
        response = client.delete(f"/api/v1/tasks/{task.id}")
        session_factory = application.state.resources.session_factory

    assert response.status_code == 200
    cancel_task.assert_awaited_once_with(
        task.id,
        session_factory=session_factory,
    )


def test_cancel_unknown_task_returns_problem_details_without_identifier() -> None:
    """A missing task has the shared semantic error without identifier leakage."""

    canceller = AsyncMock(side_effect=TaskNotFoundError(_TASK_ID))
    application = _application_with_task_canceller(canceller)

    with TestClient(application) as client:
        response = client.delete(f"/api/v1/tasks/{_TASK_ID}")

    assert response.status_code == 404
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json() == {
        "type": "urn:cims-task-service:problem:task-not-found",
        "title": "Task not found",
        "status": 404,
        "detail": "The requested task does not exist.",
    }
    canceller.assert_awaited_once_with(_TASK_ID)
    assert str(_TASK_ID) not in response.text


@pytest.mark.parametrize("current_status", [TaskStatus.COMPLETED, TaskStatus.FAILED])
def test_cancel_finished_task_returns_conflict(current_status: TaskStatus) -> None:
    """Completed and failed outcomes cannot be replaced with cancellation."""

    canceller = AsyncMock(
        side_effect=TaskNotCancellableError(_TASK_ID, current_status),
    )
    application = _application_with_task_canceller(canceller)

    with TestClient(application) as client:
        response = client.delete(f"/api/v1/tasks/{_TASK_ID}")

    assert response.status_code == 409
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json() == {
        "type": "urn:cims-task-service:problem:task-not-cancellable",
        "title": "Task cannot be cancelled",
        "status": 409,
        "detail": "A completed or failed task cannot be cancelled.",
    }
    canceller.assert_awaited_once_with(_TASK_ID)
    assert str(_TASK_ID) not in response.text
    assert current_status.value not in response.text


@pytest.mark.parametrize("task_id", ["not-a-uuid", str(_UUID_V1)])
def test_cancel_task_rejects_invalid_uuid_before_cancellation(task_id: str) -> None:
    """Malformed and non-v4 identifiers retain FastAPI's validation response."""

    canceller = AsyncMock()
    application = _application_with_task_canceller(canceller)

    with TestClient(application) as client:
        response = client.delete(f"/api/v1/tasks/{task_id}")

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")
    assert isinstance(response.json()["detail"], list)
    canceller.assert_not_awaited()


def test_openapi_documents_the_complete_task_cancellation_contract() -> None:
    """Cancellation documents its input, response, and error media types."""

    with TestClient(create_app(Settings())) as client:
        schema = client.get("/openapi.json").json()

    operation = schema["paths"]["/api/v1/tasks/{task_id}"]["delete"]
    assert "requestBody" not in operation
    assert [parameter["name"] for parameter in operation["parameters"]] == ["task_id"]
    task_id_parameter = operation["parameters"][0]
    assert task_id_parameter["in"] == "path"
    assert task_id_parameter["required"] is True
    assert task_id_parameter["schema"]["format"] == "uuid4"
    assert set(operation["responses"]) == {"200", "404", "409", "422"}
    assert operation["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/TaskResponse"
    }
    for status_code in ("404", "409"):
        problem_content = operation["responses"][status_code]["content"]
        assert set(problem_content) == {"application/problem+json"}
        assert set(problem_content["application/problem+json"]["schema"]["required"]) == {
            "type",
            "title",
            "status",
            "detail",
        }
    assert operation["responses"]["422"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/HTTPValidationError"
    }
