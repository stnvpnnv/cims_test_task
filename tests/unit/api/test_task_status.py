"""HTTP contract tests for reading a task status."""

from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cims_task_service.api import dependencies as api_dependencies
from cims_task_service.application.task_errors import TaskNotFoundError
from cims_task_service.config import Settings
from cims_task_service.domain.task import TaskStatus
from cims_task_service.infrastructure.database.task_repository import TaskStatusSnapshot
from cims_task_service.main import create_app

_TASK_ID = UUID("7683d9ce-4218-48d4-a41f-bb05df8e806f")
_UUID_V1 = UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def _application_with_status_reader(status_reader: AsyncMock) -> FastAPI:
    application = create_app(Settings())
    application.dependency_overrides[api_dependencies.get_task_status_reader] = lambda: (
        status_reader
    )
    return application


def test_get_task_status_returns_the_exact_current_snapshot() -> None:
    """A status read exposes only the task identifier and its current status."""

    snapshot = TaskStatusSnapshot(id=_TASK_ID, status=TaskStatus.IN_PROGRESS)
    status_reader = AsyncMock(return_value=snapshot)
    application = _application_with_status_reader(status_reader)

    with TestClient(application) as client:
        response = client.get(f"/api/v1/tasks/{_TASK_ID}/status")

    assert response.status_code == 200
    assert response.json() == {
        "id": str(_TASK_ID),
        "status": "IN_PROGRESS",
    }
    status_reader.assert_awaited_once_with(_TASK_ID)


def test_task_status_dependency_binds_the_application_session_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production dependency forwards the parsed UUID and lifespan resource."""

    snapshot = TaskStatusSnapshot(id=_TASK_ID, status=TaskStatus.PENDING)
    get_task_status = AsyncMock(return_value=snapshot)
    monkeypatch.setattr(api_dependencies, "get_task_status", get_task_status)
    application = create_app(Settings())

    with TestClient(application) as client:
        response = client.get(f"/api/v1/tasks/{_TASK_ID}/status")
        session_factory = application.state.resources.session_factory

    assert response.status_code == 200
    get_task_status.assert_awaited_once_with(
        _TASK_ID,
        session_factory=session_factory,
    )


def test_get_unknown_task_status_returns_problem_details_without_identifier() -> None:
    """A missing task has the shared semantic error without echoing its identifier."""

    status_reader = AsyncMock(side_effect=TaskNotFoundError(_TASK_ID))
    application = _application_with_status_reader(status_reader)

    with TestClient(application) as client:
        response = client.get(f"/api/v1/tasks/{_TASK_ID}/status")

    assert response.status_code == 404
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json() == {
        "type": "urn:cims-task-service:problem:task-not-found",
        "title": "Task not found",
        "status": 404,
        "detail": "The requested task does not exist.",
    }
    status_reader.assert_awaited_once_with(_TASK_ID)
    assert str(_TASK_ID) not in response.text


@pytest.mark.parametrize("task_id", ["not-a-uuid", str(_UUID_V1)])
def test_get_task_status_rejects_invalid_uuid_before_reading(task_id: str) -> None:
    """Malformed and non-v4 identifiers retain FastAPI's validation response."""

    status_reader = AsyncMock()
    application = _application_with_status_reader(status_reader)

    with TestClient(application) as client:
        response = client.get(f"/api/v1/tasks/{task_id}/status")

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")
    assert isinstance(response.json()["detail"], list)
    status_reader.assert_not_awaited()


def test_openapi_documents_the_task_status_contract() -> None:
    """Status lookup documents UUID v4 validation and both error formats."""

    with TestClient(create_app(Settings())) as client:
        schema = client.get("/openapi.json").json()

    operation = schema["paths"]["/api/v1/tasks/{task_id}/status"]["get"]
    task_id_parameter = next(
        parameter for parameter in operation["parameters"] if parameter["name"] == "task_id"
    )
    assert task_id_parameter["in"] == "path"
    assert task_id_parameter["required"] is True
    assert task_id_parameter["schema"]["format"] == "uuid4"
    assert set(operation["responses"]) >= {"200", "404", "422"}
    assert operation["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/TaskStatusResponse"
    }
    not_found_content = operation["responses"]["404"]["content"]
    assert set(not_found_content) == {"application/problem+json"}
    assert set(not_found_content["application/problem+json"]["schema"]["required"]) == {
        "type",
        "title",
        "status",
        "detail",
    }
    assert operation["responses"]["422"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/HTTPValidationError"
    }
