"""HTTP contract tests for filtered task listing."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cims_task_service.api import dependencies as api_dependencies
from cims_task_service.application.task_queries import ListTasksQuery, ListTasksResult
from cims_task_service.config import Settings
from cims_task_service.domain.task import TaskPriority, TaskStatus
from cims_task_service.infrastructure.database.models import TaskModel
from cims_task_service.main import create_app


def _failed_task() -> TaskModel:
    created_at = datetime(2026, 9, 10, 2, 3, 4, tzinfo=UTC)
    return TaskModel(
        id=UUID("53aed18c-9ce4-42b1-9694-0ad67b72ef83"),
        name="Daily inventory",
        description="Aggregate warehouse quantities",
        priority=TaskPriority.HIGH,
        status=TaskStatus.FAILED,
        created_at=created_at,
        idempotency_key_hash=b"k" * 32,
        request_fingerprint=b"f" * 32,
        started_at=created_at + timedelta(seconds=1),
        finished_at=created_at + timedelta(seconds=2),
        result=None,
        error={"code": "PROCESSING_FAILED", "retryable": False},
        attempt_count=3,
        max_attempts=3,
        dispatch_token=None,
        execution_token=None,
        lease_expires_at=None,
    )


def _application_with_task_list_reader(task_list_reader: AsyncMock) -> FastAPI:
    application = create_app(Settings())
    application.dependency_overrides[api_dependencies.get_task_list_reader] = lambda: (
        task_list_reader
    )
    return application


def test_list_tasks_uses_defaults_and_production_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default listing binds lifespan resources and returns only public task data."""

    task = _failed_task()
    list_tasks = AsyncMock(
        return_value=ListTasksResult(
            items=(task,),
            total=7,
            page=1,
            size=20,
        )
    )
    monkeypatch.setattr(api_dependencies, "list_tasks", list_tasks)
    application = create_app(Settings())

    with TestClient(application) as client:
        response = client.get("/api/v1/tasks")
        session_factory = application.state.resources.session_factory

    assert response.status_code == 200
    assert response.json() == {
        "items": [
            {
                "id": str(task.id),
                "name": task.name,
                "description": task.description,
                "priority": "HIGH",
                "status": "FAILED",
                "created_at": "2026-09-10T02:03:04Z",
                "started_at": "2026-09-10T02:03:05Z",
                "finished_at": "2026-09-10T02:03:06Z",
                "result": None,
                "error": {"code": "PROCESSING_FAILED", "retryable": False},
            }
        ],
        "total": 7,
        "page": 1,
        "size": 20,
    }
    list_tasks.assert_awaited_once_with(
        ListTasksQuery(),
        session_factory=session_factory,
    )


def test_list_tasks_forwards_explicit_filters_and_pagination() -> None:
    """Explicit query parameters become one typed application query."""

    task_list_reader = AsyncMock(
        return_value=ListTasksResult(
            items=(),
            total=8,
            page=2,
            size=5,
        )
    )
    application = _application_with_task_list_reader(task_list_reader)

    with TestClient(application) as client:
        response = client.get(
            "/api/v1/tasks",
            params={
                "status": "FAILED",
                "priority": "HIGH",
                "page": "2",
                "size": "5",
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "items": [],
        "total": 8,
        "page": 2,
        "size": 5,
    }
    task_list_reader.assert_awaited_once_with(
        ListTasksQuery(
            status=TaskStatus.FAILED,
            priority=TaskPriority.HIGH,
            page=2,
            size=5,
        )
    )


@pytest.mark.parametrize(
    "params",
    [
        {"status": "UNKNOWN"},
        {"priority": "URGENT"},
        {"page": "0"},
        {"page": "many"},
        {"size": "0"},
        {"size": "101"},
        {"unexpected": "value"},
    ],
)
def test_list_tasks_rejects_invalid_query_before_reading(
    params: dict[str, str],
) -> None:
    """Invalid filters, pagination, and unknown fields retain the validation contract."""

    task_list_reader = AsyncMock()
    application = _application_with_task_list_reader(task_list_reader)

    with TestClient(application) as client:
        response = client.get("/api/v1/tasks", params=params)

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")
    assert isinstance(response.json()["detail"], list)
    task_list_reader.assert_not_awaited()


def test_openapi_documents_the_complete_task_list_contract() -> None:
    """Task listing documents filters, pagination, ordering, and response shape."""

    with TestClient(create_app(Settings())) as client:
        schema = client.get("/openapi.json").json()

    operation = schema["paths"]["/api/v1/tasks"]["get"]
    parameters = {parameter["name"]: parameter for parameter in operation["parameters"]}
    assert set(parameters) == {"status", "priority", "page", "size"}
    assert all(parameter["in"] == "query" for parameter in parameters.values())
    assert all(parameter["required"] is False for parameter in parameters.values())

    status_schema = parameters["status"]["schema"]
    priority_schema = parameters["priority"]["schema"]
    assert {"$ref": "#/components/schemas/TaskStatus"} in status_schema["anyOf"]
    assert {"$ref": "#/components/schemas/TaskPriority"} in priority_schema["anyOf"]
    assert parameters["page"]["schema"]["default"] == 1
    assert parameters["page"]["schema"]["minimum"] == 1
    assert parameters["size"]["schema"]["default"] == 20
    assert parameters["size"]["schema"]["minimum"] == 1
    assert parameters["size"]["schema"]["maximum"] == 100

    description = operation["description"]
    assert "created_at DESC" in description
    assert "id DESC" in description
    assert "AND" in description
    responses = operation["responses"]
    assert "404" not in responses
    assert responses["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/TaskListResponse"
    }
    assert responses["422"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/HTTPValidationError"
    }
