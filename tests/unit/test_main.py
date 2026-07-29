"""Smoke tests for the FastAPI application."""

from importlib.metadata import version

import pytest
from fastapi.testclient import TestClient

from cims_task_service.config import Settings
from cims_task_service.main import create_app


def test_liveness_and_openapi_contract() -> None:
    """The running API exposes its liveness and OpenAPI contracts."""

    with TestClient(create_app(Settings(debug=False))) as client:
        health_response = client.get("/health/live")
        openapi_response = client.get("/openapi.json")

    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok"}
    assert openapi_response.status_code == 200

    schema = openapi_response.json()
    assert schema["info"]["title"] == "CIMS Task Service"
    assert schema["info"]["description"] == ("Fault-tolerant asynchronous task processing service")
    assert schema["info"]["version"] == version("cims-task-service")
    assert "/health/live" in schema["paths"]


def test_application_loads_debug_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The factory uses environment-backed settings supplied by its caller."""

    monkeypatch.setenv("CIMS_DEBUG", "true")

    application = create_app(Settings())

    assert application.debug is True
