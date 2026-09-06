"""Smoke tests for the FastAPI application."""

from importlib.metadata import version
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncEngine

from cims_task_service import main as main_module
from cims_task_service.api.dependencies import ApplicationResources
from cims_task_service.config import Settings
from cims_task_service.infrastructure.database.session import AsyncSessionFactory
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
    assert "/api/v1/tasks" in schema["paths"]


def test_application_loads_debug_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The factory uses environment-backed settings supplied by its caller."""

    monkeypatch.setenv("CIMS_DEBUG", "true")

    application = create_app(Settings())

    assert application.debug is True


def test_application_resources_follow_the_lifespan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Database resources are initialized lazily and their engine is disposed."""

    engine = cast(AsyncEngine, object())
    session_factory = cast(AsyncSessionFactory, object())
    create_engine = Mock(return_value=engine)
    create_sessions = Mock(return_value=session_factory)
    dispose_engine = AsyncMock()
    monkeypatch.setattr(main_module, "create_database_engine", create_engine)
    monkeypatch.setattr(main_module, "create_session_factory", create_sessions)
    monkeypatch.setattr(main_module, "dispose_database_engine", dispose_engine)
    settings = Settings(task_max_attempts=5)

    application = create_app(settings)

    create_engine.assert_not_called()
    assert not hasattr(application.state, "resources")
    with TestClient(application) as client:
        assert client.get("/health/live").status_code == 200
        assert application.state.resources == ApplicationResources(
            session_factory=session_factory,
            task_max_attempts=5,
        )
    assert not hasattr(application.state, "resources")
    create_engine.assert_called_once_with(settings)
    create_sessions.assert_called_once_with(engine)
    dispose_engine.assert_awaited_once_with(engine)
