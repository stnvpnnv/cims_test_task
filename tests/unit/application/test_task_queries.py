"""Tests for task query orchestration."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from cims_task_service.application import task_queries as task_queries_module
from cims_task_service.application.task_queries import TaskNotFoundError, get_task
from cims_task_service.infrastructure.database.models import TaskModel
from cims_task_service.infrastructure.database.session import AsyncSessionFactory


@dataclass(frozen=True, slots=True)
class _QueryHarness:
    session: AsyncSession
    session_factory: AsyncSessionFactory
    create_session: Mock
    session_context: AsyncMock
    repository_factory: Mock
    get_by_id: AsyncMock


def _query_harness(
    monkeypatch: pytest.MonkeyPatch,
    task: TaskModel | None,
) -> _QueryHarness:
    session = cast(AsyncSession, object())
    get_by_id = AsyncMock(return_value=task)
    repository = SimpleNamespace(get_by_id=get_by_id)
    repository_factory = Mock(return_value=repository)
    monkeypatch.setattr(task_queries_module, "TaskRepository", repository_factory)

    session_context = AsyncMock()
    session_context.__aenter__.return_value = session
    session_context.__aexit__.return_value = False
    create_session = Mock(return_value=session_context)
    session_factory = cast(AsyncSessionFactory, create_session)
    return _QueryHarness(
        session=session,
        session_factory=session_factory,
        create_session=create_session,
        session_context=session_context,
        repository_factory=repository_factory,
        get_by_id=get_by_id,
    )


@pytest.mark.asyncio
async def test_get_task_returns_repository_result_after_closing_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful query uses one session and closes it before returning."""

    task_id = uuid4()
    task = cast(TaskModel, object())
    harness = _query_harness(monkeypatch, task)

    result = await get_task(task_id, session_factory=harness.session_factory)

    assert result is task
    harness.create_session.assert_called_once_with()
    harness.session_context.__aenter__.assert_awaited_once_with()
    harness.repository_factory.assert_called_once_with(harness.session)
    harness.get_by_id.assert_awaited_once_with(task_id)
    harness.session_context.__aexit__.assert_awaited_once_with(None, None, None)


@pytest.mark.asyncio
async def test_get_task_raises_not_found_inside_session_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing task carries its identifier and still closes the read session."""

    task_id = uuid4()
    harness = _query_harness(monkeypatch, None)

    with pytest.raises(TaskNotFoundError) as error_info:
        await get_task(task_id, session_factory=harness.session_factory)

    assert error_info.value.task_id == task_id
    harness.get_by_id.assert_awaited_once_with(task_id)
    exit_call = harness.session_context.__aexit__.await_args
    assert exit_call is not None
    assert exit_call.args[0] is TaskNotFoundError
    assert exit_call.args[1] is error_info.value
    assert exit_call.args[2] is not None
