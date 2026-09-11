"""Tests for task query orchestration."""

from dataclasses import FrozenInstanceError, dataclass
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock, call
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from cims_task_service.application import task_queries as task_queries_module
from cims_task_service.application.task_queries import (
    DEFAULT_TASK_PAGE,
    DEFAULT_TASK_PAGE_SIZE,
    MAX_TASK_PAGE_SIZE,
    ListTasksQuery,
    ListTasksResult,
    TaskNotFoundError,
    get_task,
    get_task_status,
    list_tasks,
)
from cims_task_service.domain.task import TaskPriority, TaskStatus
from cims_task_service.infrastructure.database.models import TaskModel
from cims_task_service.infrastructure.database.session import AsyncSessionFactory
from cims_task_service.infrastructure.database.task_repository import (
    StoredTaskPage,
    TaskStatusSnapshot,
)


@dataclass(frozen=True, slots=True)
class _QueryHarness:
    session: AsyncSession
    connection: AsyncMock
    query_calls: Mock
    session_factory: AsyncSessionFactory
    create_session: Mock
    session_context: AsyncMock
    repository_factory: Mock
    get_by_id: AsyncMock
    get_status_by_id: AsyncMock
    list_page: AsyncMock


def _query_harness(
    monkeypatch: pytest.MonkeyPatch,
    task: TaskModel | None,
    *,
    status_snapshot: TaskStatusSnapshot | None = None,
    stored_page: StoredTaskPage | None = None,
) -> _QueryHarness:
    connection = AsyncMock()
    session = cast(AsyncSession, SimpleNamespace(connection=connection))
    get_by_id = AsyncMock(return_value=task)
    get_status_by_id = AsyncMock(return_value=status_snapshot)
    list_page = AsyncMock(return_value=stored_page)
    query_calls = Mock()
    query_calls.attach_mock(connection, "connection")
    query_calls.attach_mock(list_page, "list_page")
    repository = SimpleNamespace(
        get_by_id=get_by_id,
        get_status_by_id=get_status_by_id,
        list_page=list_page,
    )
    repository_factory = Mock(return_value=repository)
    monkeypatch.setattr(task_queries_module, "TaskRepository", repository_factory)

    session_context = AsyncMock()
    session_context.__aenter__.return_value = session
    session_context.__aexit__.return_value = False
    create_session = Mock(return_value=session_context)
    session_factory = cast(AsyncSessionFactory, create_session)
    return _QueryHarness(
        session=session,
        connection=connection,
        query_calls=query_calls,
        session_factory=session_factory,
        create_session=create_session,
        session_context=session_context,
        repository_factory=repository_factory,
        get_by_id=get_by_id,
        get_status_by_id=get_status_by_id,
        list_page=list_page,
    )


def test_list_tasks_query_defaults_and_offset_are_stable() -> None:
    """Public defaults use one-based pages and produce a zero-based offset."""

    default_query = ListTasksQuery()
    later_query = ListTasksQuery(page=4, size=7)

    assert DEFAULT_TASK_PAGE == 1
    assert DEFAULT_TASK_PAGE_SIZE == 20
    assert MAX_TASK_PAGE_SIZE == 100
    assert default_query == ListTasksQuery(
        status=None,
        priority=None,
        page=1,
        size=20,
    )
    assert default_query.offset == 0
    assert later_query.offset == 21
    assert not hasattr(default_query, "__dict__")
    with pytest.raises(FrozenInstanceError):
        default_query.__setattr__("page", 2)


@pytest.mark.parametrize(
    ("page", "size", "message"),
    [
        (0, 20, "page must be at least 1"),
        (-1, 20, "page must be at least 1"),
        (1, 0, "size must be between 1 and 100"),
        (1, 101, "size must be between 1 and 100"),
    ],
)
def test_list_tasks_query_rejects_invalid_pagination(
    page: int,
    size: int,
    message: str,
) -> None:
    """Invalid pages and sizes fail before opening a database session."""

    with pytest.raises(ValueError, match=rf"^{message}$"):
        ListTasksQuery(page=page, size=size)


@pytest.mark.asyncio
async def test_list_tasks_returns_repository_page_after_closing_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A filtered page is forwarded exactly and returned after session cleanup."""

    first_task = cast(TaskModel, object())
    second_task = cast(TaskModel, object())
    stored_page = StoredTaskPage(items=(first_task, second_task), total=9)
    harness = _query_harness(
        monkeypatch,
        None,
        stored_page=stored_page,
    )
    query = ListTasksQuery(
        status=TaskStatus.FAILED,
        priority=TaskPriority.HIGH,
        page=3,
        size=2,
    )

    result = await list_tasks(query, session_factory=harness.session_factory)

    assert result == ListTasksResult(
        items=(first_task, second_task),
        total=9,
        page=3,
        size=2,
    )
    harness.create_session.assert_called_once_with()
    harness.session_context.__aenter__.assert_awaited_once_with()
    harness.connection.assert_awaited_once_with(
        execution_options={"isolation_level": "REPEATABLE READ"}
    )
    harness.repository_factory.assert_called_once_with(harness.session)
    harness.list_page.assert_awaited_once_with(
        status=TaskStatus.FAILED,
        priority=TaskPriority.HIGH,
        offset=4,
        limit=2,
    )
    assert harness.query_calls.mock_calls == [
        call.connection(execution_options={"isolation_level": "REPEATABLE READ"}),
        call.list_page(
            status=TaskStatus.FAILED,
            priority=TaskPriority.HIGH,
            offset=4,
            limit=2,
        ),
    ]
    harness.get_by_id.assert_not_awaited()
    harness.get_status_by_id.assert_not_awaited()
    harness.session_context.__aexit__.assert_awaited_once_with(None, None, None)


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
    harness.connection.assert_not_awaited()
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
    harness.connection.assert_not_awaited()
    harness.get_by_id.assert_awaited_once_with(task_id)
    exit_call = harness.session_context.__aexit__.await_args
    assert exit_call is not None
    assert exit_call.args[0] is TaskNotFoundError
    assert exit_call.args[1] is error_info.value
    assert exit_call.args[2] is not None


@pytest.mark.asyncio
async def test_get_task_status_returns_snapshot_after_closing_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful status query closes its read session before returning."""

    task_id = uuid4()
    snapshot = cast(TaskStatusSnapshot, object())
    harness = _query_harness(
        monkeypatch,
        None,
        status_snapshot=snapshot,
    )

    result = await get_task_status(task_id, session_factory=harness.session_factory)

    assert result is snapshot
    harness.create_session.assert_called_once_with()
    harness.session_context.__aenter__.assert_awaited_once_with()
    harness.connection.assert_not_awaited()
    harness.repository_factory.assert_called_once_with(harness.session)
    harness.get_status_by_id.assert_awaited_once_with(task_id)
    harness.get_by_id.assert_not_awaited()
    harness.session_context.__aexit__.assert_awaited_once_with(None, None, None)


@pytest.mark.asyncio
async def test_get_task_status_raises_not_found_inside_session_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing status reuses the task-not-found error and closes its session."""

    task_id = uuid4()
    harness = _query_harness(monkeypatch, None)

    with pytest.raises(TaskNotFoundError) as error_info:
        await get_task_status(task_id, session_factory=harness.session_factory)

    assert error_info.value.task_id == task_id
    harness.connection.assert_not_awaited()
    harness.get_status_by_id.assert_awaited_once_with(task_id)
    harness.get_by_id.assert_not_awaited()
    exit_call = harness.session_context.__aexit__.await_args
    assert exit_call is not None
    assert exit_call.args[0] is TaskNotFoundError
    assert exit_call.args[1] is error_info.value
    assert exit_call.args[2] is not None
