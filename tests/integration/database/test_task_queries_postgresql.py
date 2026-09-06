"""Task query behavior exercised against independent PostgreSQL sessions."""

from datetime import timedelta
from uuid import UUID

import pytest

from cims_task_service.application.task_creation import CreateTaskCommand, create_task
from cims_task_service.application.task_queries import TaskNotFoundError, get_task
from cims_task_service.domain.task import TaskPriority, TaskStatus
from cims_task_service.infrastructure.database.session import AsyncSessionFactory

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_get_task_hydrates_complete_state_in_a_new_session(
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    """A committed task can be read as a fresh ORM instance with its full state."""

    command = CreateTaskCommand(
        name="Inventory snapshot",
        description="Aggregate the current warehouse quantities",
        priority=TaskPriority.HIGH,
    )
    created = await create_task(
        command,
        session_factory=postgres_session_factory,
        max_attempts=4,
    )

    task = await get_task(
        created.task.id,
        session_factory=postgres_session_factory,
    )

    assert created.created is True
    assert task is not created.task
    assert task.id == created.task.id
    assert task.id.version == 4
    assert task.name == command.name
    assert task.description == command.description
    assert task.priority is TaskPriority.HIGH
    assert task.status is TaskStatus.NEW
    assert task.created_at == created.task.created_at
    assert task.created_at.tzinfo is not None
    assert task.created_at.utcoffset() == timedelta(0)
    assert task.idempotency_key_hash is task.request_fingerprint is None
    assert task.started_at is task.finished_at is None
    assert task.result is task.error is None
    assert task.attempt_count == 0
    assert task.max_attempts == 4
    assert task.dispatch_token == created.task.dispatch_token
    assert task.dispatch_token is not None
    assert task.dispatch_token.version == 4
    assert task.execution_token is None
    assert task.lease_expires_at is None


async def test_get_task_raises_for_an_unknown_identifier(
    postgres_session_factory: AsyncSessionFactory,
) -> None:
    """A primary key absent from PostgreSQL remains attached to the query error."""

    task_id = UUID("ba21a692-6c11-47ac-bc71-392f27f03416")

    with pytest.raises(TaskNotFoundError) as error_info:
        await get_task(
            task_id,
            session_factory=postgres_session_factory,
        )

    assert error_info.value.task_id == task_id
