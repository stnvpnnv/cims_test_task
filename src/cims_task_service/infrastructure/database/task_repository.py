"""Persistence operations for task aggregates and their outbox events."""

from dataclasses import dataclass
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import load_only
from sqlalchemy.sql.elements import ColumnElement

from cims_task_service.domain.task import TaskPriority, TaskStatus
from cims_task_service.infrastructure.database.models import (
    OutboxEventModel,
    TaskModel,
)


@dataclass(frozen=True, slots=True)
class TaskStatusSnapshot:
    """Minimal persisted state required by the status endpoint."""

    id: UUID
    status: TaskStatus


@dataclass(frozen=True, slots=True)
class StoredTaskPage:
    """One deterministic page and the exact filtered task count."""

    items: tuple[TaskModel, ...]
    total: int


@dataclass(frozen=True, slots=True)
class StoredTaskCreation:
    """Persisted task together with whether this call created it."""

    task: TaskModel
    created: bool


class TaskRepository:
    """Store task aggregates without owning the surrounding transaction."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, task_id: UUID) -> TaskModel | None:
        """Return a task by primary key without acquiring a row lock."""

        return await self._session.get(TaskModel, task_id)

    async def get_status_by_id(self, task_id: UUID) -> TaskStatusSnapshot | None:
        """Return only the task identifier and status without locking its row."""

        statement = select(TaskModel.id, TaskModel.status).where(TaskModel.id == task_id)
        row = (await self._session.execute(statement)).one_or_none()
        if row is None:
            return None

        stored_task_id, task_status = row
        return TaskStatusSnapshot(id=stored_task_id, status=task_status)

    async def list_page(
        self,
        *,
        status: TaskStatus | None,
        priority: TaskPriority | None,
        offset: int,
        limit: int,
    ) -> StoredTaskPage:
        """Return a newest-first page and exact count without locking task rows."""

        conditions: list[ColumnElement[bool]] = []
        if status is not None:
            conditions.append(TaskModel.status == status)
        if priority is not None:
            conditions.append(TaskModel.priority == priority)

        count_statement = select(func.count()).select_from(TaskModel).where(*conditions)
        total = (await self._session.scalars(count_statement)).one()
        if offset >= total:
            return StoredTaskPage(items=(), total=total)

        page_statement = (
            select(TaskModel)
            .options(
                load_only(
                    TaskModel.id,
                    TaskModel.name,
                    TaskModel.description,
                    TaskModel.priority,
                    TaskModel.status,
                    TaskModel.created_at,
                    TaskModel.started_at,
                    TaskModel.finished_at,
                    TaskModel.result,
                    TaskModel.error,
                    raiseload=True,
                )
            )
            .where(*conditions)
            .order_by(TaskModel.created_at.desc(), TaskModel.id.desc())
            .offset(offset)
            .limit(limit)
        )
        items = tuple((await self._session.scalars(page_statement)).all())
        return StoredTaskPage(items=items, total=total)

    async def create_with_outbox(
        self,
        *,
        name: str,
        description: str,
        priority: TaskPriority,
        max_attempts: int,
        idempotency_key_hash: bytes | None,
        request_fingerprint: bytes | None,
        event_type: str,
        message_priority: int,
    ) -> StoredTaskCreation:
        """Create a task and outbox event, or return its idempotent predecessor."""

        task_id = uuid4()
        dispatch_token = uuid4()
        create_task_statement = (
            insert(TaskModel)
            .values(
                id=task_id,
                name=name,
                description=description,
                priority=priority,
                status=TaskStatus.NEW,
                idempotency_key_hash=idempotency_key_hash,
                request_fingerprint=request_fingerprint,
                started_at=None,
                finished_at=None,
                result=None,
                error=None,
                attempt_count=0,
                max_attempts=max_attempts,
                dispatch_token=dispatch_token,
                execution_token=None,
                lease_expires_at=None,
            )
            .on_conflict_do_nothing(
                index_elements=[TaskModel.idempotency_key_hash],
                index_where=TaskModel.idempotency_key_hash.is_not(None),
            )
            .returning(TaskModel)
        )

        inserted_task = (await self._session.scalars(create_task_statement)).one_or_none()
        if inserted_task is not None:
            inserted_dispatch_token = inserted_task.dispatch_token
            if inserted_dispatch_token is None:
                message = "created task has no dispatch token"
                raise RuntimeError(message)

            outbox_event = OutboxEventModel(
                id=uuid4(),
                task_id=inserted_task.id,
                event_type=event_type,
                payload={
                    "task_id": str(inserted_task.id),
                    "dispatch_token": str(inserted_dispatch_token),
                },
                message_priority=message_priority,
                published_at=None,
                discarded_at=None,
                publish_attempts=0,
                publisher_token=None,
                lease_expires_at=None,
                last_error=None,
            )
            self._session.add(outbox_event)
            await self._session.flush()
            return StoredTaskCreation(task=inserted_task, created=True)

        if idempotency_key_hash is None:
            message = "task insertion returned no row without an idempotency key"
            raise RuntimeError(message)

        existing_task_statement = select(TaskModel).where(
            TaskModel.idempotency_key_hash == idempotency_key_hash
        )
        existing_task = (await self._session.scalars(existing_task_statement)).one_or_none()
        if existing_task is None:
            message = "idempotency conflict was not followed by a visible task row"
            raise RuntimeError(message)

        return StoredTaskCreation(task=existing_task, created=False)
