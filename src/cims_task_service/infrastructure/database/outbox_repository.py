"""Lease-based persistence operations for transactional outbox publishers."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final
from uuid import UUID, uuid4

from sqlalchemy import func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from cims_task_service.domain.task import TaskStatus
from cims_task_service.infrastructure.database.models import (
    JsonObject,
    OutboxEventModel,
    TaskModel,
)

MAX_OUTBOX_ERROR_LENGTH: Final = 2_048


@dataclass(frozen=True, slots=True)
class ClaimedOutboxEvent:
    """Publication snapshot detached from its reservation transaction."""

    id: UUID
    task_id: UUID
    event_type: str
    payload: JsonObject
    message_priority: int
    created_at: datetime
    available_at: datetime
    publish_attempts: int
    publisher_token: UUID
    lease_expires_at: datetime


class OutboxRepository:
    """Reserve and finalize outbox events without owning transactions."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def claim_batch(
        self,
        *,
        event_type: str,
        batch_size: int,
        lease_duration: timedelta,
    ) -> tuple[ClaimedOutboxEvent, ...]:
        """Reserve a priority-ordered batch after locking task rows first."""

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if lease_duration <= timedelta(0):
            raise ValueError("lease_duration must be positive")

        candidate_statement = (
            select(TaskModel.id, TaskModel.status, OutboxEventModel.id)
            .join(OutboxEventModel, OutboxEventModel.task_id == TaskModel.id)
            .where(
                OutboxEventModel.event_type == event_type,
                OutboxEventModel.available_at <= func.statement_timestamp(),
                OutboxEventModel.published_at.is_(None),
                OutboxEventModel.discarded_at.is_(None),
                or_(
                    OutboxEventModel.publisher_token.is_(None),
                    OutboxEventModel.lease_expires_at <= func.statement_timestamp(),
                ),
                TaskModel.status != TaskStatus.CANCELLED,
            )
            .order_by(
                OutboxEventModel.message_priority.desc(),
                OutboxEventModel.available_at,
                OutboxEventModel.created_at,
                OutboxEventModel.id,
            )
            .limit(batch_size)
            .with_for_update(of=TaskModel, skip_locked=True)
        )
        candidates = tuple((await self._session.execute(candidate_statement)).all())
        if not candidates:
            return ()

        publisher_token = uuid4()
        candidate_event_ids = tuple(event_id for _task_id, _status, event_id in candidates)
        claim_statement = (
            update(OutboxEventModel)
            .where(
                OutboxEventModel.id.in_(candidate_event_ids),
                OutboxEventModel.event_type == event_type,
                OutboxEventModel.available_at <= func.clock_timestamp(),
                OutboxEventModel.published_at.is_(None),
                OutboxEventModel.discarded_at.is_(None),
                or_(
                    OutboxEventModel.publisher_token.is_(None),
                    OutboxEventModel.lease_expires_at <= func.clock_timestamp(),
                ),
            )
            .values(
                publish_attempts=OutboxEventModel.publish_attempts + 1,
                publisher_token=publisher_token,
                lease_expires_at=func.clock_timestamp() + lease_duration,
            )
            .returning(OutboxEventModel)
            .execution_options(populate_existing=True)
        )
        claimed_events = tuple((await self._session.scalars(claim_statement)).all())
        claimed_by_id = {event.id: event for event in claimed_events}
        claimed_snapshots_by_id = {
            event.id: self._to_claimed_event(event) for event in claimed_events
        }
        pending_task_ids = tuple(
            dict.fromkeys(
                task_id
                for task_id, task_status, event_id in candidates
                if task_status is TaskStatus.NEW and event_id in claimed_by_id
            )
        )

        if pending_task_ids:
            pending_statement = (
                update(TaskModel)
                .where(
                    TaskModel.id.in_(pending_task_ids),
                    TaskModel.status == TaskStatus.NEW,
                )
                .values(status=TaskStatus.PENDING)
            )
            await self._session.execute(pending_statement)

        return tuple(
            claimed_snapshots_by_id[event_id]
            for _task_id, _status, event_id in candidates
            if event_id in claimed_snapshots_by_id
        )

    async def mark_published(
        self,
        event_id: UUID,
        *,
        publisher_token: UUID,
    ) -> bool:
        """Finalize a confirmed publication only for its current lease owner."""

        statement = (
            update(OutboxEventModel)
            .where(
                OutboxEventModel.id == event_id,
                OutboxEventModel.publisher_token == publisher_token,
                OutboxEventModel.published_at.is_(None),
                OutboxEventModel.discarded_at.is_(None),
            )
            .values(
                published_at=func.clock_timestamp(),
                publisher_token=None,
                lease_expires_at=None,
                last_error=None,
            )
            .returning(OutboxEventModel.id)
        )
        return await self._session.scalar(statement) is not None

    async def reschedule(
        self,
        event_id: UUID,
        *,
        publisher_token: UUID,
        retry_delay: timedelta,
        failure_summary: str,
    ) -> bool:
        """Release a failed publication lease and defer its next reservation."""

        if retry_delay < timedelta(0):
            raise ValueError("retry_delay must not be negative")

        statement = (
            update(OutboxEventModel)
            .where(
                OutboxEventModel.id == event_id,
                OutboxEventModel.publisher_token == publisher_token,
                OutboxEventModel.published_at.is_(None),
                OutboxEventModel.discarded_at.is_(None),
            )
            .values(
                available_at=func.clock_timestamp() + retry_delay,
                publisher_token=None,
                lease_expires_at=None,
                last_error=failure_summary[:MAX_OUTBOX_ERROR_LENGTH],
            )
            .returning(OutboxEventModel.id)
        )
        return await self._session.scalar(statement) is not None

    @staticmethod
    def _to_claimed_event(event: OutboxEventModel) -> ClaimedOutboxEvent:
        if event.publisher_token is None or event.lease_expires_at is None:
            raise RuntimeError("claimed outbox event is missing its publisher lease")

        return ClaimedOutboxEvent(
            id=event.id,
            task_id=event.task_id,
            event_type=event.event_type,
            payload=dict(event.payload),
            message_priority=event.message_priority,
            created_at=event.created_at,
            available_at=event.available_at,
            publish_attempts=event.publish_attempts,
            publisher_token=event.publisher_token,
            lease_expires_at=event.lease_expires_at,
        )
