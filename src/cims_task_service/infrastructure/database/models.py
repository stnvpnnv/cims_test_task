"""SQLAlchemy models for tasks and their transactional outbox events."""

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    Uuid,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from cims_task_service.domain.task import TaskPriority, TaskStatus
from cims_task_service.infrastructure.database.base import Base

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]


class TaskModel(Base):
    """Persisted task state and worker concurrency metadata."""

    __tablename__ = "tasks"
    __table_args__ = (
        CheckConstraint("name ~ '[^[:space:]]'", name="name_nonblank"),
        CheckConstraint(
            """
            attempt_count >= 0
            AND max_attempts > 0
            AND attempt_count <= max_attempts
            AND (status <> 'NEW' OR attempt_count = 0)
            AND (
                status NOT IN ('NEW', 'PENDING')
                OR attempt_count < max_attempts
            )
            AND (
                status NOT IN ('IN_PROGRESS', 'COMPLETED', 'FAILED')
                OR attempt_count >= 1
            )
            """,
            name="attempts",
        ),
        CheckConstraint(
            "result IS NULL OR jsonb_typeof(result) = 'object'",
            name="result_object",
        ),
        CheckConstraint(
            "error IS NULL OR jsonb_typeof(error) = 'object'",
            name="error_object",
        ),
        CheckConstraint(
            """
            (
                status = 'COMPLETED'
                AND error IS NULL
            )
            OR (
                status = 'FAILED'
                AND result IS NULL
            )
            OR (
                status IN ('NEW', 'PENDING', 'IN_PROGRESS', 'CANCELLED')
                AND result IS NULL
                AND error IS NULL
            )
            """,
            name="result_error_status",
        ),
        CheckConstraint(
            """
            (
                status IN ('COMPLETED', 'FAILED', 'CANCELLED')
            ) = (finished_at IS NOT NULL)
            """,
            name="finished_status",
        ),
        CheckConstraint(
            """
            (status <> 'NEW' OR started_at IS NULL)
            AND (
                status NOT IN ('IN_PROGRESS', 'COMPLETED', 'FAILED')
                OR started_at IS NOT NULL
            )
            """,
            name="started_status",
        ),
        CheckConstraint(
            """
            (started_at IS NULL OR started_at >= created_at)
            AND (finished_at IS NULL OR finished_at >= created_at)
            AND (
                finished_at IS NULL
                OR started_at IS NULL
                OR finished_at >= started_at
            )
            AND (
                lease_expires_at IS NULL
                OR lease_expires_at >= created_at
            )
            """,
            name="time_order",
        ),
        CheckConstraint(
            """
            (
                status IN ('NEW', 'PENDING')
                AND dispatch_token IS NOT NULL
                AND execution_token IS NULL
                AND lease_expires_at IS NULL
            )
            OR (
                status = 'IN_PROGRESS'
                AND dispatch_token IS NULL
                AND execution_token IS NOT NULL
                AND lease_expires_at IS NOT NULL
            )
            OR (
                status IN ('COMPLETED', 'FAILED', 'CANCELLED')
                AND dispatch_token IS NULL
                AND execution_token IS NULL
                AND lease_expires_at IS NULL
            )
            """,
            name="tokens_status",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    name: Mapped[str] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text)
    priority: Mapped[TaskPriority] = mapped_column(
        Enum(
            TaskPriority,
            name="task_priority",
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            length=8,
            values_callable=lambda enum: [item.value for item in enum],
        )
    )
    status: Mapped[TaskStatus] = mapped_column(
        Enum(
            TaskStatus,
            name="task_status",
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            length=16,
            values_callable=lambda enum: [item.value for item in enum],
        ),
        default=TaskStatus.NEW,
        server_default=text("'NEW'"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    result: Mapped[JsonObject | None] = mapped_column(
        JSONB(none_as_null=True),
        nullable=True,
    )
    error: Mapped[JsonObject | None] = mapped_column(
        JSONB(none_as_null=True),
        nullable=True,
    )

    attempt_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        server_default=text("0"),
    )
    max_attempts: Mapped[int] = mapped_column(Integer)
    dispatch_token: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        nullable=True,
    )
    execution_token: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        nullable=True,
    )
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class OutboxEventModel(Base):
    """Message awaiting reliable publication to RabbitMQ."""

    __tablename__ = "outbox_events"
    __table_args__ = (
        CheckConstraint(
            "event_type ~ '[^[:space:]]'",
            name="event_type_nonblank",
        ),
        CheckConstraint(
            "jsonb_typeof(payload) = 'object'",
            name="payload_object",
        ),
        CheckConstraint(
            "message_priority BETWEEN 1 AND 3",
            name="message_priority",
        ),
        CheckConstraint("publish_attempts >= 0", name="publish_attempts"),
        CheckConstraint(
            """
            available_at >= created_at
            AND (published_at IS NULL OR published_at >= created_at)
            AND (discarded_at IS NULL OR discarded_at >= created_at)
            """,
            name="time_order",
        ),
        CheckConstraint(
            "published_at IS NULL OR discarded_at IS NULL",
            name="single_outcome",
        ),
        CheckConstraint(
            """
            (
                publisher_token IS NULL
                AND lease_expires_at IS NULL
            )
            OR (
                publisher_token IS NOT NULL
                AND lease_expires_at IS NOT NULL
                AND published_at IS NULL
                AND discarded_at IS NULL
            )
            """,
            name="publisher_lease",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    task_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("tasks.id", ondelete="RESTRICT"),
    )
    event_type: Mapped[str] = mapped_column(String(64))
    payload: Mapped[JsonObject] = mapped_column(JSONB(none_as_null=True))
    message_priority: Mapped[int] = mapped_column(SmallInteger)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    discarded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    publish_attempts: Mapped[int] = mapped_column(
        Integer,
        default=0,
        server_default=text("0"),
    )
    publisher_token: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        nullable=True,
    )
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)


Index(
    "ix_tasks_created_at_id",
    TaskModel.created_at.desc(),
    TaskModel.id.desc(),
)
Index(
    "ix_tasks_status_created_at_id",
    TaskModel.status,
    TaskModel.created_at.desc(),
    TaskModel.id.desc(),
)
Index(
    "ix_tasks_priority_created_at_id",
    TaskModel.priority,
    TaskModel.created_at.desc(),
    TaskModel.id.desc(),
)
Index(
    "ix_tasks_expired_lease",
    TaskModel.lease_expires_at,
    TaskModel.id,
    postgresql_where=TaskModel.status == TaskStatus.IN_PROGRESS,
)

Index(
    "ix_outbox_events_ready",
    OutboxEventModel.message_priority.desc(),
    OutboxEventModel.available_at,
    OutboxEventModel.created_at,
    OutboxEventModel.id,
    postgresql_where=OutboxEventModel.published_at.is_(None)
    & OutboxEventModel.discarded_at.is_(None),
)
Index(
    "ix_outbox_events_task_id_created_at",
    OutboxEventModel.task_id,
    OutboxEventModel.created_at,
)
Index(
    "ix_outbox_events_published_at_id",
    OutboxEventModel.published_at,
    OutboxEventModel.id,
    postgresql_where=OutboxEventModel.published_at.is_not(None),
)
Index(
    "ix_outbox_events_discarded_at_id",
    OutboxEventModel.discarded_at,
    OutboxEventModel.id,
    postgresql_where=OutboxEventModel.discarded_at.is_not(None),
)
