"""Create tasks and transactional outbox tables.

Revision ID: db0f59be3ae6
Revises:
Create Date: 2026-08-02 13:54:37.817029+00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "db0f59be3ae6"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the initial task processing schema."""

    op.create_table(
        "tasks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("priority", sa.String(length=8), nullable=False),
        sa.Column(
            "status",
            sa.String(length=16),
            server_default=sa.text("'NEW'"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("result", postgresql.JSONB(), nullable=True),
        sa.Column("error", postgresql.JSONB(), nullable=True),
        sa.Column(
            "attempt_count",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column("dispatch_token", sa.Uuid(), nullable=True),
        sa.Column("execution_token", sa.Uuid(), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "name ~ '[^[:space:]]'",
            name=op.f("ck_tasks_name_nonblank"),
        ),
        sa.CheckConstraint(
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
            name=op.f("ck_tasks_attempts"),
        ),
        sa.CheckConstraint(
            "result IS NULL OR jsonb_typeof(result) = 'object'",
            name=op.f("ck_tasks_result_object"),
        ),
        sa.CheckConstraint(
            "error IS NULL OR jsonb_typeof(error) = 'object'",
            name=op.f("ck_tasks_error_object"),
        ),
        sa.CheckConstraint(
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
            name=op.f("ck_tasks_result_error_status"),
        ),
        sa.CheckConstraint(
            """
            (
                status IN ('COMPLETED', 'FAILED', 'CANCELLED')
            ) = (finished_at IS NOT NULL)
            """,
            name=op.f("ck_tasks_finished_status"),
        ),
        sa.CheckConstraint(
            """
            (status <> 'NEW' OR started_at IS NULL)
            AND (
                status NOT IN ('IN_PROGRESS', 'COMPLETED', 'FAILED')
                OR started_at IS NOT NULL
            )
            """,
            name=op.f("ck_tasks_started_status"),
        ),
        sa.CheckConstraint(
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
            name=op.f("ck_tasks_time_order"),
        ),
        sa.CheckConstraint(
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
            name=op.f("ck_tasks_tokens_status"),
        ),
        sa.CheckConstraint(
            "priority IN ('LOW', 'MEDIUM', 'HIGH')",
            name=op.f("ck_tasks_task_priority"),
        ),
        sa.CheckConstraint(
            """
            status IN (
                'NEW',
                'PENDING',
                'IN_PROGRESS',
                'COMPLETED',
                'FAILED',
                'CANCELLED'
            )
            """,
            name=op.f("ck_tasks_task_status"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_tasks")),
    )
    op.create_index(
        op.f("ix_tasks_created_at_id"),
        "tasks",
        [sa.literal_column("created_at DESC"), sa.literal_column("id DESC")],
        unique=False,
    )
    op.create_index(
        op.f("ix_tasks_status_created_at_id"),
        "tasks",
        [
            "status",
            sa.literal_column("created_at DESC"),
            sa.literal_column("id DESC"),
        ],
        unique=False,
    )
    op.create_index(
        op.f("ix_tasks_priority_created_at_id"),
        "tasks",
        [
            "priority",
            sa.literal_column("created_at DESC"),
            sa.literal_column("id DESC"),
        ],
        unique=False,
    )
    op.create_index(
        op.f("ix_tasks_expired_lease"),
        "tasks",
        ["lease_expires_at", "id"],
        unique=False,
        postgresql_where=sa.text("status = 'IN_PROGRESS'"),
    )

    op.create_table(
        "outbox_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("task_id", sa.Uuid(), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("message_priority", sa.SmallInteger(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "available_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("discarded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "publish_attempts",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column("publisher_token", sa.Uuid(), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "event_type ~ '[^[:space:]]'",
            name=op.f("ck_outbox_events_event_type_nonblank"),
        ),
        sa.CheckConstraint(
            "jsonb_typeof(payload) = 'object'",
            name=op.f("ck_outbox_events_payload_object"),
        ),
        sa.CheckConstraint(
            "message_priority BETWEEN 1 AND 3",
            name=op.f("ck_outbox_events_message_priority"),
        ),
        sa.CheckConstraint(
            "publish_attempts >= 0",
            name=op.f("ck_outbox_events_publish_attempts"),
        ),
        sa.CheckConstraint(
            """
            available_at >= created_at
            AND (published_at IS NULL OR published_at >= created_at)
            AND (discarded_at IS NULL OR discarded_at >= created_at)
            """,
            name=op.f("ck_outbox_events_time_order"),
        ),
        sa.CheckConstraint(
            "published_at IS NULL OR discarded_at IS NULL",
            name=op.f("ck_outbox_events_single_outcome"),
        ),
        sa.CheckConstraint(
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
            name=op.f("ck_outbox_events_publisher_lease"),
        ),
        sa.ForeignKeyConstraint(
            ["task_id"],
            ["tasks.id"],
            name=op.f("fk_outbox_events_task_id_tasks"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_outbox_events")),
    )
    op.create_index(
        op.f("ix_outbox_events_ready"),
        "outbox_events",
        [
            sa.literal_column("message_priority DESC"),
            "available_at",
            "created_at",
            "id",
        ],
        unique=False,
        postgresql_where=sa.text("published_at IS NULL AND discarded_at IS NULL"),
    )
    op.create_index(
        op.f("ix_outbox_events_task_id_created_at"),
        "outbox_events",
        ["task_id", "created_at"],
        unique=False,
    )
    op.create_index(
        op.f("ix_outbox_events_published_at_id"),
        "outbox_events",
        ["published_at", "id"],
        unique=False,
        postgresql_where=sa.text("published_at IS NOT NULL"),
    )
    op.create_index(
        op.f("ix_outbox_events_discarded_at_id"),
        "outbox_events",
        ["discarded_at", "id"],
        unique=False,
        postgresql_where=sa.text("discarded_at IS NOT NULL"),
    )


def downgrade() -> None:
    """Remove the initial task processing schema."""

    op.drop_index(
        op.f("ix_outbox_events_discarded_at_id"),
        table_name="outbox_events",
    )
    op.drop_index(
        op.f("ix_outbox_events_published_at_id"),
        table_name="outbox_events",
    )
    op.drop_index(
        op.f("ix_outbox_events_task_id_created_at"),
        table_name="outbox_events",
    )
    op.drop_index(
        op.f("ix_outbox_events_ready"),
        table_name="outbox_events",
    )
    op.drop_table("outbox_events")

    op.drop_index(op.f("ix_tasks_expired_lease"), table_name="tasks")
    op.drop_index(
        op.f("ix_tasks_priority_created_at_id"),
        table_name="tasks",
    )
    op.drop_index(
        op.f("ix_tasks_status_created_at_id"),
        table_name="tasks",
    )
    op.drop_index(op.f("ix_tasks_created_at_id"), table_name="tasks")
    op.drop_table("tasks")
