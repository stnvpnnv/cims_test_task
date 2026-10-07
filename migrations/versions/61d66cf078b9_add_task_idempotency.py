"""Add task creation idempotency metadata.

Revision ID: 61d66cf078b9
Revises: db0f59be3ae6
Create Date: 2026-09-04 11:05:41.720953+00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "61d66cf078b9"
down_revision: str | Sequence[str] | None = "db0f59be3ae6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Store bounded hashes used to deduplicate task creation."""

    op.add_column(
        "tasks",
        sa.Column("idempotency_key_hash", sa.LargeBinary(length=32), nullable=True),
    )
    op.add_column(
        "tasks",
        sa.Column("request_fingerprint", sa.LargeBinary(length=32), nullable=True),
    )
    op.create_check_constraint(
        op.f("ck_tasks_idempotency_pair"),
        "tasks",
        "(idempotency_key_hash IS NULL) = (request_fingerprint IS NULL)",
    )
    op.create_check_constraint(
        op.f("ck_tasks_idempotency_hash_lengths"),
        "tasks",
        """
        (
            idempotency_key_hash IS NULL
            OR octet_length(idempotency_key_hash) = 32
        )
        AND (
            request_fingerprint IS NULL
            OR octet_length(request_fingerprint) = 32
        )
        """,
    )
    op.create_index(
        op.f("ix_tasks_idempotency_key_hash"),
        "tasks",
        ["idempotency_key_hash"],
        unique=True,
        postgresql_where=sa.text("idempotency_key_hash IS NOT NULL"),
    )


def downgrade() -> None:
    """Remove task creation idempotency metadata."""

    op.drop_index(op.f("ix_tasks_idempotency_key_hash"), table_name="tasks")
    op.drop_constraint(
        op.f("ck_tasks_idempotency_hash_lengths"),
        "tasks",
        type_="check",
    )
    op.drop_constraint(
        op.f("ck_tasks_idempotency_pair"),
        "tasks",
        type_="check",
    )
    op.drop_column("tasks", "request_fingerprint")
    op.drop_column("tasks", "idempotency_key_hash")
