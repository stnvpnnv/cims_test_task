"""Metadata tests for the PostgreSQL persistence models."""

from collections.abc import Iterable
from typing import cast
from uuid import UUID

from sqlalchemy import CheckConstraint, Enum, Index, LargeBinary, Table, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql.base import PGDialect
from sqlalchemy.orm import configure_mappers
from sqlalchemy.schema import CreateIndex, CreateTable

from cims_task_service.infrastructure.database.base import Base
from cims_task_service.infrastructure.database.models import (
    OutboxEventModel,
    TaskModel,
)

_POSTGRESQL_DIALECT = PGDialect()  # type: ignore[no-untyped-call]


def _constraint_names(constraints: Iterable[object]) -> set[str]:
    return {
        str(constraint.name)
        for constraint in constraints
        if isinstance(constraint, CheckConstraint) and constraint.name is not None
    }


def _index_sql(index: Index) -> str:
    return str(CreateIndex(index).compile(dialect=_POSTGRESQL_DIALECT))


def test_metadata_contains_the_persistence_tables() -> None:
    """Both models share the metadata consumed later by Alembic."""

    assert set(Base.metadata.tables) == {"tasks", "outbox_events"}
    assert TaskModel.metadata is Base.metadata
    assert OutboxEventModel.metadata is Base.metadata


def test_task_columns_cover_contract_and_concurrency_state() -> None:
    """The task table contains public fields plus minimal fencing metadata."""

    table = cast(Table, TaskModel.__table__)

    assert set(table.c.keys()) == {
        "id",
        "name",
        "description",
        "priority",
        "status",
        "created_at",
        "idempotency_key_hash",
        "request_fingerprint",
        "started_at",
        "finished_at",
        "result",
        "error",
        "attempt_count",
        "max_attempts",
        "dispatch_token",
        "execution_token",
        "lease_expires_at",
    }
    assert table.c.id.primary_key
    assert table.c.name.nullable is False
    assert table.c.description.nullable is False
    priority_type = cast(Enum, table.c.priority.type)
    status_type = cast(Enum, table.c.status.type)
    assert priority_type.native_enum is False
    assert priority_type.enums == ["LOW", "MEDIUM", "HIGH"]
    assert status_type.native_enum is False
    assert status_type.enums == [
        "NEW",
        "PENDING",
        "IN_PROGRESS",
        "COMPLETED",
        "FAILED",
        "CANCELLED",
    ]
    assert cast(Uuid[UUID], table.c.id.type).as_uuid is True
    assert cast(JSONB, table.c.result.type).none_as_null is True
    assert cast(JSONB, table.c.error.type).none_as_null is True
    assert cast(LargeBinary, table.c.idempotency_key_hash.type).length == 32
    assert cast(LargeBinary, table.c.request_fingerprint.type).length == 32
    assert table.c.idempotency_key_hash.nullable is True
    assert table.c.request_fingerprint.nullable is True
    assert table.c.idempotency_key_hash.server_default is None
    assert table.c.request_fingerprint.server_default is None
    assert table.c.status.server_default is not None
    assert table.c.created_at.server_default is not None
    assert table.c.max_attempts.server_default is None
    assert table.primary_key.name == "pk_tasks"


def test_task_constraints_have_stable_names() -> None:
    """Database invariants can be changed safely by named migrations."""

    table = cast(Table, TaskModel.__table__)

    assert _constraint_names(table.constraints) == {
        "ck_tasks_attempts",
        "ck_tasks_error_object",
        "ck_tasks_finished_status",
        "ck_tasks_idempotency_hash_lengths",
        "ck_tasks_idempotency_pair",
        "ck_tasks_name_nonblank",
        "ck_tasks_result_error_status",
        "ck_tasks_result_object",
        "ck_tasks_started_status",
        "ck_tasks_task_priority",
        "ck_tasks_task_status",
        "ck_tasks_time_order",
        "ck_tasks_tokens_status",
    }


def test_task_idempotency_constraints_require_paired_sha256_hashes() -> None:
    """Optional idempotency metadata is paired and fixed to SHA-256 digest size."""

    table = cast(Table, TaskModel.__table__)
    constraints = {
        str(constraint.name): constraint
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint) and constraint.name is not None
    }

    pair_sql = " ".join(str(constraints["ck_tasks_idempotency_pair"].sqltext).split())
    lengths_sql = " ".join(str(constraints["ck_tasks_idempotency_hash_lengths"].sqltext).split())
    assert pair_sql == ("(idempotency_key_hash IS NULL) = (request_fingerprint IS NULL)")
    assert lengths_sql.count("octet_length(idempotency_key_hash) = 32") == 1
    assert lengths_sql.count("octet_length(request_fingerprint) = 32") == 1


def test_task_indexes_match_list_and_recovery_queries() -> None:
    """Indexes support deterministic pagination, filters, and lease recovery."""

    table = cast(Table, TaskModel.__table__)
    indexes = {str(index.name): index for index in table.indexes}

    assert set(indexes) == {
        "ix_tasks_created_at_id",
        "ix_tasks_expired_lease",
        "ix_tasks_idempotency_key_hash",
        "ix_tasks_priority_created_at_id",
        "ix_tasks_status_created_at_id",
    }
    assert "WHERE status = 'IN_PROGRESS'" in _index_sql(indexes["ix_tasks_expired_lease"])
    assert "created_at DESC, id DESC" in _index_sql(indexes["ix_tasks_created_at_id"])
    idempotency_index = indexes["ix_tasks_idempotency_key_hash"]
    assert idempotency_index.unique is True
    assert list(idempotency_index.columns.keys()) == ["idempotency_key_hash"]
    assert "WHERE idempotency_key_hash IS NOT NULL" in _index_sql(idempotency_index)


def test_outbox_columns_cover_publication_lifecycle() -> None:
    """The outbox stores payload, scheduling, outcomes, and publisher fencing."""

    table = cast(Table, OutboxEventModel.__table__)

    assert set(table.c.keys()) == {
        "id",
        "task_id",
        "event_type",
        "payload",
        "message_priority",
        "created_at",
        "available_at",
        "published_at",
        "discarded_at",
        "publish_attempts",
        "publisher_token",
        "lease_expires_at",
        "last_error",
    }
    foreign_key = next(iter(table.c.task_id.foreign_keys))
    assert foreign_key.target_fullname == "tasks.id"
    assert foreign_key.ondelete == "RESTRICT"
    foreign_key_constraint = foreign_key.constraint
    assert foreign_key_constraint is not None
    assert foreign_key_constraint.name == "fk_outbox_events_task_id_tasks"
    assert table.primary_key.name == "pk_outbox_events"
    assert cast(Uuid[UUID], table.c.id.type).as_uuid is True
    assert cast(JSONB, table.c.payload.type).none_as_null is True
    assert table.c.available_at.server_default is not None


def test_outbox_constraints_have_stable_names() -> None:
    """Outbox invariants use predictable names for Alembic operations."""

    table = cast(Table, OutboxEventModel.__table__)

    assert _constraint_names(table.constraints) == {
        "ck_outbox_events_event_type_nonblank",
        "ck_outbox_events_message_priority",
        "ck_outbox_events_payload_object",
        "ck_outbox_events_publish_attempts",
        "ck_outbox_events_publisher_lease",
        "ck_outbox_events_single_outcome",
        "ck_outbox_events_time_order",
    }


def test_outbox_indexes_match_dispatch_and_retention_queries() -> None:
    """Partial indexes exclude handled events from the dispatcher workload."""

    table = cast(Table, OutboxEventModel.__table__)
    indexes = {str(index.name): index for index in table.indexes}

    assert set(indexes) == {
        "ix_outbox_events_discarded_at_id",
        "ix_outbox_events_published_at_id",
        "ix_outbox_events_ready",
        "ix_outbox_events_task_id_created_at",
    }
    ready_sql = _index_sql(indexes["ix_outbox_events_ready"])
    assert "message_priority DESC, available_at, created_at, id" in ready_sql
    assert "WHERE published_at IS NULL AND discarded_at IS NULL" in ready_sql


def test_mappers_configure_without_relationship_side_effects() -> None:
    """Declarative mappings are valid without opening a database connection."""

    configure_mappers()


def test_postgresql_ddl_uses_portable_enums_and_native_storage_types() -> None:
    """DDL uses named checks, JSONB, UUID, and timezone-aware timestamps."""

    task_table = cast(Table, TaskModel.__table__)
    outbox_table = cast(Table, OutboxEventModel.__table__)
    task_ddl = str(CreateTable(task_table).compile(dialect=_POSTGRESQL_DIALECT))
    outbox_ddl = str(CreateTable(outbox_table).compile(dialect=_POSTGRESQL_DIALECT))

    assert "UUID NOT NULL" in task_ddl
    assert "JSONB" in task_ddl
    assert task_ddl.count("BYTEA") == 2
    assert "TIMESTAMP WITH TIME ZONE" in task_ddl
    assert "VARCHAR(8)" in task_ddl
    assert "VARCHAR(16)" in task_ddl
    assert "CREATE TYPE" not in task_ddl
    assert "gen_random_uuid" not in task_ddl
    assert "updated_at" not in task_ddl
    assert "version" not in task_ddl
    assert "worker_id" not in task_ddl
    assert "next_attempt_at" not in task_ddl
    assert "FOREIGN KEY(task_id) REFERENCES tasks (id) ON DELETE RESTRICT" in outbox_ddl
    assert "payload JSONB NOT NULL" in outbox_ddl
