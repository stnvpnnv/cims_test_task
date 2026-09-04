"""Offline tests for the Alembic revision history."""

import re
from io import StringIO
from itertools import pairwise
from pathlib import Path
from typing import Never

import pytest
import sqlalchemy.ext.asyncio as sqlalchemy_asyncio
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy.dialects.postgresql.base import PGDialect
from sqlalchemy.schema import CreateIndex

from cims_task_service.infrastructure.database.base import Base
from cims_task_service.infrastructure.database.models import (
    OutboxEventModel,
    TaskModel,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_CONFIG_PATH = _PROJECT_ROOT / "pyproject.toml"
_DSN_SENTINEL = "must-not-appear-in-migration-output"
_BASE_REVISION = "db0f59be3ae6"
_IDEMPOTENCY_REVISION = "61d66cf078b9"
_POSTGRESQL_DIALECT = PGDialect()  # type: ignore[no-untyped-call]


def _alembic_config(output: StringIO | None = None) -> Config:
    stream = output if output is not None else StringIO()
    return Config(
        toml_file=_CONFIG_PATH,
        stdout=stream,
        output_buffer=stream,
    )


def _reject_engine_creation(*_args: object, **_kwargs: object) -> Never:
    message = "offline migrations must not open database connections"
    raise AssertionError(message)


def _normalize_sql(sql: str) -> str:
    normalized = " ".join(sql.split())
    normalized = re.sub(r"\(\s+", "(", normalized)
    return re.sub(r"\s+\)", ")", normalized)


def _guard_offline_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "CIMS_DATABASE_URL",
        f"postgresql+asyncpg://service:{_DSN_SENTINEL}@127.0.0.1:1/cims",
    )
    monkeypatch.setattr(
        sqlalchemy_asyncio,
        "create_async_engine",
        _reject_engine_creation,
    )
    monkeypatch.setattr("asyncpg.connect", _reject_engine_creation)


def test_revision_history_has_one_base_and_a_single_linear_head() -> None:
    """The migration graph can grow without introducing branches or extra bases."""

    script = ScriptDirectory.from_config(_alembic_config())
    revisions = list(script.walk_revisions())
    heads = script.get_heads()

    assert script.get_bases() == [_BASE_REVISION]
    assert len(heads) == 1
    assert heads == [_IDEMPOTENCY_REVISION]
    assert revisions
    assert revisions[0].revision == heads[0]
    assert revisions[-1].revision == _BASE_REVISION
    assert revisions[-1].down_revision is None
    assert all(revision.is_branch_point is False for revision in revisions)
    assert all(revision.is_merge_point is False for revision in revisions)
    for revision, parent in pairwise(revisions):
        assert revision.down_revision == parent.revision


def test_offline_upgrade_renders_complete_migration_chain(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The full upgrade stays offline, ordered, and free of credentials."""

    _guard_offline_mode(monkeypatch)
    output = StringIO()

    command.upgrade(_alembic_config(output), "head", sql=True)

    sql = _normalize_sql(output.getvalue())
    captured = capsys.readouterr()
    log_output = "\n".join(record.getMessage() for record in caplog.records)
    assert sql.startswith("BEGIN;")
    assert sql.endswith("COMMIT;")
    assert sql.index("CREATE TABLE tasks") < sql.index("CREATE TABLE outbox_events")
    assert sql.index("CREATE TABLE outbox_events") < sql.index(
        "ALTER TABLE tasks ADD COLUMN idempotency_key_hash BYTEA"
    )
    assert TaskModel.metadata is Base.metadata
    assert OutboxEventModel.metadata is Base.metadata
    for table in Base.metadata.sorted_tables:
        for index in table.indexes:
            expected_index = _normalize_sql(
                str(CreateIndex(index).compile(dialect=_POSTGRESQL_DIALECT))
            )
            assert expected_index in sql
    assert "CREATE TYPE" not in sql
    assert "gen_random_uuid" not in sql
    assert _DSN_SENTINEL not in sql
    assert _DSN_SENTINEL not in captured.out
    assert _DSN_SENTINEL not in captured.err
    assert _DSN_SENTINEL not in log_output


def test_offline_idempotency_upgrade_adds_bounded_unique_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The additive revision creates paired hashes before their unique index."""

    _guard_offline_mode(monkeypatch)
    output = StringIO()

    command.upgrade(
        _alembic_config(output),
        f"{_BASE_REVISION}:{_IDEMPOTENCY_REVISION}",
        sql=True,
    )

    sql = _normalize_sql(output.getvalue())
    key_column = "ALTER TABLE tasks ADD COLUMN idempotency_key_hash BYTEA;"
    fingerprint_column = "ALTER TABLE tasks ADD COLUMN request_fingerprint BYTEA;"
    pair_constraint = (
        "ALTER TABLE tasks ADD CONSTRAINT ck_tasks_idempotency_pair "
        "CHECK ((idempotency_key_hash IS NULL) = (request_fingerprint IS NULL));"
    )
    lengths_constraint = (
        "ALTER TABLE tasks ADD CONSTRAINT ck_tasks_idempotency_hash_lengths "
        "CHECK ((idempotency_key_hash IS NULL "
        "OR octet_length(idempotency_key_hash) = 32) "
        "AND (request_fingerprint IS NULL "
        "OR octet_length(request_fingerprint) = 32));"
    )
    unique_index = (
        "CREATE UNIQUE INDEX ix_tasks_idempotency_key_hash "
        "ON tasks (idempotency_key_hash) WHERE idempotency_key_hash IS NOT NULL;"
    )

    assert sql.startswith("BEGIN;")
    assert sql.endswith("COMMIT;")
    assert "CREATE TABLE" not in sql
    assert sql.index(key_column) < sql.index(fingerprint_column)
    assert sql.index(fingerprint_column) < sql.index(pair_constraint)
    assert sql.index(pair_constraint) < sql.index(lengths_constraint)
    assert sql.index(lengths_constraint) < sql.index(unique_index)
    assert "ADD COLUMN idempotency_key_hash BYTEA NOT NULL" not in sql
    assert "ADD COLUMN request_fingerprint BYTEA NOT NULL" not in sql


def test_offline_idempotency_downgrade_removes_dependants_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The additive revision drops indexes and checks before their columns."""

    _guard_offline_mode(monkeypatch)
    output = StringIO()

    command.downgrade(
        _alembic_config(output),
        f"{_IDEMPOTENCY_REVISION}:{_BASE_REVISION}",
        sql=True,
    )

    sql = _normalize_sql(output.getvalue())
    drop_index = "DROP INDEX ix_tasks_idempotency_key_hash;"
    drop_lengths = "ALTER TABLE tasks DROP CONSTRAINT ck_tasks_idempotency_hash_lengths;"
    drop_pair = "ALTER TABLE tasks DROP CONSTRAINT ck_tasks_idempotency_pair;"
    drop_fingerprint = "ALTER TABLE tasks DROP COLUMN request_fingerprint;"
    drop_key = "ALTER TABLE tasks DROP COLUMN idempotency_key_hash;"

    assert sql.startswith("BEGIN;")
    assert sql.endswith("COMMIT;")
    assert "DROP TABLE" not in sql
    assert sql.index(drop_index) < sql.index(drop_lengths)
    assert sql.index(drop_lengths) < sql.index(drop_pair)
    assert sql.index(drop_pair) < sql.index(drop_fingerprint)
    assert sql.index(drop_fingerprint) < sql.index(drop_key)


def test_offline_base_downgrade_removes_dependent_schema_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The base downgrade removes outbox objects before their task dependency."""

    _guard_offline_mode(monkeypatch)
    output = StringIO()

    command.downgrade(
        _alembic_config(output),
        f"{_BASE_REVISION}:base",
        sql=True,
    )

    sql = _normalize_sql(output.getvalue())
    assert sql.startswith("BEGIN;")
    assert sql.endswith("COMMIT;")
    assert sql.index("DROP INDEX ix_outbox_events_ready") < sql.index("DROP TABLE outbox_events")
    assert sql.index("DROP TABLE outbox_events") < sql.index("DROP TABLE tasks")
    assert sql.index("DROP INDEX ix_tasks_created_at_id") < sql.index("DROP TABLE tasks")
