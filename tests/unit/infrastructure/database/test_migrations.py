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
from sqlalchemy.schema import CreateIndex, CreateTable

from cims_task_service.infrastructure.database.base import Base
from cims_task_service.infrastructure.database.models import (
    OutboxEventModel,
    TaskModel,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_CONFIG_PATH = _PROJECT_ROOT / "pyproject.toml"
_DSN_SENTINEL = "must-not-appear-in-migration-output"
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

    assert script.get_bases() == ["db0f59be3ae6"]
    assert len(heads) == 1
    assert revisions
    assert revisions[0].revision == heads[0]
    assert revisions[-1].revision == "db0f59be3ae6"
    assert revisions[-1].down_revision is None
    assert all(revision.is_branch_point is False for revision in revisions)
    assert all(revision.is_merge_point is False for revision in revisions)
    for revision, parent in pairwise(revisions):
        assert revision.down_revision == parent.revision


def test_offline_upgrade_renders_complete_postgresql_schema(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Upgrade SQL mirrors metadata without connecting or exposing credentials."""

    _guard_offline_mode(monkeypatch)
    output = StringIO()

    command.upgrade(_alembic_config(output), "head", sql=True)

    sql = _normalize_sql(output.getvalue())
    captured = capsys.readouterr()
    log_output = "\n".join(record.getMessage() for record in caplog.records)
    assert sql.startswith("BEGIN;")
    assert sql.endswith("COMMIT;")
    assert sql.index("CREATE TABLE tasks") < sql.index("CREATE TABLE outbox_events")
    assert TaskModel.metadata is Base.metadata
    assert OutboxEventModel.metadata is Base.metadata
    for table in Base.metadata.sorted_tables:
        expected_table = _normalize_sql(
            str(CreateTable(table).compile(dialect=_POSTGRESQL_DIALECT))
        )
        assert expected_table in sql
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


def test_offline_downgrade_removes_dependent_schema_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Downgrade SQL removes outbox objects before their task dependency."""

    _guard_offline_mode(monkeypatch)
    output = StringIO()

    command.downgrade(_alembic_config(output), "head:base", sql=True)

    sql = _normalize_sql(output.getvalue())
    assert sql.startswith("BEGIN;")
    assert sql.endswith("COMMIT;")
    assert sql.index("DROP INDEX ix_outbox_events_ready") < sql.index("DROP TABLE outbox_events")
    assert sql.index("DROP TABLE outbox_events") < sql.index("DROP TABLE tasks")
    assert sql.index("DROP INDEX ix_tasks_created_at_id") < sql.index("DROP TABLE tasks")
