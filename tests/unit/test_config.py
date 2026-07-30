"""Tests for environment-backed service configuration."""

import pytest
from pydantic import SecretStr, ValidationError

from cims_task_service.config import Settings

_DATABASE_ENVIRONMENT_VARIABLES = (
    "CIMS_DATABASE_URL",
    "CIMS_DATABASE_POOL_SIZE",
    "CIMS_DATABASE_MAX_OVERFLOW",
    "CIMS_DATABASE_POOL_TIMEOUT_SECONDS",
    "CIMS_DATABASE_POOL_RECYCLE_SECONDS",
)


def test_database_settings_have_safe_local_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defaults are usable locally without embedding a password."""

    for variable_name in _DATABASE_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable_name, raising=False)

    settings = Settings()

    assert settings.database_url.get_secret_value() == (
        "postgresql+asyncpg://cims@localhost:5432/cims"
    )
    assert settings.database_pool_size == 5
    assert settings.database_max_overflow == 10
    assert settings.database_pool_timeout_seconds == 30.0
    assert settings.database_pool_recycle_seconds == 1800


def test_database_settings_load_environment_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deployment values are read through the project-specific prefix."""

    monkeypatch.setenv(
        "CIMS_DATABASE_URL",
        "postgresql+asyncpg://service@postgres:5432/tasks",
    )
    monkeypatch.setenv("CIMS_DATABASE_POOL_SIZE", "7")
    monkeypatch.setenv("CIMS_DATABASE_MAX_OVERFLOW", "3")
    monkeypatch.setenv("CIMS_DATABASE_POOL_TIMEOUT_SECONDS", "11.5")
    monkeypatch.setenv("CIMS_DATABASE_POOL_RECYCLE_SECONDS", "600")

    settings = Settings()

    assert settings.database_url.get_secret_value() == (
        "postgresql+asyncpg://service@postgres:5432/tasks"
    )
    assert settings.database_pool_size == 7
    assert settings.database_max_overflow == 3
    assert settings.database_pool_timeout_seconds == 11.5
    assert settings.database_pool_recycle_seconds == 600


def test_database_settings_reject_non_asyncpg_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A synchronous PostgreSQL URL cannot reach the async engine factory."""

    monkeypatch.setenv(
        "CIMS_DATABASE_URL",
        "postgresql://service:do-not-log@postgres:5432/tasks",
    )

    with pytest.raises(
        ValidationError,
        match=r"postgresql\+asyncpg",
    ) as error_info:
        Settings()

    assert "do-not-log" not in str(error_info.value)


def test_database_settings_reject_malformed_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed deployment input produces a configuration error."""

    monkeypatch.setenv("CIMS_DATABASE_URL", "://")

    with pytest.raises(
        ValidationError,
        match="valid SQLAlchemy URL",
    ):
        Settings()


def test_database_settings_require_host_and_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Incomplete URLs are rejected even when their driver is valid."""

    monkeypatch.setenv(
        "CIMS_DATABASE_URL",
        "postgresql+asyncpg://postgres",
    )

    with pytest.raises(
        ValidationError,
        match="host and database name",
    ):
        Settings()


def test_database_url_is_hidden_from_settings_representation() -> None:
    """Credentials cannot leak through routine settings logging."""

    settings = Settings(
        database_url=SecretStr("postgresql+asyncpg://service:do-not-log@postgres:5432/tasks")
    )

    assert "do-not-log" not in repr(settings)
    assert "**********" in repr(settings)


@pytest.mark.parametrize(
    ("variable_name", "invalid_value"),
    [
        ("CIMS_DATABASE_POOL_SIZE", "0"),
        ("CIMS_DATABASE_MAX_OVERFLOW", "-1"),
        ("CIMS_DATABASE_POOL_TIMEOUT_SECONDS", "0"),
        ("CIMS_DATABASE_POOL_RECYCLE_SECONDS", "0"),
    ],
)
def test_database_settings_reject_invalid_pool_values(
    monkeypatch: pytest.MonkeyPatch,
    variable_name: str,
    invalid_value: str,
) -> None:
    """Invalid pool bounds fail during startup rather than under load."""

    monkeypatch.setenv(variable_name, invalid_value)

    with pytest.raises(ValidationError):
        Settings()
