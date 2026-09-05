"""Tests for environment-backed service configuration."""

import pytest
from pydantic import SecretStr, ValidationError

from cims_task_service.config import Settings

_SETTINGS_ENVIRONMENT_VARIABLES = (
    "CIMS_DEBUG",
    "CIMS_DATABASE_URL",
    "CIMS_DATABASE_POOL_SIZE",
    "CIMS_DATABASE_MAX_OVERFLOW",
    "CIMS_DATABASE_POOL_TIMEOUT_SECONDS",
    "CIMS_DATABASE_POOL_RECYCLE_SECONDS",
    "CIMS_TASK_MAX_ATTEMPTS",
    "CIMS_RABBITMQ_URL",
    "CIMS_RABBITMQ_CONNECTION_TIMEOUT_SECONDS",
    "CIMS_RABBITMQ_RECONNECT_INTERVAL_SECONDS",
)


@pytest.fixture(autouse=True)
def clear_settings_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep configuration tests independent from the developer environment."""

    for variable_name in _SETTINGS_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable_name, raising=False)


def test_settings_have_safe_non_secret_defaults() -> None:
    """Defaults describe local services without embedding passwords."""

    settings = Settings()

    assert settings.database_url.get_secret_value() == (
        "postgresql+asyncpg://cims@localhost:5432/cims"
    )
    assert settings.database_pool_size == 5
    assert settings.database_max_overflow == 10
    assert settings.database_pool_timeout_seconds == 30.0
    assert settings.database_pool_recycle_seconds == 1800
    assert settings.task_max_attempts == 3
    assert settings.rabbitmq_url.get_secret_value() == ("amqp://cims@localhost:5672/cims")
    assert settings.rabbitmq_connection_timeout_seconds == 10.0
    assert settings.rabbitmq_reconnect_interval_seconds == 5.0


def test_settings_load_environment_overrides(
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
    monkeypatch.setenv("CIMS_TASK_MAX_ATTEMPTS", "5")
    monkeypatch.setenv(
        "CIMS_RABBITMQ_URL",
        "amqp://service:rabbit-secret@rabbitmq:5672/tasks",
    )
    monkeypatch.setenv("CIMS_RABBITMQ_CONNECTION_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("CIMS_RABBITMQ_RECONNECT_INTERVAL_SECONDS", "2.5")

    settings = Settings()

    assert settings.database_url.get_secret_value() == (
        "postgresql+asyncpg://service@postgres:5432/tasks"
    )
    assert settings.database_pool_size == 7
    assert settings.database_max_overflow == 3
    assert settings.database_pool_timeout_seconds == 11.5
    assert settings.database_pool_recycle_seconds == 600
    assert settings.task_max_attempts == 5
    assert settings.rabbitmq_url.get_secret_value() == (
        "amqp://service:rabbit-secret@rabbitmq:5672/tasks"
    )
    assert settings.rabbitmq_connection_timeout_seconds == 12.5
    assert settings.rabbitmq_reconnect_interval_seconds == 2.5


def test_rabbitmq_settings_accept_tls_and_encoded_vhost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TLS URLs and percent-encoded virtual hosts remain valid inputs."""

    rabbitmq_url = "amqps://service:rabbit-secret@rabbitmq/%2Ftenant"
    monkeypatch.setenv("CIMS_RABBITMQ_URL", rabbitmq_url)

    settings = Settings()

    assert settings.rabbitmq_url.get_secret_value() == rabbitmq_url


def test_rabbitmq_settings_store_the_normalized_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runtime client receives the same canonical URL that was validated."""

    monkeypatch.setenv(
        "CIMS_RABBITMQ_URL",
        "amqp://service:rabbit-secret@rabbitmq/tasks ",
    )

    settings = Settings()

    assert settings.rabbitmq_url.get_secret_value() == (
        "amqp://service:rabbit-secret@rabbitmq/tasks"
    )


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


def test_connection_urls_are_hidden_from_settings_representation() -> None:
    """Database and broker credentials cannot leak through settings logging."""

    settings = Settings(
        database_url=SecretStr(
            "postgresql+asyncpg://service:database-do-not-log@postgres:5432/tasks"
        ),
        rabbitmq_url=SecretStr("amqp://service:rabbitmq-do-not-log@rabbitmq:5672/tasks"),
    )

    assert "database-do-not-log" not in repr(settings)
    assert "rabbitmq-do-not-log" not in repr(settings)
    assert "**********" in repr(settings)


@pytest.mark.parametrize(
    "rabbitmq_url",
    [
        "http://rabbitmq/cims",
        "://",
        "amqp:///cims",
        "amqp://rabbitmq:not-a-port/cims",
    ],
)
def test_rabbitmq_settings_reject_invalid_urls_without_leaking_credentials(
    monkeypatch: pytest.MonkeyPatch,
    rabbitmq_url: str,
) -> None:
    """Invalid broker URLs fail safely before a connection is attempted."""

    sentinel = "rabbit-secret"
    monkeypatch.setenv(
        "CIMS_RABBITMQ_URL",
        rabbitmq_url.replace("rabbitmq", f"service:{sentinel}@rabbitmq"),
    )

    with pytest.raises(
        ValidationError,
        match="RabbitMQ URL must be a valid amqp or amqps URL with a host",
    ) as error_info:
        Settings()

    assert sentinel not in str(error_info.value)


@pytest.mark.parametrize(
    ("variable_name", "invalid_value"),
    [
        ("CIMS_DATABASE_POOL_SIZE", "0"),
        ("CIMS_DATABASE_MAX_OVERFLOW", "-1"),
        ("CIMS_DATABASE_POOL_TIMEOUT_SECONDS", "0"),
        ("CIMS_DATABASE_POOL_RECYCLE_SECONDS", "0"),
        ("CIMS_TASK_MAX_ATTEMPTS", "0"),
        ("CIMS_RABBITMQ_CONNECTION_TIMEOUT_SECONDS", "0"),
        ("CIMS_RABBITMQ_CONNECTION_TIMEOUT_SECONDS", "-0.1"),
        ("CIMS_RABBITMQ_RECONNECT_INTERVAL_SECONDS", "0"),
        ("CIMS_RABBITMQ_RECONNECT_INTERVAL_SECONDS", "-0.1"),
    ],
)
def test_settings_reject_invalid_operational_values(
    monkeypatch: pytest.MonkeyPatch,
    variable_name: str,
    invalid_value: str,
) -> None:
    """Invalid operational bounds fail during startup rather than under load."""

    monkeypatch.setenv(variable_name, invalid_value)

    with pytest.raises(ValidationError):
        Settings()
