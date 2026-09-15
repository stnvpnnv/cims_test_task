"""Tests for environment-backed service configuration."""

import pytest
from pydantic import SecretStr, ValidationError

from cims_task_service.application.task_dispatcher import MAX_DISPATCH_BATCH_SIZE
from cims_task_service.config import DispatcherSettings, Settings

_DISPATCHER_DURATION_ENVIRONMENT_VARIABLES = (
    "CIMS_DISPATCHER_POLL_INTERVAL_SECONDS",
    "CIMS_DISPATCHER_LEASE_DURATION_SECONDS",
    "CIMS_DISPATCHER_RETRY_INITIAL_DELAY_SECONDS",
    "CIMS_DISPATCHER_RETRY_MAXIMUM_DELAY_SECONDS",
    "CIMS_RABBITMQ_PUBLISH_TIMEOUT_SECONDS",
    "CIMS_DISPATCHER_SHUTDOWN_GRACE_SECONDS",
)

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
    "CIMS_DISPATCHER_BATCH_SIZE",
    *_DISPATCHER_DURATION_ENVIRONMENT_VARIABLES,
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


def test_dispatcher_settings_have_coherent_defaults() -> None:
    """Defaults leave operational headroom for publishing and finalization."""

    settings = DispatcherSettings()

    assert settings.dispatcher_batch_size == 10
    assert settings.dispatcher_poll_interval_seconds == 0.5
    assert settings.dispatcher_lease_duration_seconds == 60.0
    assert settings.dispatcher_retry_initial_delay_seconds == 1.0
    assert settings.dispatcher_retry_maximum_delay_seconds == 60.0
    assert settings.rabbitmq_publish_timeout_seconds == 10.0
    assert settings.dispatcher_shutdown_grace_seconds == 45.0


def test_dispatcher_settings_load_environment_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dedicated process receives typed deployment-specific values."""

    monkeypatch.setenv("CIMS_DISPATCHER_BATCH_SIZE", "12")
    monkeypatch.setenv("CIMS_DISPATCHER_POLL_INTERVAL_SECONDS", "0.25")
    monkeypatch.setenv("CIMS_DISPATCHER_LEASE_DURATION_SECONDS", "45")
    monkeypatch.setenv("CIMS_DISPATCHER_RETRY_INITIAL_DELAY_SECONDS", "1.5")
    monkeypatch.setenv("CIMS_DISPATCHER_RETRY_MAXIMUM_DELAY_SECONDS", "90")
    monkeypatch.setenv("CIMS_RABBITMQ_PUBLISH_TIMEOUT_SECONDS", "12")
    monkeypatch.setenv("CIMS_DISPATCHER_SHUTDOWN_GRACE_SECONDS", "20")

    settings = DispatcherSettings()

    assert settings.dispatcher_batch_size == 12
    assert settings.dispatcher_poll_interval_seconds == 0.25
    assert settings.dispatcher_lease_duration_seconds == 45.0
    assert settings.dispatcher_retry_initial_delay_seconds == 1.5
    assert settings.dispatcher_retry_maximum_delay_seconds == 90.0
    assert settings.rabbitmq_publish_timeout_seconds == 12.0
    assert settings.dispatcher_shutdown_grace_seconds == 20.0


@pytest.mark.parametrize(
    ("variable_name", "attribute_name"),
    [
        ("CIMS_DISPATCHER_BATCH_SIZE", "dispatcher_batch_size"),
        (
            "CIMS_DISPATCHER_POLL_INTERVAL_SECONDS",
            "dispatcher_poll_interval_seconds",
        ),
        (
            "CIMS_DISPATCHER_LEASE_DURATION_SECONDS",
            "dispatcher_lease_duration_seconds",
        ),
        (
            "CIMS_DISPATCHER_RETRY_INITIAL_DELAY_SECONDS",
            "dispatcher_retry_initial_delay_seconds",
        ),
        (
            "CIMS_DISPATCHER_RETRY_MAXIMUM_DELAY_SECONDS",
            "dispatcher_retry_maximum_delay_seconds",
        ),
        (
            "CIMS_RABBITMQ_PUBLISH_TIMEOUT_SECONDS",
            "rabbitmq_publish_timeout_seconds",
        ),
        (
            "CIMS_DISPATCHER_SHUTDOWN_GRACE_SECONDS",
            "dispatcher_shutdown_grace_seconds",
        ),
    ],
)
def test_base_settings_ignore_dispatcher_only_environment(
    monkeypatch: pytest.MonkeyPatch,
    variable_name: str,
    attribute_name: str,
) -> None:
    """An API process is not rejected by settings it never consumes."""

    monkeypatch.setenv(variable_name, "not-a-valid-value")

    settings = Settings()

    assert not hasattr(settings, attribute_name)


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


@pytest.mark.parametrize("batch_size", [1, MAX_DISPATCH_BATCH_SIZE])
def test_dispatcher_batch_size_accepts_core_boundaries(batch_size: int) -> None:
    """Configuration and dispatcher resource limits remain synchronized."""

    settings = DispatcherSettings(
        dispatcher_batch_size=batch_size,
        database_pool_size=batch_size,
        database_max_overflow=0,
    )

    assert settings.dispatcher_batch_size == batch_size


@pytest.mark.parametrize(
    "invalid_batch_size",
    ["0", "-1", "1.5", str(MAX_DISPATCH_BATCH_SIZE + 1)],
)
def test_dispatcher_batch_size_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    invalid_batch_size: str,
) -> None:
    """Invalid or resource-amplifying batch values fail during startup."""

    monkeypatch.setenv("CIMS_DISPATCHER_BATCH_SIZE", invalid_batch_size)
    monkeypatch.setenv(
        "CIMS_DATABASE_POOL_SIZE",
        str(MAX_DISPATCH_BATCH_SIZE + 1),
    )

    with pytest.raises(ValidationError):
        DispatcherSettings()


@pytest.mark.parametrize("variable_name", _DISPATCHER_DURATION_ENVIRONMENT_VARIABLES)
@pytest.mark.parametrize("invalid_value", ["0", "-0.1"])
def test_dispatcher_durations_must_be_positive(
    monkeypatch: pytest.MonkeyPatch,
    variable_name: str,
    invalid_value: str,
) -> None:
    """Zero and negative durations cannot reach runtime timeout primitives."""

    monkeypatch.setenv(variable_name, invalid_value)

    with pytest.raises(ValidationError):
        DispatcherSettings()


@pytest.mark.parametrize("variable_name", _DISPATCHER_DURATION_ENVIRONMENT_VARIABLES)
@pytest.mark.parametrize("invalid_value", ["nan", "inf", "-inf"])
def test_dispatcher_durations_must_be_finite(
    monkeypatch: pytest.MonkeyPatch,
    variable_name: str,
    invalid_value: str,
) -> None:
    """IEEE special values are rejected before timeout calculations."""

    monkeypatch.setenv(variable_name, invalid_value)

    with pytest.raises(ValidationError):
        DispatcherSettings()


def test_dispatcher_retry_maximum_must_cover_the_initial_delay() -> None:
    """A capped backoff cannot start above its own maximum."""

    with pytest.raises(
        ValidationError,
        match="dispatcher retry maximum delay must be at least its initial delay",
    ):
        DispatcherSettings(
            dispatcher_retry_initial_delay_seconds=2.0,
            dispatcher_retry_maximum_delay_seconds=1.0,
        )


def test_dispatcher_retry_delay_allows_a_fixed_cap() -> None:
    """Equal initial and maximum values intentionally produce jitter-only retry."""

    settings = DispatcherSettings(
        dispatcher_retry_initial_delay_seconds=2.0,
        dispatcher_retry_maximum_delay_seconds=2.0,
    )

    assert settings.dispatcher_retry_maximum_delay_seconds == 2.0


def test_dispatcher_batch_must_fit_the_database_pool() -> None:
    """Every event in a finalization wave must be able to acquire a connection."""

    with pytest.raises(
        ValidationError,
        match="dispatcher batch size must not exceed database pool capacity",
    ):
        DispatcherSettings(
            dispatcher_batch_size=3,
            database_pool_size=1,
            database_max_overflow=1,
        )


@pytest.mark.parametrize("lease_duration", [40.0, 39.9, 40.0000004])
def test_dispatcher_lease_must_exceed_publish_and_pool_timeouts(
    lease_duration: float,
) -> None:
    """A normal publish and connection wait cannot guarantee an expired lease."""

    with pytest.raises(
        ValidationError,
        match="dispatcher lease duration must exceed the publish and database pool timeouts",
    ):
        DispatcherSettings(
            dispatcher_lease_duration_seconds=lease_duration,
            rabbitmq_publish_timeout_seconds=10.0,
            database_pool_timeout_seconds=30.0,
        )


def test_dispatcher_lease_accepts_a_duration_above_required_timeouts() -> None:
    """Strictly greater lease duration satisfies the finalization headroom rule."""

    settings = DispatcherSettings(
        dispatcher_lease_duration_seconds=40.000001,
        rabbitmq_publish_timeout_seconds=10.0,
        database_pool_timeout_seconds=30.0,
    )

    assert settings.dispatcher_lease_duration_seconds == 40.000001


@pytest.mark.parametrize(
    "field_name",
    [
        "dispatcher_lease_duration_seconds",
        "dispatcher_retry_initial_delay_seconds",
        "dispatcher_retry_maximum_delay_seconds",
    ],
)
def test_dispatcher_rejects_durations_that_round_to_zero(
    field_name: str,
) -> None:
    """Validated durations remain positive after conversion to timedelta."""

    with pytest.raises(
        ValidationError,
        match=("dispatcher lease and retry durations must resolve to at least one microsecond"),
    ):
        DispatcherSettings.model_validate({field_name: 1e-10})


@pytest.mark.parametrize(
    "overrides",
    [
        {"dispatcher_lease_duration_seconds": 1e308},
        {"dispatcher_retry_initial_delay_seconds": 1e308},
        {"dispatcher_retry_maximum_delay_seconds": 1e308},
        {"rabbitmq_publish_timeout_seconds": 1e308},
        {"database_pool_timeout_seconds": 1e308},
    ],
)
def test_dispatcher_rejects_durations_outside_timedelta_range(
    overrides: dict[str, float],
) -> None:
    """Finite deployment values must still be representable by runtime types."""

    with pytest.raises(
        ValidationError,
        match="dispatcher durations must fit within Python timedelta range",
    ):
        DispatcherSettings.model_validate(overrides)
