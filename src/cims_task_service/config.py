"""Runtime configuration loaded from environment variables."""

from datetime import timedelta
from typing import Annotated, Final, Self

from pydantic import (
    AnyUrl,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    SecretStr,
    TypeAdapter,
    UrlConstraints,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

type _AmqpUrl = Annotated[
    AnyUrl,
    UrlConstraints(allowed_schemes=["amqp", "amqps"], host_required=True),
]

_AMQP_URL_ADAPTER: TypeAdapter[_AmqpUrl] = TypeAdapter(_AmqpUrl)
MAX_EXECUTION_RETRY_DELAY_SECONDS: Final = 86_400.0


class Settings(BaseSettings):
    """Service settings with a project-specific environment prefix."""

    model_config = SettingsConfigDict(
        allow_inf_nan=False,
        env_prefix="CIMS_",
        frozen=True,
        hide_input_in_errors=True,
        validate_default=True,
    )

    debug: bool = False
    database_url: SecretStr = SecretStr("postgresql+asyncpg://cims@localhost:5432/cims")
    database_pool_size: PositiveInt = 5
    database_max_overflow: NonNegativeInt = 10
    database_pool_timeout_seconds: PositiveFloat = 30.0
    database_pool_recycle_seconds: PositiveInt = 1800
    task_max_attempts: PositiveInt = 3
    rabbitmq_url: SecretStr = SecretStr("amqp://cims@localhost:5672/cims")
    rabbitmq_connection_timeout_seconds: PositiveFloat = 10.0
    rabbitmq_reconnect_interval_seconds: PositiveFloat = 5.0

    @field_validator("database_url")
    @classmethod
    def require_asyncpg_driver(cls, value: SecretStr) -> SecretStr:
        """Reject URLs for drivers not installed by this service."""

        try:
            database_url = make_url(value.get_secret_value())
        except (ArgumentError, ValueError) as error:
            message = "database URL is not a valid SQLAlchemy URL"
            raise ValueError(message) from error

        if database_url.drivername != "postgresql+asyncpg":
            message = "database URL must use the postgresql+asyncpg scheme"
            raise ValueError(message)
        if not database_url.host or not database_url.database:
            message = "database URL must include a host and database name"
            raise ValueError(message)
        return value

    @field_validator("rabbitmq_url")
    @classmethod
    def require_amqp_url(cls, value: SecretStr) -> SecretStr:
        """Reject broker URLs that aio-pika cannot use."""

        try:
            rabbitmq_url = _AMQP_URL_ADAPTER.validate_python(value.get_secret_value())
        except ValidationError:
            message = "RabbitMQ URL must be a valid amqp or amqps URL with a host"
            raise ValueError(message) from None
        return SecretStr(str(rabbitmq_url))


class DispatcherSettings(Settings):
    """Settings used only by the independently deployed background process."""

    dispatcher_batch_size: Annotated[int, Field(ge=1, le=100)] = 10
    dispatcher_poll_interval_seconds: PositiveFloat = 0.5
    dispatcher_lease_duration_seconds: PositiveFloat = 60.0
    dispatcher_retry_initial_delay_seconds: PositiveFloat = 1.0
    dispatcher_retry_maximum_delay_seconds: PositiveFloat = 60.0
    rabbitmq_publish_timeout_seconds: PositiveFloat = 10.0
    dispatcher_shutdown_grace_seconds: PositiveFloat = 45.0
    execution_recovery_batch_size: Annotated[int, Field(ge=1, le=100)] = 10
    execution_recovery_poll_interval_seconds: PositiveFloat = 5.0
    execution_retry_initial_delay_seconds: Annotated[
        float,
        Field(gt=0, le=MAX_EXECUTION_RETRY_DELAY_SECONDS),
    ] = 5.0
    execution_retry_maximum_delay_seconds: Annotated[
        float,
        Field(gt=0, le=MAX_EXECUTION_RETRY_DELAY_SECONDS),
    ] = 300.0

    @model_validator(mode="after")
    def require_coherent_dispatcher_policy(self) -> Self:
        """Reject invalid dispatcher resource and timer relationships."""

        try:
            retry_initial_delay = timedelta(seconds=self.dispatcher_retry_initial_delay_seconds)
            retry_maximum_delay = timedelta(seconds=self.dispatcher_retry_maximum_delay_seconds)
            lease_duration = timedelta(seconds=self.dispatcher_lease_duration_seconds)
            required_lease_duration = timedelta(
                seconds=(self.rabbitmq_publish_timeout_seconds + self.database_pool_timeout_seconds)
            )
        except OverflowError as error:
            message = "dispatcher durations must fit within Python timedelta range"
            raise ValueError(message) from error

        if (
            retry_initial_delay <= timedelta(0)
            or retry_maximum_delay <= timedelta(0)
            or lease_duration <= timedelta(0)
        ):
            message = (
                "dispatcher lease and retry durations must resolve to at least one microsecond"
            )
            raise ValueError(message)

        if retry_maximum_delay < retry_initial_delay:
            message = "dispatcher retry maximum delay must be at least its initial delay"
            raise ValueError(message)

        database_pool_capacity = self.database_pool_size + self.database_max_overflow
        if self.dispatcher_batch_size > database_pool_capacity:
            message = "dispatcher batch size must not exceed database pool capacity"
            raise ValueError(message)

        if lease_duration <= required_lease_duration:
            message = "dispatcher lease duration must exceed the publish and database pool timeouts"
            raise ValueError(message)

        return self

    @model_validator(mode="after")
    def require_coherent_execution_recovery_policy(self) -> Self:
        """Reject execution retry delays that cannot form a safe schedule."""

        retry_initial_delay = timedelta(seconds=self.execution_retry_initial_delay_seconds)
        retry_maximum_delay = timedelta(seconds=self.execution_retry_maximum_delay_seconds)

        if retry_initial_delay <= timedelta(0) or retry_maximum_delay <= timedelta(0):
            message = "execution retry delays must resolve to at least one microsecond"
            raise ValueError(message)

        if retry_maximum_delay < retry_initial_delay:
            message = "execution retry maximum delay must be at least its initial delay"
            raise ValueError(message)

        return self
