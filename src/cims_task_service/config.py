"""Runtime configuration loaded from environment variables."""

from typing import Annotated

from pydantic import (
    AnyUrl,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    SecretStr,
    TypeAdapter,
    UrlConstraints,
    ValidationError,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

type _AmqpUrl = Annotated[
    AnyUrl,
    UrlConstraints(allowed_schemes=["amqp", "amqps"], host_required=True),
]

_AMQP_URL_ADAPTER: TypeAdapter[_AmqpUrl] = TypeAdapter(_AmqpUrl)


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
    rabbitmq_url: SecretStr = SecretStr("amqp://cims@localhost:5672/cims")

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
