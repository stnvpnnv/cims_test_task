"""Factories for robust RabbitMQ connections."""

from collections.abc import Awaitable
from typing import Protocol, cast

from aio_pika import connect_robust
from aio_pika.abc import AbstractRobustConnection

from cims_task_service.config import Settings


class _RobustConnector(Protocol):
    """Typed view of options absent from aio-pika's overload declarations."""

    def __call__(
        self,
        url: str,
        *,
        timeout: float,
        reconnect_interval: float,
        fail_fast: bool,
    ) -> Awaitable[AbstractRobustConnection]: ...


async def connect_rabbitmq(settings: Settings) -> AbstractRobustConnection:
    """Open a robust connection using the configured startup policy."""

    connector = cast(_RobustConnector, connect_robust)
    return await connector(
        settings.rabbitmq_url.get_secret_value(),
        timeout=settings.rabbitmq_connection_timeout_seconds,
        reconnect_interval=settings.rabbitmq_reconnect_interval_seconds,
        fail_fast=True,
    )


async def close_rabbitmq_connection(
    connection: AbstractRobustConnection,
) -> None:
    """Close the connection and stop its background reconnect task."""

    await connection.close()
