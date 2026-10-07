"""Factories for robust RabbitMQ connections."""

from aio_pika import RobustConnection
from aio_pika.abc import AbstractRobustConnection
from aio_pika.connection import make_url

from cims_task_service.config import Settings


async def connect_rabbitmq(settings: Settings) -> AbstractRobustConnection:
    """Open a robust connection using the configured startup policy."""

    connection = RobustConnection(
        make_url(settings.rabbitmq_url.get_secret_value()),
        reconnect_interval=settings.rabbitmq_reconnect_interval_seconds,
        fail_fast=True,
    )
    try:
        await connection.connect(
            timeout=settings.rabbitmq_connection_timeout_seconds,
        )
    except BaseException:
        await connection.close()
        raise

    return connection


async def close_rabbitmq_connection(
    connection: AbstractRobustConnection,
) -> None:
    """Close the connection and stop its background reconnect task."""

    await connection.close()
