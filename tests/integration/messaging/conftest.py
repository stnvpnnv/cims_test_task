"""RabbitMQ publisher routes with resources owned by each test."""

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest_asyncio
from aio_pika import ExchangeType
from aio_pika.abc import AbstractRobustChannel, AbstractRobustExchange, AbstractRobustQueue
from aio_pika.exceptions import ChannelInvalidStateError, ChannelNotFoundEntity

from cims_task_service.infrastructure.messaging.topology import TASK_ROUTING_KEY

_OPERATION_TIMEOUT_SECONDS = 5.0


@pytest_asyncio.fixture
async def rabbitmq_exchange(
    rabbitmq_publisher_channel: AbstractRobustChannel,
) -> AsyncIterator[AbstractRobustExchange]:
    """Declare one uniquely named direct exchange and remove only that exchange."""

    exchange = await rabbitmq_publisher_channel.declare_exchange(
        f"cims.tests.publisher.{uuid4().hex}",
        type=ExchangeType.DIRECT,
        durable=False,
        auto_delete=True,
        timeout=_OPERATION_TIMEOUT_SECONDS,
        robust=False,
    )
    try:
        yield exchange
    finally:
        await _delete_exchange_if_available(exchange, rabbitmq_publisher_channel)


@pytest_asyncio.fixture
async def rabbitmq_task_route(
    rabbitmq_exchange: AbstractRobustExchange,
    rabbitmq_publisher_channel: AbstractRobustChannel,
) -> AsyncIterator[tuple[AbstractRobustExchange, AbstractRobustQueue]]:
    """Bind an isolated classic queue to the task routing key."""

    queue = await rabbitmq_publisher_channel.declare_queue(
        f"{rabbitmq_exchange.name}.queue",
        durable=False,
        exclusive=True,
        auto_delete=True,
        arguments={"x-queue-type": "classic"},
        timeout=_OPERATION_TIMEOUT_SECONDS,
        robust=False,
    )
    try:
        await queue.bind(
            rabbitmq_exchange,
            routing_key=TASK_ROUTING_KEY,
            timeout=_OPERATION_TIMEOUT_SECONDS,
            robust=False,
        )
        yield rabbitmq_exchange, queue
    finally:
        await _delete_queue_if_available(queue, rabbitmq_publisher_channel)


async def _delete_exchange_if_available(
    exchange: AbstractRobustExchange,
    channel: AbstractRobustChannel,
) -> None:
    """Delete an owned exchange unless RabbitMQ already removed its channel."""

    if channel.is_closed:
        return

    try:
        await exchange.delete(
            if_unused=False,
            timeout=_OPERATION_TIMEOUT_SECONDS,
        )
    except (ChannelInvalidStateError, ChannelNotFoundEntity):
        return


async def _delete_queue_if_available(
    queue: AbstractRobustQueue,
    channel: AbstractRobustChannel,
) -> None:
    """Delete an owned queue unless its exclusive connection already did so."""

    if channel.is_closed:
        return

    try:
        await queue.delete(
            if_unused=False,
            if_empty=False,
            timeout=_OPERATION_TIMEOUT_SECONDS,
        )
    except (ChannelInvalidStateError, ChannelNotFoundEntity):
        return
