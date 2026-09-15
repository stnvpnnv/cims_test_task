"""Opt-in RabbitMQ fixtures with resources owned by each test."""

import asyncio
import os
from collections.abc import AsyncIterator
from urllib.parse import unquote_to_bytes, urlsplit
from uuid import uuid4

import pytest
import pytest_asyncio
from aio_pika import ExchangeType
from aio_pika.abc import (
    AbstractRobustChannel,
    AbstractRobustConnection,
    AbstractRobustExchange,
    AbstractRobustQueue,
)
from aio_pika.exceptions import ChannelInvalidStateError, ChannelNotFoundEntity
from pydantic import SecretStr

from cims_task_service.config import Settings
from cims_task_service.infrastructure.messaging.connection import (
    close_rabbitmq_connection,
    connect_rabbitmq,
)
from cims_task_service.infrastructure.messaging.publisher import open_publisher_channel
from cims_task_service.infrastructure.messaging.topology import TASK_ROUTING_KEY

_OPERATION_TIMEOUT_SECONDS = 5.0


@pytest.fixture
def rabbitmq_test_url() -> SecretStr:
    """Require an explicit test vhost; never fall back to the application URL."""

    raw_url = os.environ.get("CIMS_TEST_RABBITMQ_URL")
    if raw_url is None:
        pytest.skip("Set CIMS_TEST_RABBITMQ_URL to run RabbitMQ integration tests")

    try:
        parsed_url = urlsplit(raw_url)
        host = parsed_url.hostname
        username = parsed_url.username
        password = parsed_url.password
        port = parsed_url.port
        vhost = unquote_to_bytes(parsed_url.path.removeprefix("/")).decode("utf-8")
    except (UnicodeDecodeError, ValueError):
        pytest.fail("CIMS_TEST_RABBITMQ_URL must be a valid AMQP URL", pytrace=False)

    if (
        parsed_url.scheme not in {"amqp", "amqps"}
        or not host
        or not username
        or password is None
        or port == 0
        or parsed_url.query
        or parsed_url.fragment
        or not parsed_url.path.startswith("/")
        or not vhost
        or "/" in vhost
        or "%" in vhost
        or not vhost.endswith("_test")
    ):
        pytest.fail(
            "CIMS_TEST_RABBITMQ_URL must use amqp or amqps, include credentials and "
            "a host, use a non-zero port when specified, contain no query or fragment, "
            "and target one plainly encoded vhost ending in _test",
            pytrace=False,
        )

    return SecretStr(raw_url)


@pytest_asyncio.fixture
async def rabbitmq_connection(
    rabbitmq_test_url: SecretStr,
) -> AsyncIterator[AbstractRobustConnection]:
    """Open and close a production-configured connection to the guarded vhost."""

    settings = Settings(
        rabbitmq_url=rabbitmq_test_url,
        rabbitmq_connection_timeout_seconds=10.0,
        rabbitmq_reconnect_interval_seconds=1.0,
    )
    connection = await connect_rabbitmq(settings)
    try:
        yield connection
    finally:
        async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
            await close_rabbitmq_connection(connection)


@pytest_asyncio.fixture
async def rabbitmq_publisher_channel(
    rabbitmq_connection: AbstractRobustConnection,
) -> AsyncIterator[AbstractRobustChannel]:
    """Open the production publisher channel and bound its cleanup."""

    channel = await open_publisher_channel(rabbitmq_connection)
    try:
        yield channel
    finally:
        async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
            await channel.close()


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
