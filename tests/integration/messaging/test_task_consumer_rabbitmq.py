"""RabbitMQ contracts for bounded delivery intake and message ownership."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from uuid import uuid4

import pytest
from aio_pika import Message
from aio_pika.abc import (
    AbstractIncomingMessage,
    AbstractRobustChannel,
    AbstractRobustConnection,
    AbstractRobustQueue,
)

from cims_task_service.infrastructure.messaging.task_consumer import (
    TaskDeliveryHandler,
    open_consumer_channel,
    run_task_consumer,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_OPERATION_TIMEOUT_SECONDS = 5.0
_POLL_INTERVAL_SECONDS = 0.01


@dataclass(frozen=True, slots=True)
class _ConsumerRoute:
    channel: AbstractRobustChannel
    queue: AbstractRobustQueue
    observer_channel: AbstractRobustChannel


@asynccontextmanager
async def _consumer_route(
    connection: AbstractRobustConnection,
    publisher_channel: AbstractRobustChannel,
    *,
    prefetch_count: int,
) -> AsyncIterator[_ConsumerRoute]:
    """Keep an owned queue alive across consumer cancellation and channel close."""

    queue = await publisher_channel.declare_queue(
        f"cims.tests.consumer.{uuid4().hex}",
        durable=True,
        exclusive=False,
        auto_delete=False,
        arguments={"x-queue-type": "classic"},
        timeout=_OPERATION_TIMEOUT_SECONDS,
        robust=False,
    )
    try:
        async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
            channel = await open_consumer_channel(connection, prefetch_count=prefetch_count)
        try:
            consumer_queue = await channel.declare_queue(
                queue.name,
                passive=True,
                timeout=_OPERATION_TIMEOUT_SECONDS,
                robust=False,
            )
            yield _ConsumerRoute(channel, consumer_queue, publisher_channel)
        finally:
            async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
                await channel.close()
    finally:
        await queue.delete(
            if_unused=False,
            if_empty=False,
            timeout=_OPERATION_TIMEOUT_SECONDS,
        )


@asynccontextmanager
async def _running_consumer(
    route: _ConsumerRoute,
    handler: TaskDeliveryHandler,
    *,
    concurrency: int,
    stop_event: asyncio.Event,
) -> AsyncIterator[asyncio.Task[None]]:
    """Always cancel and reap a consumer if a broker assertion fails."""

    consumer = asyncio.create_task(
        run_task_consumer(
            route.queue,
            handler,
            concurrency=concurrency,
            stop_event=stop_event,
        ),
        name="integration-task-consumer",
    )
    try:
        yield consumer
    finally:
        stop_event.set()
        consumer.cancel()
        _completed, pending = await asyncio.wait((consumer,), timeout=_OPERATION_TIMEOUT_SECONDS)
        if pending:
            # Closing the owned channel releases consumer cleanup waiting on the broker.
            try:
                async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
                    await route.channel.close()
            finally:
                consumer.cancel()
                _completed, pending = await asyncio.wait(
                    (consumer,), timeout=_OPERATION_TIMEOUT_SECONDS
                )
        if pending:
            raise TimeoutError("integration consumer did not cancel after channel close")
        await asyncio.gather(consumer, return_exceptions=True)


async def _publish(route: _ConsumerRoute, body: bytes) -> None:
    await route.observer_channel.default_exchange.publish(
        Message(body),
        routing_key=route.queue.name,
        mandatory=True,
        timeout=_OPERATION_TIMEOUT_SECONDS,
    )


async def _counts(route: _ConsumerRoute) -> tuple[int, int]:
    """Read broker state using a fresh passive declaration on another channel."""

    observed_queue = await route.observer_channel.declare_queue(
        route.queue.name,
        passive=True,
        timeout=_OPERATION_TIMEOUT_SECONDS,
        robust=False,
    )
    declaration = observed_queue.declaration_result
    assert declaration.message_count is not None
    assert declaration.consumer_count is not None
    return declaration.message_count, declaration.consumer_count


async def _wait_for_counts(
    route: _ConsumerRoute,
    *,
    ready: int,
    consumers: int,
) -> None:
    # AMQP exposes these counters through passive declarations, not event notifications.
    async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
        while await _counts(route) != (ready, consumers):  # noqa: ASYNC110
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)


async def _get_message(route: _ConsumerRoute) -> AbstractIncomingMessage:
    observer_queue = await route.observer_channel.declare_queue(
        route.queue.name,
        passive=True,
        timeout=_OPERATION_TIMEOUT_SECONDS,
        robust=False,
    )
    return await observer_queue.get(
        no_ack=False,
        fail=True,
        timeout=_OPERATION_TIMEOUT_SECONDS,
    )


async def test_acknowledged_deliveries_do_not_return_after_consumer_channel_close(
    rabbitmq_connection: AbstractRobustConnection,
    rabbitmq_publisher_channel: AbstractRobustChannel,
) -> None:
    """ACKs survive channel close; no completed message remains ready or unacked."""

    stop_event = asyncio.Event()
    received: list[AbstractIncomingMessage] = []
    bodies = (b"first", b"second")

    async def acknowledge(message: AbstractIncomingMessage) -> None:
        await message.ack(multiple=False)
        received.append(message)
        if len(received) == len(bodies):
            stop_event.set()

    async with _consumer_route(
        rabbitmq_connection,
        rabbitmq_publisher_channel,
        prefetch_count=1,
    ) as route:
        for body in bodies:
            await _publish(route, body)
        async with _running_consumer(
            route,
            acknowledge,
            concurrency=1,
            stop_event=stop_event,
        ) as consumer:
            async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
                await asyncio.shield(consumer)

        async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
            await route.channel.close()
        assert tuple(message.body for message in received) == bodies
        assert all(message.processed and not message.redelivered for message in received)
        assert await _counts(route) == (0, 0)


async def test_broker_prefetch_and_graceful_stop_preserve_pending_deliveries(
    rabbitmq_connection: AbstractRobustConnection,
    rabbitmq_publisher_channel: AbstractRobustChannel,
) -> None:
    """Broker credit bounds intake, while shutdown drains only started handlers."""

    stop_event = asyncio.Event()
    two_started = asyncio.Event()
    release_handlers = asyncio.Event()
    started: list[bytes] = []
    completed: list[bytes] = []

    async def acknowledge_after_release(message: AbstractIncomingMessage) -> None:
        started.append(message.body)
        if len(started) == 2:
            two_started.set()
        await release_handlers.wait()
        await message.ack(multiple=False)
        completed.append(message.body)

    async with _consumer_route(
        rabbitmq_connection,
        rabbitmq_publisher_channel,
        prefetch_count=2,
    ) as route:
        for body in (b"first", b"second", b"queued-before-stop"):
            await _publish(route, body)
        async with _running_consumer(
            route,
            acknowledge_after_release,
            concurrency=4,
            stop_event=stop_event,
        ) as consumer:
            async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
                await two_started.wait()
            await _wait_for_counts(route, ready=1, consumers=1)

            stop_event.set()
            await _wait_for_counts(route, ready=1, consumers=0)
            assert consumer.done() is False
            assert started == [b"first", b"second"]
            assert completed == []

            await _publish(route, b"queued-after-stop")
            assert await _counts(route) == (2, 0)
            release_handlers.set()
            async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
                await asyncio.shield(consumer)

        assert sorted(completed) == [b"first", b"second"]
        assert started == [b"first", b"second"]
        assert await _counts(route) == (2, 0)
        for expected_body in (b"queued-before-stop", b"queued-after-stop"):
            message = await _get_message(route)
            assert message.body == expected_body
            assert message.redelivered is False
            await message.ack(multiple=False)


async def test_handler_failure_propagates_and_nacked_delivery_is_redelivered(
    rabbitmq_connection: AbstractRobustConnection,
    rabbitmq_publisher_channel: AbstractRobustChannel,
) -> None:
    """A failed handler terminates intake without losing its explicitly requeued message."""

    stop_event = asyncio.Event()
    expected_error = RuntimeError("integration handler failed")

    async def fail_after_requeue(message: AbstractIncomingMessage) -> None:
        await message.nack(multiple=False, requeue=True)
        raise expected_error

    async with _consumer_route(
        rabbitmq_connection,
        rabbitmq_publisher_channel,
        prefetch_count=1,
    ) as route:
        await _publish(route, b"retry-after-failure")
        async with _running_consumer(
            route,
            fail_after_requeue,
            concurrency=1,
            stop_event=stop_event,
        ) as consumer:
            with pytest.raises(RuntimeError) as error_info:
                async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
                    await asyncio.shield(consumer)
            assert error_info.value is expected_error

        async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
            await route.channel.close()
        await _wait_for_counts(route, ready=1, consumers=0)
        redelivery = await _get_message(route)
        assert redelivery.body == b"retry-after-failure"
        assert redelivery.redelivered is True
        await redelivery.ack(multiple=False)


async def test_forced_cancellation_requeues_unacked_delivery_when_channel_closes(
    rabbitmq_connection: AbstractRobustConnection,
    rabbitmq_publisher_channel: AbstractRobustChannel,
) -> None:
    """Cancelling active work leaves settlement to channel close and broker redelivery."""

    stop_event = asyncio.Event()
    started = asyncio.Event()
    handler_cancelled = asyncio.Event()
    never_release = asyncio.Event()

    async def wait_without_settlement(_message: AbstractIncomingMessage) -> None:
        started.set()
        try:
            await never_release.wait()
        finally:
            handler_cancelled.set()

    async with _consumer_route(
        rabbitmq_connection,
        rabbitmq_publisher_channel,
        prefetch_count=1,
    ) as route:
        await _publish(route, b"interrupted-work")
        async with _running_consumer(
            route,
            wait_without_settlement,
            concurrency=1,
            stop_event=stop_event,
        ) as consumer:
            async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
                await started.wait()
            consumer.cancel()
            with pytest.raises(asyncio.CancelledError):
                async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
                    await asyncio.shield(consumer)
            assert handler_cancelled.is_set()
            assert await _counts(route) == (0, 0)

        async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
            await route.channel.close()
        await _wait_for_counts(route, ready=1, consumers=0)
        redelivery = await _get_message(route)
        assert redelivery.body == b"interrupted-work"
        assert redelivery.redelivered is True
        await redelivery.ack(multiple=False)
