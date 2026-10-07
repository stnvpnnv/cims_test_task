"""Tests for reliable publication of claimed task outbox events."""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock
from uuid import UUID

import pytest
from aio_pika import DeliveryMode, Message
from aio_pika.abc import (
    AbstractRobustChannel,
    AbstractRobustConnection,
    AbstractRobustExchange,
)

from cims_task_service.infrastructure.database.outbox_repository import ClaimedOutboxEvent
from cims_task_service.infrastructure.messaging.publisher import (
    RabbitMQTaskPublisher,
    open_publisher_channel,
)
from cims_task_service.infrastructure.messaging.topology import TASK_ROUTING_KEY

_EVENT_ID = UUID("10000000-0000-4000-8000-000000000001")
_TASK_ID = UUID("20000000-0000-4000-8000-000000000002")
_DISPATCH_TOKEN = UUID("30000000-0000-4000-8000-000000000003")
_PUBLISHER_TOKEN = UUID("40000000-0000-4000-8000-000000000004")
_CREATED_AT = datetime(2026, 9, 15, 1, 30, tzinfo=UTC)

_EVENT = ClaimedOutboxEvent(
    id=_EVENT_ID,
    task_id=_TASK_ID,
    event_type=TASK_ROUTING_KEY,
    payload={
        "task_id": str(_TASK_ID),
        "dispatch_token": str(_DISPATCH_TOKEN),
    },
    message_priority=3,
    created_at=_CREATED_AT,
    available_at=_CREATED_AT,
    publish_attempts=2,
    publisher_token=_PUBLISHER_TOKEN,
    lease_expires_at=_CREATED_AT + timedelta(minutes=1),
)


def _exchange(publish: AsyncMock) -> AbstractRobustExchange:
    return cast(AbstractRobustExchange, SimpleNamespace(publish=publish))


@pytest.mark.asyncio
async def test_publisher_channel_requires_confirms_and_returned_message_errors() -> None:
    """The dedicated robust channel makes ACK and routing failures observable."""

    expected_channel = cast(AbstractRobustChannel, object())
    channel_result: asyncio.Future[AbstractRobustChannel] = (
        asyncio.get_running_loop().create_future()
    )
    channel_result.set_result(expected_channel)
    channel = Mock(return_value=channel_result)
    connection = cast(AbstractRobustConnection, SimpleNamespace(channel=channel))

    opened = await open_publisher_channel(connection)

    assert opened is expected_channel
    channel.assert_called_once_with(
        publisher_confirms=True,
        on_return_raises=True,
    )


@pytest.mark.asyncio
async def test_publish_sends_canonical_persistent_task_message() -> None:
    """Retries produce the same body and broker-visible identity metadata."""

    publish = AsyncMock(return_value=object())
    publisher = RabbitMQTaskPublisher(
        _exchange(publish),
        publish_timeout_seconds=4.5,
    )

    await publisher.publish(_EVENT)

    assert publish.await_args is not None
    assert len(publish.await_args.args) == 1
    message = publish.await_args.args[0]
    assert isinstance(message, Message)
    assert (
        message.body
        == (f'{{"dispatch_token":"{_DISPATCH_TOKEN}","task_id":"{_TASK_ID}"}}').encode()
    )
    assert message.content_type == "application/json"
    assert message.content_encoding == "utf-8"
    assert message.delivery_mode is DeliveryMode.PERSISTENT
    assert message.priority == 3
    assert message.correlation_id == str(_TASK_ID)
    assert message.message_id == str(_EVENT_ID)
    assert message.timestamp == _CREATED_AT
    assert message.type == TASK_ROUTING_KEY
    assert message.app_id == "cims-task-service"
    assert publish.await_args.kwargs == {
        "routing_key": TASK_ROUTING_KEY,
        "mandatory": True,
        "timeout": 4.5,
    }


@pytest.mark.asyncio
async def test_publish_rejects_an_unsupported_event_before_broker_access() -> None:
    """The task adapter cannot accidentally route another outbox event family."""

    publish = AsyncMock()
    publisher = RabbitMQTaskPublisher(
        _exchange(publish),
        publish_timeout_seconds=2.0,
    )

    with pytest.raises(
        ValueError,
        match=r"^unsupported outbox event type: task\.audit\.v1$",
    ):
        await publisher.publish(replace(_EVENT, event_type="task.audit.v1"))

    publish.assert_not_awaited()


@pytest.mark.parametrize(
    "timeout_seconds",
    [0.0, -1.0, float("inf"), float("-inf"), float("nan")],
)
def test_publisher_rejects_a_non_finite_or_non_positive_timeout(
    timeout_seconds: float,
) -> None:
    """Invalid operational configuration fails before any network operation."""

    publish = AsyncMock()

    with pytest.raises(
        ValueError,
        match=r"^publish_timeout_seconds must be finite and positive$",
    ):
        RabbitMQTaskPublisher(
            _exchange(publish),
            publish_timeout_seconds=timeout_seconds,
        )

    publish.assert_not_called()


@pytest.mark.asyncio
async def test_publish_rejects_a_missing_confirmation() -> None:
    """A channel without confirms can never report publication success."""

    publisher = RabbitMQTaskPublisher(
        _exchange(AsyncMock(return_value=None)),
        publish_timeout_seconds=2.0,
    )

    with pytest.raises(
        RuntimeError,
        match=r"^RabbitMQ publisher confirms are required$",
    ):
        await publisher.publish(_EVENT)


@pytest.mark.asyncio
async def test_publish_propagates_the_original_broker_failure() -> None:
    """The dispatcher receives the concrete failure needed for its retry policy."""

    expected_error = ConnectionError("broker connection lost")
    publisher = RabbitMQTaskPublisher(
        _exchange(AsyncMock(side_effect=expected_error)),
        publish_timeout_seconds=2.0,
    )

    with pytest.raises(ConnectionError) as error_info:
        await publisher.publish(_EVENT)

    assert error_info.value is expected_error


@pytest.mark.asyncio
async def test_publish_does_not_swallow_task_cancellation() -> None:
    """Cooperative process shutdown is not converted into a publication failure."""

    expected_error = asyncio.CancelledError()
    publisher = RabbitMQTaskPublisher(
        _exchange(AsyncMock(side_effect=expected_error)),
        publish_timeout_seconds=2.0,
    )

    with pytest.raises(asyncio.CancelledError) as error_info:
        await publisher.publish(_EVENT)

    assert error_info.value is expected_error


@pytest.mark.asyncio
async def test_publish_bounds_the_complete_exchange_operation() -> None:
    """The outer timeout also covers robust-channel recovery before AMQP publish."""

    async def stalled_publish(*_args: object, **_kwargs: object) -> object:
        await asyncio.Event().wait()
        return object()

    publish = AsyncMock(side_effect=stalled_publish)
    publisher = RabbitMQTaskPublisher(
        _exchange(publish),
        publish_timeout_seconds=0.01,
    )
    safety_timeout = asyncio.timeout(1)

    with pytest.raises(TimeoutError):
        async with safety_timeout:
            await publisher.publish(_EVENT)

    assert safety_timeout.expired() is False
    publish.assert_awaited_once()
