"""RabbitMQ contract tests for task event publication."""

import json
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from aio_pika import DeliveryMode
from aio_pika.abc import (
    AbstractRobustChannel,
    AbstractRobustExchange,
    AbstractRobustQueue,
)
from aio_pika.exceptions import PublishError

from cims_task_service.infrastructure.database.outbox_repository import ClaimedOutboxEvent
from cims_task_service.infrastructure.messaging.publisher import RabbitMQTaskPublisher
from cims_task_service.infrastructure.messaging.topology import TASK_ROUTING_KEY

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_EVENT_ID = UUID("10000000-0000-4000-8000-000000000001")
_TASK_ID = UUID("20000000-0000-4000-8000-000000000002")
_DISPATCH_TOKEN = UUID("30000000-0000-4000-8000-000000000003")
_PUBLISHER_TOKEN = UUID("40000000-0000-4000-8000-000000000004")
_CREATED_AT = datetime(2026, 9, 15, 1, 30, tzinfo=UTC)
_PUBLISH_TIMEOUT_SECONDS = 5.0

_EVENT = ClaimedOutboxEvent(
    id=_EVENT_ID,
    task_id=_TASK_ID,
    event_type=TASK_ROUTING_KEY,
    payload={
        "task_id": str(_TASK_ID),
        "dispatch_token": str(_DISPATCH_TOKEN),
        "details": {"label": "проверка", "active": True},
    },
    message_priority=3,
    created_at=_CREATED_AT,
    available_at=_CREATED_AT,
    publish_attempts=2,
    publisher_token=_PUBLISHER_TOKEN,
    lease_expires_at=_CREATED_AT + timedelta(minutes=1),
)

_EXPECTED_BODY = (
    '{"details":{"active":true,"label":"проверка"},'
    f'"dispatch_token":"{_DISPATCH_TOKEN}","task_id":"{_TASK_ID}"}}'
).encode()


async def test_publish_routes_a_confirmed_message_with_exact_metadata(
    rabbitmq_task_route: tuple[AbstractRobustExchange, AbstractRobustQueue],
) -> None:
    """A successful return means RabbitMQ accepted and routed the durable message."""

    exchange, queue = rabbitmq_task_route
    publisher = RabbitMQTaskPublisher(
        exchange,
        publish_timeout_seconds=_PUBLISH_TIMEOUT_SECONDS,
    )

    await publisher.publish(_EVENT)
    incoming = await queue.get(
        no_ack=False,
        fail=True,
        timeout=_PUBLISH_TIMEOUT_SECONDS,
    )

    async with incoming.process(requeue=False):
        assert incoming.body == _EXPECTED_BODY
        assert json.loads(incoming.body) == _EVENT.payload
        assert incoming.content_type == "application/json"
        assert incoming.content_encoding == "utf-8"
        assert incoming.delivery_mode is DeliveryMode.PERSISTENT
        assert incoming.priority == _EVENT.message_priority
        assert incoming.correlation_id == str(_TASK_ID)
        assert incoming.message_id == str(_EVENT_ID)
        assert incoming.timestamp == _CREATED_AT
        assert incoming.type == TASK_ROUTING_KEY
        assert incoming.app_id == "cims-task-service"
        assert incoming.exchange == exchange.name
        assert incoming.routing_key == TASK_ROUTING_KEY
        assert incoming.redelivered is False

    assert incoming.processed is True


async def test_publish_raises_for_an_unroutable_mandatory_message(
    rabbitmq_exchange: AbstractRobustExchange,
    rabbitmq_publisher_channel: AbstractRobustChannel,
) -> None:
    """The production channel exposes Basic.Return instead of reporting success."""

    publisher = RabbitMQTaskPublisher(
        rabbitmq_exchange,
        publish_timeout_seconds=_PUBLISH_TIMEOUT_SECONDS,
    )

    with pytest.raises(PublishError) as error_info:
        await publisher.publish(_EVENT)

    error = error_info.value
    returned_message = error.message
    assert error.args == ("NO_ROUTE", TASK_ROUTING_KEY)
    assert returned_message is not None
    assert returned_message.body == _EXPECTED_BODY
    assert returned_message.exchange == rabbitmq_exchange.name
    assert returned_message.routing_key == TASK_ROUTING_KEY
    assert rabbitmq_publisher_channel.is_closed is False
