"""Reliable RabbitMQ publication for claimed task outbox events."""

import asyncio
import json
from math import isfinite
from typing import Final, cast

from aio_pika import DeliveryMode, Message
from aio_pika.abc import (
    AbstractRobustChannel,
    AbstractRobustConnection,
    AbstractRobustExchange,
)

from cims_task_service.infrastructure.database.outbox_repository import ClaimedOutboxEvent
from cims_task_service.infrastructure.messaging.topology import TASK_ROUTING_KEY

_APPLICATION_ID: Final = "cims-task-service"
_JSON_CONTENT_TYPE: Final = "application/json"
_UTF8_ENCODING: Final = "utf-8"


async def open_publisher_channel(
    connection: AbstractRobustConnection,
) -> AbstractRobustChannel:
    """Open a recoverable channel that rejects unconfirmed or unroutable messages."""

    return cast(
        AbstractRobustChannel,
        await connection.channel(
            publisher_confirms=True,
            on_return_raises=True,
        ),
    )


class RabbitMQTaskPublisher:
    """Publish one claimed execution event without mutating its database state."""

    def __init__(
        self,
        exchange: AbstractRobustExchange,
        *,
        publish_timeout_seconds: float,
    ) -> None:
        if not isfinite(publish_timeout_seconds) or publish_timeout_seconds <= 0:
            raise ValueError("publish_timeout_seconds must be finite and positive")

        self._exchange = exchange
        self._publish_timeout_seconds = publish_timeout_seconds

    async def publish(self, event: ClaimedOutboxEvent) -> None:
        """Publish a persistent, confirm-tracked message for the claimed event."""

        if event.event_type != TASK_ROUTING_KEY:
            error_message = f"unsupported outbox event type: {event.event_type}"
            raise ValueError(error_message)

        body = json.dumps(
            event.payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode(_UTF8_ENCODING)
        message = Message(
            body,
            content_type=_JSON_CONTENT_TYPE,
            content_encoding=_UTF8_ENCODING,
            delivery_mode=DeliveryMode.PERSISTENT,
            priority=event.message_priority,
            correlation_id=str(event.task_id),
            message_id=str(event.id),
            timestamp=event.created_at,
            type=event.event_type,
            app_id=_APPLICATION_ID,
        )

        async with asyncio.timeout(self._publish_timeout_seconds):
            confirmation = await self._exchange.publish(
                message,
                routing_key=TASK_ROUTING_KEY,
                mandatory=True,
                timeout=self._publish_timeout_seconds,
            )

        if confirmation is None:
            raise RuntimeError("RabbitMQ publisher confirms are required")
