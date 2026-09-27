"""Tests for the versioned RabbitMQ task message contract."""

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast
from uuid import UUID, uuid1

import pytest
from aio_pika import DeliveryMode
from aio_pika.abc import AbstractIncomingMessage
from pydantic import ValidationError

from cims_task_service.infrastructure.messaging.task_message import (
    TASK_MESSAGE_APPLICATION_ID,
    TASK_MESSAGE_CONTENT_ENCODING,
    TASK_MESSAGE_CONTENT_TYPE,
    TASK_ROUTING_KEY,
    InvalidTaskMessageError,
    decode_task_message,
)

_TASK_ID = UUID("20000000-0000-4000-8000-000000000002")
_DISPATCH_TOKEN = UUID("30000000-0000-4000-8000-000000000003")
_MESSAGE_ID = UUID("10000000-0000-4000-8000-000000000001")
_NAIVE_TIMESTAMP = datetime(2026, 9, 27, tzinfo=UTC).replace(tzinfo=None)
_BODY = (f'{{"task_id":"{_TASK_ID}","dispatch_token":"{_DISPATCH_TOKEN}"}}').encode()


def _message(**overrides: object) -> AbstractIncomingMessage:
    values: dict[str, object] = {
        "body": _BODY,
        "exchange": "cims.tasks",
        "routing_key": TASK_ROUTING_KEY,
        "type": TASK_ROUTING_KEY,
        "content_type": TASK_MESSAGE_CONTENT_TYPE,
        "content_encoding": TASK_MESSAGE_CONTENT_ENCODING,
        "delivery_mode": DeliveryMode.PERSISTENT,
        "priority": 2,
        "correlation_id": str(_TASK_ID),
        "message_id": str(_MESSAGE_ID),
        "timestamp": datetime(2026, 9, 27, tzinfo=UTC),
        "app_id": TASK_MESSAGE_APPLICATION_ID,
        "headers": {},
        "redelivered": False,
    }
    values.update(overrides)
    return cast(AbstractIncomingMessage, SimpleNamespace(**values))


def test_decode_task_message_returns_a_frozen_typed_payload() -> None:
    """A valid v1 delivery becomes an immutable UUID-based contract object."""

    payload = decode_task_message(_message())

    assert payload.task_id == _TASK_ID
    assert payload.dispatch_token == _DISPATCH_TOKEN
    with pytest.raises(ValidationError):
        setattr(payload, "task_id", _MESSAGE_ID)  # noqa: B010


@pytest.mark.parametrize("priority", [1, 2, 3])
def test_decode_task_message_accepts_every_supported_priority(priority: int) -> None:
    """All domain priorities have a valid RabbitMQ wire representation."""

    payload = decode_task_message(
        _message(
            priority=priority,
            content_encoding="UTF-8",
            headers={"x-delivery-count": 2},
            redelivered=True,
        )
    )

    assert payload.task_id == _TASK_ID


@pytest.mark.parametrize(
    "body",
    [
        b"\xff",
        b"not-json",
        b"[]",
        b'{"task_id":"20000000-0000-4000-8000-000000000002"}',
        (
            b'{"task_id":"20000000-0000-4000-8000-000000000002",'
            b'"dispatch_token":"30000000-0000-4000-8000-000000000003",'
            b'"unexpected":true}'
        ),
        b'{"task_id":42,"dispatch_token":"30000000-0000-4000-8000-000000000003"}',
        b'{"task_id":"not-a-uuid","dispatch_token":"30000000-0000-4000-8000-000000000003"}',
        (f'{{"task_id":"{uuid1()}","dispatch_token":"{_DISPATCH_TOKEN}"}}').encode(),
        (f'{{"task_id":"{_TASK_ID}","dispatch_token":"{uuid1()}"}}').encode(),
    ],
)
def test_decode_task_message_rejects_an_invalid_body(body: bytes) -> None:
    """Malformed JSON, shape changes, and non-v4 identifiers are poison messages."""

    with pytest.raises(
        InvalidTaskMessageError,
        match=r"^task message body is invalid$",
    ):
        decode_task_message(_message(body=body))


@pytest.mark.parametrize(
    ("overrides", "error_message"),
    [
        ({"routing_key": "task.execute.v2"}, "task message routing_key is invalid"),
        ({"type": None}, "task message type is invalid"),
        ({"content_type": "text/plain"}, "task message content_type is invalid"),
        ({"content_encoding": "latin-1"}, "task message content_encoding is invalid"),
        (
            {"delivery_mode": DeliveryMode.NOT_PERSISTENT},
            "task message delivery_mode is invalid",
        ),
        ({"priority": 0}, "task message priority is invalid"),
        ({"priority": 4}, "task message priority is invalid"),
        ({"priority": True}, "task message priority is invalid"),
        ({"timestamp": None}, "task message timestamp is invalid"),
        (
            {"timestamp": _NAIVE_TIMESTAMP},
            "task message timestamp is invalid",
        ),
        ({"app_id": "other-service"}, "task message app_id is invalid"),
        ({"correlation_id": None}, "task message correlation_id is invalid"),
        ({"correlation_id": "not-a-uuid"}, "task message correlation_id is invalid"),
        ({"correlation_id": str(uuid1())}, "task message correlation_id is invalid"),
        (
            {"correlation_id": str(_DISPATCH_TOKEN)},
            "task message correlation_id does not match task_id",
        ),
        ({"message_id": None}, "task message message_id is invalid"),
        ({"message_id": "not-a-uuid"}, "task message message_id is invalid"),
        ({"message_id": str(uuid1())}, "task message message_id is invalid"),
    ],
)
def test_decode_task_message_rejects_invalid_envelope_metadata(
    overrides: dict[str, object],
    error_message: str,
) -> None:
    """Producer-controlled AMQP metadata is part of the versioned contract."""

    with pytest.raises(
        InvalidTaskMessageError,
        match=rf"^{error_message}$",
    ):
        decode_task_message(_message(**overrides))
