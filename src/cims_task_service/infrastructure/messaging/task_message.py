"""Versioned RabbitMQ contract for task execution messages."""

from datetime import datetime
from typing import Final
from uuid import UUID

from aio_pika import DeliveryMode
from aio_pika.abc import AbstractIncomingMessage
from pydantic import UUID4, BaseModel, ConfigDict, ValidationError

TASK_ROUTING_KEY: Final = "task.execute.v1"
TASK_MESSAGE_CONTENT_TYPE: Final = "application/json"
TASK_MESSAGE_CONTENT_ENCODING: Final = "utf-8"
TASK_MESSAGE_APPLICATION_ID: Final = "cims-task-service"
TASK_MESSAGE_LOW_PRIORITY: Final = 1
TASK_MESSAGE_MEDIUM_PRIORITY: Final = 2
TASK_MESSAGE_HIGH_PRIORITY: Final = 3
TASK_MESSAGE_PRIORITIES: Final = frozenset(
    {
        TASK_MESSAGE_LOW_PRIORITY,
        TASK_MESSAGE_MEDIUM_PRIORITY,
        TASK_MESSAGE_HIGH_PRIORITY,
    }
)


class TaskExecutionMessage(BaseModel):
    """Validated payload carried by the version-one execution route."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    task_id: UUID4
    dispatch_token: UUID4


class InvalidTaskMessageError(ValueError):
    """Raised when a delivery does not satisfy the task message contract."""


def decode_task_message(message: AbstractIncomingMessage) -> TaskExecutionMessage:
    """Validate an incoming AMQP envelope and decode its execution payload."""

    _validate_envelope(message)

    try:
        payload = TaskExecutionMessage.model_validate_json(message.body)
    except ValidationError:
        raise InvalidTaskMessageError("task message body is invalid") from None

    correlation_id = _parse_uuid4(
        message.correlation_id,
        error_message="task message correlation_id is invalid",
    )
    if correlation_id != payload.task_id:
        raise InvalidTaskMessageError("task message correlation_id does not match task_id")

    _parse_uuid4(
        message.message_id,
        error_message="task message message_id is invalid",
    )
    return payload


def _validate_envelope(message: AbstractIncomingMessage) -> None:
    if message.routing_key != TASK_ROUTING_KEY:
        raise InvalidTaskMessageError("task message routing_key is invalid")
    if message.type != TASK_ROUTING_KEY:
        raise InvalidTaskMessageError("task message type is invalid")
    if message.content_type != TASK_MESSAGE_CONTENT_TYPE:
        raise InvalidTaskMessageError("task message content_type is invalid")
    if (
        not isinstance(message.content_encoding, str)
        or message.content_encoding.casefold() != TASK_MESSAGE_CONTENT_ENCODING
    ):
        raise InvalidTaskMessageError("task message content_encoding is invalid")
    if message.delivery_mode is not DeliveryMode.PERSISTENT:
        raise InvalidTaskMessageError("task message delivery_mode is invalid")
    if (
        not isinstance(message.priority, int)
        or isinstance(message.priority, bool)
        or message.priority not in TASK_MESSAGE_PRIORITIES
    ):
        raise InvalidTaskMessageError("task message priority is invalid")
    if not _is_aware_datetime(message.timestamp):
        raise InvalidTaskMessageError("task message timestamp is invalid")
    if message.app_id != TASK_MESSAGE_APPLICATION_ID:
        raise InvalidTaskMessageError("task message app_id is invalid")


def _parse_uuid4(value: str | None, *, error_message: str) -> UUID:
    if not isinstance(value, str):
        raise InvalidTaskMessageError(error_message)

    try:
        parsed = UUID(value)
    except ValueError:
        raise InvalidTaskMessageError(error_message) from None

    if parsed.version != 4:
        raise InvalidTaskMessageError(error_message)
    return parsed


def _is_aware_datetime(value: datetime | None) -> bool:
    return isinstance(value, datetime) and value.utcoffset() is not None
