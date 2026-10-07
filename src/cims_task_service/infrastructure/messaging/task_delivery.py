"""Settlement policy for one RabbitMQ task delivery."""

from typing import assert_never

from aio_pika.abc import AbstractIncomingMessage

from cims_task_service.application.task_execution import (
    TaskExecutionOutcome,
    TaskExecutor,
)
from cims_task_service.infrastructure.messaging.task_message import (
    InvalidTaskMessageError,
    decode_task_message,
)


async def handle_task_delivery(
    message: AbstractIncomingMessage,
    executor: TaskExecutor,
) -> None:
    """Execute one valid delivery and settle it according to its durable outcome."""

    try:
        payload = decode_task_message(message)
    except InvalidTaskMessageError:
        await message.reject(requeue=False)
        return
    except Exception as error:
        await _nack_after_failure(message, error)
        raise

    try:
        outcome = await executor.execute(
            payload.task_id,
            dispatch_token=payload.dispatch_token,
        )
        _require_acknowledgeable_outcome(outcome)
    except Exception as error:
        await _nack_after_failure(message, error)
        raise

    await message.ack(multiple=False)


def _require_acknowledgeable_outcome(outcome: TaskExecutionOutcome) -> None:
    match outcome:
        case TaskExecutionOutcome.NOT_CLAIMED:
            return
        case TaskExecutionOutcome.COMPLETED:
            return
        case TaskExecutionOutcome.RETRY_SCHEDULED:
            return
        case TaskExecutionOutcome.FAILED:
            return
        case TaskExecutionOutcome.LOST_OWNERSHIP:
            return

    assert_never(outcome)


async def _nack_after_failure(
    message: AbstractIncomingMessage,
    error: Exception,
) -> None:
    try:
        await message.nack(multiple=False, requeue=True)
    except Exception as nack_error:
        raise ExceptionGroup(
            "task delivery failure and RabbitMQ requeue failure",
            [error, nack_error],
        ) from None
