"""Tests for RabbitMQ task delivery settlement policy."""

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Never, cast
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from aio_pika import DeliveryMode
from aio_pika.abc import AbstractIncomingMessage

from cims_task_service.application.task_execution import (
    TaskExecutionOutcome,
    TaskExecutor,
)
from cims_task_service.infrastructure.messaging import task_delivery
from cims_task_service.infrastructure.messaging.task_delivery import handle_task_delivery
from cims_task_service.infrastructure.messaging.task_message import (
    TASK_MESSAGE_APPLICATION_ID,
    TASK_MESSAGE_CONTENT_ENCODING,
    TASK_MESSAGE_CONTENT_TYPE,
    TASK_ROUTING_KEY,
    InvalidTaskMessageError,
)

_TASK_ID = UUID("20000000-0000-4000-8000-000000000002")
_DISPATCH_TOKEN = UUID("30000000-0000-4000-8000-000000000003")
_MESSAGE_ID = UUID("10000000-0000-4000-8000-000000000001")
_BODY = (f'{{"task_id":"{_TASK_ID}","dispatch_token":"{_DISPATCH_TOKEN}"}}').encode()


@dataclass(frozen=True, slots=True)
class _DeliveryHarness:
    message: AbstractIncomingMessage
    ack: AsyncMock
    reject: AsyncMock
    nack: AsyncMock


def _delivery(**overrides: object) -> _DeliveryHarness:
    ack = AsyncMock()
    reject = AsyncMock()
    nack = AsyncMock()
    values: dict[str, object] = {
        "body": _BODY,
        "routing_key": TASK_ROUTING_KEY,
        "type": TASK_ROUTING_KEY,
        "content_type": TASK_MESSAGE_CONTENT_TYPE,
        "content_encoding": TASK_MESSAGE_CONTENT_ENCODING,
        "delivery_mode": DeliveryMode.PERSISTENT,
        "priority": 2,
        "correlation_id": str(_TASK_ID),
        "message_id": str(_MESSAGE_ID),
        "timestamp": datetime(2026, 9, 28, tzinfo=UTC),
        "app_id": TASK_MESSAGE_APPLICATION_ID,
        "ack": ack,
        "reject": reject,
        "nack": nack,
    }
    values.update(overrides)
    return _DeliveryHarness(
        message=cast(AbstractIncomingMessage, SimpleNamespace(**values)),
        ack=ack,
        reject=reject,
        nack=nack,
    )


def _executor(execute: AsyncMock) -> TaskExecutor:
    return cast(TaskExecutor, SimpleNamespace(execute=execute))


def _assert_not_settled(delivery: _DeliveryHarness) -> None:
    delivery.ack.assert_not_awaited()
    delivery.reject.assert_not_awaited()
    delivery.nack.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", tuple(TaskExecutionOutcome))
async def test_every_durable_execution_outcome_acknowledges_the_delivery(
    outcome: TaskExecutionOutcome,
) -> None:
    """Every current durable or stale outcome makes the delivery disposable."""

    delivery = _delivery()
    execute = AsyncMock(return_value=outcome)

    await handle_task_delivery(delivery.message, _executor(execute))

    execute.assert_awaited_once_with(_TASK_ID, dispatch_token=_DISPATCH_TOKEN)
    delivery.ack.assert_awaited_once_with(multiple=False)
    delivery.reject.assert_not_awaited()
    delivery.nack.assert_not_awaited()


@pytest.mark.asyncio
async def test_delivery_is_acknowledged_only_after_executor_returns() -> None:
    """The ACK cannot precede the executor's durable transaction boundary."""

    delivery = _delivery()

    async def execute(
        task_id: UUID,
        *,
        dispatch_token: UUID,
    ) -> TaskExecutionOutcome:
        assert task_id == _TASK_ID
        assert dispatch_token == _DISPATCH_TOKEN
        delivery.ack.assert_not_awaited()
        return TaskExecutionOutcome.COMPLETED

    await handle_task_delivery(delivery.message, _executor(AsyncMock(side_effect=execute)))

    delivery.ack.assert_awaited_once_with(multiple=False)


@pytest.mark.asyncio
async def test_invalid_delivery_is_rejected_without_executing_a_task() -> None:
    """A known contract violation is dead-lettered instead of touching the database."""

    delivery = _delivery(body=b"not-json")
    execute = AsyncMock()

    await handle_task_delivery(delivery.message, _executor(execute))

    delivery.reject.assert_awaited_once_with(requeue=False)
    delivery.ack.assert_not_awaited()
    delivery.nack.assert_not_awaited()
    execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_unexpected_decoder_failure_is_requeued_and_propagated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A decoder defect remains retryable and terminates the supervised worker."""

    expected_error = RuntimeError("decoder failed")

    def fail_decode(_message: AbstractIncomingMessage) -> Never:
        raise expected_error

    monkeypatch.setattr(task_delivery, "decode_task_message", fail_decode)
    delivery = _delivery()
    execute = AsyncMock()

    with pytest.raises(RuntimeError) as error_info:
        await handle_task_delivery(delivery.message, _executor(execute))

    assert error_info.value is expected_error
    delivery.nack.assert_awaited_once_with(multiple=False, requeue=True)
    delivery.ack.assert_not_awaited()
    delivery.reject.assert_not_awaited()
    execute.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "expected_error",
    [
        ConnectionError("database unavailable"),
        InvalidTaskMessageError("executor boundary failure"),
    ],
)
async def test_execution_failure_is_requeued_and_propagated(
    expected_error: Exception,
) -> None:
    """Only decoder contract errors are poison messages; executor errors remain retryable."""

    delivery = _delivery()
    execute = AsyncMock(side_effect=expected_error)

    with pytest.raises(type(expected_error)) as error_info:
        await handle_task_delivery(delivery.message, _executor(execute))

    assert error_info.value is expected_error
    delivery.nack.assert_awaited_once_with(multiple=False, requeue=True)
    delivery.ack.assert_not_awaited()
    delivery.reject.assert_not_awaited()


@pytest.mark.asyncio
async def test_external_cancellation_leaves_the_delivery_unsettled() -> None:
    """Forced shutdown reaps execution and lets channel close requeue the message."""

    delivery = _delivery()
    started = asyncio.Event()
    reaped = asyncio.Event()

    async def execute(
        _task_id: UUID,
        *,
        dispatch_token: UUID,
    ) -> TaskExecutionOutcome:
        assert dispatch_token == _DISPATCH_TOKEN
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            reaped.set()
        return TaskExecutionOutcome.COMPLETED

    handler_task = asyncio.create_task(
        handle_task_delivery(
            delivery.message,
            _executor(AsyncMock(side_effect=execute)),
        )
    )
    async with asyncio.timeout(1):
        await started.wait()
    handler_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await handler_task

    assert reaped.is_set()
    _assert_not_settled(delivery)


@pytest.mark.asyncio
async def test_unknown_execution_outcome_is_requeued_and_fails_fast() -> None:
    """A future outcome cannot be acknowledged without an explicit policy decision."""

    delivery = _delivery()
    execute = AsyncMock(return_value=cast(TaskExecutionOutcome, "future_outcome"))

    with pytest.raises(AssertionError, match="Expected code to be unreachable"):
        await handle_task_delivery(delivery.message, _executor(execute))

    delivery.nack.assert_awaited_once_with(multiple=False, requeue=True)
    delivery.ack.assert_not_awaited()
    delivery.reject.assert_not_awaited()


@pytest.mark.asyncio
async def test_ack_failure_propagates_without_another_settlement() -> None:
    """An uncertain ACK is resolved by process failure and idempotent redelivery."""

    expected_error = ConnectionError("ack failed")
    delivery = _delivery()
    delivery.ack.side_effect = expected_error

    with pytest.raises(ConnectionError) as error_info:
        await handle_task_delivery(
            delivery.message,
            _executor(AsyncMock(return_value=TaskExecutionOutcome.COMPLETED)),
        )

    assert error_info.value is expected_error
    delivery.reject.assert_not_awaited()
    delivery.nack.assert_not_awaited()


@pytest.mark.asyncio
async def test_reject_failure_propagates_without_another_settlement() -> None:
    """A failed poison-message rejection leaves broker state deliberately unresolved."""

    expected_error = ConnectionError("reject failed")
    delivery = _delivery(body=b"not-json")
    delivery.reject.side_effect = expected_error
    execute = AsyncMock()

    with pytest.raises(ConnectionError) as error_info:
        await handle_task_delivery(delivery.message, _executor(execute))

    assert error_info.value is expected_error
    delivery.ack.assert_not_awaited()
    delivery.nack.assert_not_awaited()
    execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_nack_failure_preserves_processing_and_settlement_errors() -> None:
    """Both causes remain observable when processing and requeue fail together."""

    execution_error = ConnectionError("database unavailable")
    nack_error = RuntimeError("nack failed")
    delivery = _delivery()
    delivery.nack.side_effect = nack_error

    with pytest.raises(ExceptionGroup) as error_info:
        await handle_task_delivery(
            delivery.message,
            _executor(AsyncMock(side_effect=execution_error)),
        )

    assert error_info.value.exceptions == (execution_error, nack_error)
    delivery.ack.assert_not_awaited()
    delivery.reject.assert_not_awaited()
