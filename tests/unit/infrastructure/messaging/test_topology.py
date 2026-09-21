"""Tests for the RabbitMQ task topology."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock, call

import pytest
from aio_pika import ExchangeType
from aio_pika.abc import (
    AbstractRobustChannel,
    AbstractRobustExchange,
    AbstractRobustQueue,
)

from cims_task_service.domain.task import TaskPriority
from cims_task_service.infrastructure.messaging.topology import (
    DEAD_LETTER_EXCHANGE_NAME,
    DEAD_LETTER_QUEUE_NAME,
    DEAD_LETTER_ROUTING_KEY,
    TASK_EXCHANGE_NAME,
    TASK_QUEUE_NAME,
    TASK_ROUTING_KEY,
    declare_task_topology,
    task_message_priority,
)


@pytest.mark.parametrize(
    ("priority", "message_priority"),
    [
        (TaskPriority.LOW, 1),
        (TaskPriority.MEDIUM, 2),
        (TaskPriority.HIGH, 3),
    ],
)
def test_task_priority_maps_to_stable_rabbitmq_metadata(
    priority: TaskPriority,
    message_priority: int,
) -> None:
    """Creation and recovery share one broker-priority contract."""

    assert task_message_priority(priority) == message_priority


@pytest.mark.asyncio
async def test_task_topology_declares_durable_resources_in_dependency_order() -> None:
    """Topology is versioned, recoverable, and safe for broker restarts."""

    operations = Mock()
    declare_exchange = AsyncMock()
    declare_queue = AsyncMock()
    dead_letter_bind = AsyncMock()
    task_bind = AsyncMock()
    operations.attach_mock(declare_exchange, "declare_exchange")
    operations.attach_mock(declare_queue, "declare_queue")
    operations.attach_mock(dead_letter_bind, "dead_letter_bind")
    operations.attach_mock(task_bind, "task_bind")

    dead_letter_exchange = cast(AbstractRobustExchange, object())
    task_exchange = cast(AbstractRobustExchange, object())
    dead_letter_queue = cast(
        AbstractRobustQueue,
        SimpleNamespace(bind=dead_letter_bind),
    )
    task_queue = cast(
        AbstractRobustQueue,
        SimpleNamespace(bind=task_bind),
    )
    declare_exchange.side_effect = [dead_letter_exchange, task_exchange]
    declare_queue.side_effect = [dead_letter_queue, task_queue]
    channel = cast(
        AbstractRobustChannel,
        SimpleNamespace(
            declare_exchange=declare_exchange,
            declare_queue=declare_queue,
        ),
    )

    topology = await declare_task_topology(channel)

    assert operations.mock_calls == [
        call.declare_exchange(
            DEAD_LETTER_EXCHANGE_NAME,
            type=ExchangeType.DIRECT,
            durable=True,
            auto_delete=False,
            robust=True,
        ),
        call.declare_queue(
            DEAD_LETTER_QUEUE_NAME,
            durable=True,
            exclusive=False,
            auto_delete=False,
            arguments={"x-queue-type": "quorum"},
            robust=True,
        ),
        call.dead_letter_bind(
            dead_letter_exchange,
            routing_key=DEAD_LETTER_ROUTING_KEY,
            robust=True,
        ),
        call.declare_exchange(
            TASK_EXCHANGE_NAME,
            type=ExchangeType.DIRECT,
            durable=True,
            auto_delete=False,
            robust=True,
        ),
        call.declare_queue(
            TASK_QUEUE_NAME,
            durable=True,
            exclusive=False,
            auto_delete=False,
            arguments={
                "x-queue-type": "quorum",
                "x-dead-letter-exchange": DEAD_LETTER_EXCHANGE_NAME,
                "x-dead-letter-routing-key": DEAD_LETTER_ROUTING_KEY,
            },
            robust=True,
        ),
        call.task_bind(
            task_exchange,
            routing_key=TASK_ROUTING_KEY,
            robust=True,
        ),
    ]
    main_queue_arguments = declare_queue.await_args_list[1].kwargs["arguments"]
    assert "x-max-priority" not in main_queue_arguments
    assert topology.task_exchange is task_exchange
    assert topology.task_queue is task_queue
    assert topology.dead_letter_exchange is dead_letter_exchange
    assert topology.dead_letter_queue is dead_letter_queue


@pytest.mark.asyncio
async def test_task_topology_propagates_declaration_failure() -> None:
    """The process owner can fail startup when broker topology is incompatible."""

    expected_error = RuntimeError("incompatible broker topology")
    declare_exchange = AsyncMock(side_effect=expected_error)
    declare_queue = AsyncMock()
    channel = cast(
        AbstractRobustChannel,
        SimpleNamespace(
            declare_exchange=declare_exchange,
            declare_queue=declare_queue,
        ),
    )

    with pytest.raises(RuntimeError) as error_info:
        await declare_task_topology(channel)

    assert error_info.value is expected_error
    declare_queue.assert_not_awaited()
