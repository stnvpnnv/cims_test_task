"""RabbitMQ topology for task execution."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from aio_pika import ExchangeType
from aio_pika.abc import (
    AbstractRobustChannel,
    AbstractRobustExchange,
    AbstractRobustQueue,
)

from cims_task_service.domain.task import TaskPriority
from cims_task_service.infrastructure.messaging.task_message import (
    TASK_MESSAGE_HIGH_PRIORITY,
    TASK_MESSAGE_LOW_PRIORITY,
    TASK_MESSAGE_MEDIUM_PRIORITY,
)
from cims_task_service.infrastructure.messaging.task_message import (
    TASK_ROUTING_KEY as _TASK_ROUTING_KEY,
)

TASK_EXCHANGE_NAME = "cims.tasks"
TASK_QUEUE_NAME = "cims.tasks.execute.v1"
TASK_ROUTING_KEY: Final = _TASK_ROUTING_KEY

DEAD_LETTER_EXCHANGE_NAME = "cims.tasks.dead-letter"
DEAD_LETTER_QUEUE_NAME = "cims.tasks.dead-letter.v1"
DEAD_LETTER_ROUTING_KEY = "task.dead-letter.v1"

_TASK_MESSAGE_PRIORITIES: Final[Mapping[TaskPriority, int]] = MappingProxyType(
    {
        TaskPriority.LOW: TASK_MESSAGE_LOW_PRIORITY,
        TaskPriority.MEDIUM: TASK_MESSAGE_MEDIUM_PRIORITY,
        TaskPriority.HIGH: TASK_MESSAGE_HIGH_PRIORITY,
    }
)


def task_message_priority(priority: TaskPriority) -> int:
    """Translate the domain priority into RabbitMQ message metadata."""

    return _TASK_MESSAGE_PRIORITIES[priority]


@dataclass(frozen=True, slots=True)
class TaskTopology:
    """Declared broker resources used by dispatchers and workers."""

    task_exchange: AbstractRobustExchange
    task_queue: AbstractRobustQueue
    dead_letter_exchange: AbstractRobustExchange
    dead_letter_queue: AbstractRobustQueue


async def declare_task_topology(
    channel: AbstractRobustChannel,
) -> TaskTopology:
    """Declare recoverable durable resources in dependency order."""

    dead_letter_exchange = await channel.declare_exchange(
        DEAD_LETTER_EXCHANGE_NAME,
        type=ExchangeType.DIRECT,
        durable=True,
        auto_delete=False,
        robust=True,
    )
    dead_letter_queue = await channel.declare_queue(
        DEAD_LETTER_QUEUE_NAME,
        durable=True,
        exclusive=False,
        auto_delete=False,
        arguments={"x-queue-type": "quorum"},
        robust=True,
    )
    await dead_letter_queue.bind(
        dead_letter_exchange,
        routing_key=DEAD_LETTER_ROUTING_KEY,
        robust=True,
    )

    task_exchange = await channel.declare_exchange(
        TASK_EXCHANGE_NAME,
        type=ExchangeType.DIRECT,
        durable=True,
        auto_delete=False,
        robust=True,
    )
    task_queue = await channel.declare_queue(
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
    )
    await task_queue.bind(
        task_exchange,
        routing_key=TASK_ROUTING_KEY,
        robust=True,
    )

    return TaskTopology(
        task_exchange=task_exchange,
        task_queue=task_queue,
        dead_letter_exchange=dead_letter_exchange,
        dead_letter_queue=dead_letter_queue,
    )
