"""Broker delivery-limit recovery before a worker acquires execution ownership."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from uuid import uuid4

import pytest
import pytest_asyncio
from aio_pika.abc import AbstractIncomingMessage, AbstractRobustConnection
from aio_pika.exceptions import ChannelNotFoundEntity
from sqlalchemy import select

from cims_task_service.application.pending_delivery_recovery import PendingTaskDeliveryRecovery
from cims_task_service.application.task_creation import CreateTaskCommand, create_task
from cims_task_service.application.task_dispatcher import DispatchBatchResult, TaskOutboxDispatcher
from cims_task_service.application.task_execution import TaskExecutionOutcome, TaskExecutor
from cims_task_service.application.task_execution_recovery import (
    RecoveryBatchResult,
    TaskExecutionRecovery,
)
from cims_task_service.application.task_processor import TextStatisticsProcessor
from cims_task_service.domain.task import TaskPriority, TaskStatus
from cims_task_service.infrastructure.database.models import OutboxEventModel, TaskModel
from cims_task_service.infrastructure.database.session import AsyncSessionFactory
from cims_task_service.infrastructure.messaging import topology as topology_module
from cims_task_service.infrastructure.messaging.publisher import (
    RabbitMQTaskPublisher,
    open_publisher_channel,
)
from cims_task_service.infrastructure.messaging.task_consumer import open_consumer_channel
from cims_task_service.infrastructure.messaging.task_delivery import handle_task_delivery
from cims_task_service.infrastructure.messaging.task_message import decode_task_message
from cims_task_service.infrastructure.messaging.topology import TaskTopology

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_OPERATION_TIMEOUT_SECONDS = 5.0
_DELIVERY_LIMIT = 20


@pytest_asyncio.fixture
async def pending_recovery_topology(
    monkeypatch: pytest.MonkeyPatch,
    rabbitmq_connection: AbstractRobustConnection,
) -> AsyncIterator[TaskTopology]:
    """Confine real production declarations to four uniquely owned broker names."""

    prefix = f"cims.tests.pending-recovery.{uuid4().hex}"
    names = {
        "TASK_EXCHANGE_NAME": prefix,
        "TASK_QUEUE_NAME": f"{prefix}.execute.v1",
        "DEAD_LETTER_EXCHANGE_NAME": f"{prefix}.dead-letter",
        "DEAD_LETTER_QUEUE_NAME": f"{prefix}.dead-letter.v1",
    }
    for constant, name in names.items():
        monkeypatch.setattr(topology_module, constant, name)

    channel = await open_publisher_channel(rabbitmq_connection)
    try:
        async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
            topology = await topology_module.declare_task_topology(channel)
        yield topology
    finally:
        try:
            async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
                await channel.close()
        finally:
            failures: list[Exception] = []
            for is_queue, name in (
                (True, names["TASK_QUEUE_NAME"]),
                (True, names["DEAD_LETTER_QUEUE_NAME"]),
                (False, names["TASK_EXCHANGE_NAME"]),
                (False, names["DEAD_LETTER_EXCHANGE_NAME"]),
            ):
                try:
                    async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
                        cleanup_channel = await open_publisher_channel(rabbitmq_connection)
                        try:
                            if is_queue:
                                await cleanup_channel.queue_delete(
                                    name,
                                    if_unused=False,
                                    if_empty=False,
                                    timeout=_OPERATION_TIMEOUT_SECONDS,
                                )
                            else:
                                await cleanup_channel.exchange_delete(
                                    name,
                                    if_unused=False,
                                    timeout=_OPERATION_TIMEOUT_SECONDS,
                                )
                        finally:
                            await cleanup_channel.close()
                except ChannelNotFoundEntity:
                    continue
                except Exception as error:
                    failures.append(error)
            if failures:
                raise ExceptionGroup("pending recovery topology cleanup failed", failures)


@asynccontextmanager
async def _consume_one(
    connection: AbstractRobustConnection,
    queue_name: str,
) -> AsyncIterator[asyncio.Future[AbstractIncomingMessage]]:
    """Capture one real delivery, leaving settlement to the caller or channel close."""

    channel = await open_consumer_channel(connection, prefetch_count=1)
    future: asyncio.Future[AbstractIncomingMessage] = asyncio.get_running_loop().create_future()

    async def receive(delivery: AbstractIncomingMessage) -> None:
        if not future.done():
            future.set_result(delivery)

    try:
        async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
            queue = await channel.declare_queue(
                queue_name,
                passive=True,
                timeout=_OPERATION_TIMEOUT_SECONDS,
                robust=False,
            )
            await queue.consume(
                receive,
                no_ack=False,
                timeout=_OPERATION_TIMEOUT_SECONDS,
                robust=False,
            )
        yield future
    finally:
        future.cancel()
        # No basic.cancel or nack: close only this test's channel with unacked intake.
        async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
            await channel.close()


async def _wait_for_broker_result(
    futures: tuple[asyncio.Future[AbstractIncomingMessage], ...],
    observed_failure_counts: list[int],
) -> set[asyncio.Future[AbstractIncomingMessage]]:
    completed, _pending = await asyncio.wait(
        futures,
        timeout=_OPERATION_TIMEOUT_SECONDS,
        return_when=asyncio.FIRST_COMPLETED,
    )
    assert completed, (
        "Broker did not deliver to source or DLQ after "
        f"{len(observed_failure_counts)} channel closures; "
        f"observed failure counts: {observed_failure_counts}"
    )
    return completed


async def test_pending_recovery_republishes_delivery_limited_task_without_an_extra_execution(
    postgres_session_factory: AsyncSessionFactory,
    rabbitmq_connection: AbstractRobustConnection,
    pending_recovery_topology: TaskTopology,
) -> None:
    """A pre-claim broker failure is recovered through the same outbox event and token."""

    topology = pending_recovery_topology
    created = await create_task(
        CreateTaskCommand(
            name="Delivery-limit candidate",
            description="Worker crashes before database claim",
            priority=TaskPriority.HIGH,
        ),
        session_factory=postgres_session_factory,
        max_attempts=3,
    )
    assert created.created
    dispatch_token = created.task.dispatch_token
    assert dispatch_token is not None
    publisher = RabbitMQTaskPublisher(
        topology.task_exchange,
        publish_timeout_seconds=_OPERATION_TIMEOUT_SECONDS,
    )
    dispatcher = TaskOutboxDispatcher(
        postgres_session_factory,
        publisher,
        batch_size=1,
        lease_duration=timedelta(seconds=10),
        retry_initial_delay=timedelta(seconds=1),
        retry_maximum_delay=timedelta(seconds=2),
    )
    assert await dispatcher.dispatch_once() == DispatchBatchResult(
        claimed=1, published=1, rescheduled=0, lost_ownership=0
    )
    async with postgres_session_factory() as session:
        initial_event = (await session.scalars(select(OutboxEventModel))).one()
    assert initial_event.published_at is not None
    assert initial_event.publish_attempts == 1

    # RabbitMQ 4.3 counts unacknowledged channel closures as failed deliveries;
    # basic.nack(requeue=True) is an explicit return and would not reproduce this.
    observed_failure_counts: list[int] = []
    async with _consume_one(rabbitmq_connection, topology.dead_letter_queue.name) as dlq_future:
        for _attempt in range(_DELIVERY_LIMIT + 1):
            async with _consume_one(rabbitmq_connection, topology.task_queue.name) as source_future:
                completed = await _wait_for_broker_result(
                    (source_future, dlq_future), observed_failure_counts
                )
                if dlq_future in completed:
                    break
                delivery = source_future.result()
                decoded = decode_task_message(delivery)
                assert decoded.task_id == created.task.id
                assert decoded.dispatch_token == dispatch_token
                failure_count = delivery.headers.get("x-delivery-count", 0)
                assert isinstance(failure_count, int)
                observed_failure_counts.append(failure_count)
                assert not delivery.processed
        await _wait_for_broker_result((dlq_future,), observed_failure_counts)
        assert _DELIVERY_LIMIT <= len(observed_failure_counts) <= _DELIVERY_LIMIT + 1
        assert observed_failure_counts == list(range(len(observed_failure_counts)))
        dead_letter = dlq_future.result()
        assert json.loads(dead_letter.body) == {
            "task_id": str(created.task.id),
            "dispatch_token": str(dispatch_token),
        }
        deaths = dead_letter.headers["x-death"]
        assert isinstance(deaths, list)
        first_death = deaths[0]
        assert isinstance(first_death, dict)
        assert first_death["reason"] == "delivery_limit"
        assert first_death["queue"] == topology.task_queue.name
        async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
            await dead_letter.ack()

    # Observer channels are closed; unacknowledged leftovers would now be deliverable.
    async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
        assert await topology.task_queue.get(fail=False) is None
        assert await topology.dead_letter_queue.get(fail=False) is None

    recovery = TaskExecutionRecovery(
        postgres_session_factory,
        batch_size=1,
        retry_delay_for_attempt=lambda _attempt_count: timedelta(seconds=1),
    )
    assert await recovery.recover_once() == RecoveryBatchResult(locked=0, retried=0, failed=0)
    async with postgres_session_factory() as session:
        task = (await session.scalars(select(TaskModel))).one()
        outbox = (await session.scalars(select(OutboxEventModel))).one()
    assert task.status is TaskStatus.PENDING
    assert task.attempt_count == 0
    assert task.started_at is task.finished_at is None
    assert task.dispatch_token == dispatch_token
    assert task.execution_token is None
    assert task.lease_expires_at is None
    assert outbox.published_at is not None
    assert outbox.discarded_at is None
    assert await dispatcher.dispatch_once() == DispatchBatchResult(
        claimed=0, published=0, rescheduled=0, lost_ownership=0
    )

    pending_recovery = PendingTaskDeliveryRecovery(
        postgres_session_factory,
        batch_size=1,
        delivery_timeout=timedelta(microseconds=1),
    )
    assert await pending_recovery.recover_once() == 1
    assert await pending_recovery.recover_once() == 0
    async with postgres_session_factory() as session:
        recovered_task = (await session.scalars(select(TaskModel))).one()
        recovered_event = (await session.scalars(select(OutboxEventModel))).one()
    assert recovered_task.status is TaskStatus.PENDING
    assert recovered_task.attempt_count == 0
    assert recovered_task.dispatch_token == dispatch_token
    assert recovered_task.execution_token is None
    assert recovered_task.lease_expires_at is None
    assert recovered_task.started_at is recovered_task.finished_at is None
    assert recovered_task.result is recovered_task.error is None
    assert recovered_event.id == initial_event.id
    assert recovered_event.payload == initial_event.payload
    assert recovered_event.publish_attempts == 1
    assert recovered_event.published_at is None
    assert recovered_event.discarded_at is None
    assert recovered_event.publisher_token is None
    assert recovered_event.lease_expires_at is None

    assert await dispatcher.dispatch_once() == DispatchBatchResult(
        claimed=1, published=1, rescheduled=0, lost_ownership=0
    )
    executor = TaskExecutor(
        postgres_session_factory,
        TextStatisticsProcessor(),
        lease_duration=timedelta(seconds=10),
        heartbeat_interval=timedelta(seconds=1),
        processing_timeout=timedelta(seconds=5),
        retry_delay_for_attempt=lambda _attempt_count: timedelta(seconds=1),
    )
    async with _consume_one(rabbitmq_connection, topology.task_queue.name) as republished_future:
        await _wait_for_broker_result((republished_future,), observed_failure_counts)
        republished = republished_future.result()
        payload = decode_task_message(republished)
        assert payload.task_id == created.task.id
        assert payload.dispatch_token == dispatch_token
        assert republished.message_id == str(initial_event.id)
        assert republished.headers.get("x-delivery-count", 0) == 0
        async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
            await handle_task_delivery(republished, executor)
        assert republished.processed

    assert (
        await executor.execute(created.task.id, dispatch_token=dispatch_token)
        is TaskExecutionOutcome.NOT_CLAIMED
    )
    async with postgres_session_factory() as session:
        completed_task = (await session.scalars(select(TaskModel))).one()
        completed_event = (await session.scalars(select(OutboxEventModel))).one()
    assert completed_task.status is TaskStatus.COMPLETED
    assert completed_task.attempt_count == 1
    assert completed_task.result == {
        "name_length": len(created.task.name),
        "description_length": len(created.task.description),
    }
    assert completed_task.error is None
    assert completed_task.started_at is not None
    assert completed_task.finished_at is not None
    assert completed_task.created_at <= completed_task.started_at <= completed_task.finished_at
    assert completed_task.dispatch_token is completed_task.execution_token is None
    assert completed_task.lease_expires_at is None
    assert completed_event.id == initial_event.id
    assert completed_event.payload == initial_event.payload
    assert completed_event.publish_attempts == 2
    assert completed_event.published_at is not None
    assert completed_event.published_at >= initial_event.published_at
    assert completed_event.discarded_at is None
    assert completed_event.publisher_token is None
    assert completed_event.lease_expires_at is None
    assert completed_event.last_error is None
    async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
        assert await topology.task_queue.get(fail=False) is None
        assert await topology.dead_letter_queue.get(fail=False) is None
