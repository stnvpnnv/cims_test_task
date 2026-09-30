"""Composition root for the task execution worker process."""

import asyncio
from contextlib import AsyncExitStack
from datetime import timedelta
from functools import partial

from cims_task_service.application.execution_retry import ExecutionRetryDelayPolicy
from cims_task_service.application.task_execution import TaskExecutor
from cims_task_service.application.task_processor import TextStatisticsProcessor
from cims_task_service.config import WorkerSettings
from cims_task_service.infrastructure.database.session import (
    create_database_engine,
    create_session_factory,
    dispose_database_engine,
)
from cims_task_service.infrastructure.messaging.connection import (
    close_rabbitmq_connection,
    connect_rabbitmq,
)
from cims_task_service.infrastructure.messaging.task_consumer import (
    open_consumer_channel,
    run_task_consumer,
)
from cims_task_service.infrastructure.messaging.task_delivery import handle_task_delivery
from cims_task_service.infrastructure.messaging.topology import declare_task_topology


async def run_worker(
    settings: WorkerSettings,
    *,
    stop_event: asyncio.Event,
) -> None:
    """Build owned resources and consume task deliveries until shutdown."""

    async with AsyncExitStack() as resources:
        engine = create_database_engine(settings)
        resources.push_async_callback(dispose_database_engine, engine)
        session_factory = create_session_factory(engine)

        connection = await connect_rabbitmq(settings)
        resources.push_async_callback(close_rabbitmq_connection, connection)

        channel = await open_consumer_channel(
            connection,
            prefetch_count=settings.worker_concurrency,
        )
        resources.push_async_callback(channel.close)
        topology = await declare_task_topology(channel)

        retry_delay = ExecutionRetryDelayPolicy(
            initial_delay=timedelta(
                seconds=settings.execution_retry_initial_delay_seconds,
            ),
            maximum_delay=timedelta(
                seconds=settings.execution_retry_maximum_delay_seconds,
            ),
        )
        processor = TextStatisticsProcessor()
        executor = TaskExecutor(
            session_factory,
            processor,
            lease_duration=timedelta(
                seconds=settings.worker_lease_duration_seconds,
            ),
            heartbeat_interval=timedelta(
                seconds=settings.worker_heartbeat_interval_seconds,
            ),
            processing_timeout=timedelta(
                seconds=settings.worker_processing_timeout_seconds,
            ),
            retry_delay_for_attempt=retry_delay,
        )

        await run_task_consumer(
            topology.task_queue,
            partial(handle_task_delivery, executor=executor),
            concurrency=settings.worker_concurrency,
            stop_event=stop_event,
        )
