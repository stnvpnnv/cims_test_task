"""Composition root for the task outbox dispatcher process."""

import asyncio
from contextlib import AsyncExitStack
from datetime import timedelta

from cims_task_service.application.task_dispatcher import (
    TaskOutboxDispatcher,
    run_dispatcher_loop,
)
from cims_task_service.config import DispatcherSettings
from cims_task_service.infrastructure.database.session import (
    create_database_engine,
    create_session_factory,
    dispose_database_engine,
)
from cims_task_service.infrastructure.messaging.connection import (
    close_rabbitmq_connection,
    connect_rabbitmq,
)
from cims_task_service.infrastructure.messaging.publisher import (
    RabbitMQTaskPublisher,
    open_publisher_channel,
)
from cims_task_service.infrastructure.messaging.topology import declare_task_topology


async def run_dispatcher(
    settings: DispatcherSettings,
    *,
    stop_event: asyncio.Event,
) -> None:
    """Build owned resources and run the dispatcher until shutdown is requested."""

    async with AsyncExitStack() as resources:
        engine = create_database_engine(settings)
        resources.push_async_callback(dispose_database_engine, engine)
        session_factory = create_session_factory(engine)

        connection = await connect_rabbitmq(settings)
        resources.push_async_callback(close_rabbitmq_connection, connection)

        channel = await open_publisher_channel(connection)
        resources.push_async_callback(channel.close)
        topology = await declare_task_topology(channel)

        publisher = RabbitMQTaskPublisher(
            topology.task_exchange,
            publish_timeout_seconds=settings.rabbitmq_publish_timeout_seconds,
        )
        dispatcher = TaskOutboxDispatcher(
            session_factory,
            publisher,
            batch_size=settings.dispatcher_batch_size,
            lease_duration=timedelta(
                seconds=settings.dispatcher_lease_duration_seconds,
            ),
            retry_initial_delay=timedelta(
                seconds=settings.dispatcher_retry_initial_delay_seconds,
            ),
            retry_maximum_delay=timedelta(
                seconds=settings.dispatcher_retry_maximum_delay_seconds,
            ),
        )
        await run_dispatcher_loop(
            dispatcher.dispatch_once,
            stop_event=stop_event,
            poll_interval_seconds=settings.dispatcher_poll_interval_seconds,
        )
