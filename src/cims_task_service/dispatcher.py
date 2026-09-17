"""Composition root for the task outbox dispatcher process."""

import asyncio
import signal
from collections.abc import Iterator
from contextlib import AsyncExitStack, ExitStack, contextmanager, suppress
from datetime import timedelta
from types import FrameType
from typing import Final

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

_SHUTDOWN_SIGNALS: Final = (signal.SIGINT, signal.SIGTERM)


class DispatcherShutdownTimeoutError(RuntimeError):
    """Raised after a dispatcher exceeds its graceful shutdown deadline."""


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


async def supervise_dispatcher(settings: DispatcherSettings) -> None:
    """Run the dispatcher until it exits or an operating-system signal stops it."""

    stop_event = asyncio.Event()
    with _install_shutdown_signal_handlers(stop_event):
        dispatcher_task = asyncio.create_task(
            run_dispatcher(settings, stop_event=stop_event),
            name="task-outbox-dispatcher",
        )
        shutdown_task = asyncio.create_task(
            _wait_for_shutdown(stop_event),
            name="task-outbox-dispatcher-shutdown",
        )
        try:
            completed, _pending = await asyncio.wait(
                (dispatcher_task, shutdown_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if dispatcher_task in completed:
                await dispatcher_task
                return

            try:
                async with asyncio.timeout(settings.dispatcher_shutdown_grace_seconds):
                    await asyncio.shield(dispatcher_task)
            except TimeoutError as error:
                dispatcher_task.cancel()
                with suppress(asyncio.CancelledError):
                    await dispatcher_task
                message = (
                    "dispatcher did not stop within "
                    f"{settings.dispatcher_shutdown_grace_seconds:g} seconds"
                )
                raise DispatcherShutdownTimeoutError(message) from error
        finally:
            await _cancel_and_wait(shutdown_task)
            await _cancel_and_wait(dispatcher_task)


def main() -> None:
    """Load environment settings and run the dispatcher process."""

    asyncio.run(supervise_dispatcher(DispatcherSettings()))


@contextmanager
def _install_shutdown_signal_handlers(stop_event: asyncio.Event) -> Iterator[None]:
    """Install portable process handlers and restore their predecessors."""

    loop = asyncio.get_running_loop()

    def request_shutdown(_signal_number: int, _frame: FrameType | None) -> None:
        loop.call_soon_threadsafe(stop_event.set)

    with ExitStack() as handlers:
        for shutdown_signal in _SHUTDOWN_SIGNALS:
            previous_handler = signal.signal(shutdown_signal, request_shutdown)
            handlers.callback(signal.signal, shutdown_signal, previous_handler)
        yield


async def _wait_for_shutdown(stop_event: asyncio.Event) -> None:
    await stop_event.wait()


async def _cancel_and_wait[T](task: asyncio.Task[T]) -> None:
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
