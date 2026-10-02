"""Composition root for the task execution worker process."""

import asyncio
import signal
from collections.abc import Iterator
from contextlib import AsyncExitStack, ExitStack, contextmanager, suppress
from datetime import timedelta
from functools import partial
from types import FrameType
from typing import Final

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

_SHUTDOWN_SIGNALS: Final = (signal.SIGINT, signal.SIGTERM)


class WorkerShutdownTimeoutError(RuntimeError):
    """Raised after a worker exceeds its graceful shutdown deadline."""


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


async def supervise_worker(settings: WorkerSettings) -> None:
    """Run the worker until it exits or an operating-system signal stops it."""

    stop_event = asyncio.Event()
    with _install_shutdown_signal_handlers(stop_event):
        worker_task = asyncio.create_task(
            run_worker(settings, stop_event=stop_event),
            name="task-execution-worker",
        )
        shutdown_task = asyncio.create_task(
            _wait_for_shutdown(stop_event),
            name="task-execution-worker-shutdown",
        )
        try:
            completed, _pending = await asyncio.wait(
                (worker_task, shutdown_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if worker_task in completed:
                await worker_task
                return

            try:
                async with asyncio.timeout(settings.worker_shutdown_grace_seconds):
                    await asyncio.shield(worker_task)
            except TimeoutError as error:
                worker_task.cancel()
                with suppress(asyncio.CancelledError):
                    await worker_task
                message = (
                    f"worker did not stop within {settings.worker_shutdown_grace_seconds:g} seconds"
                )
                raise WorkerShutdownTimeoutError(message) from error
        finally:
            await _cancel_and_wait(shutdown_task)
            await _cancel_and_wait(worker_task)


def main() -> None:
    """Load environment settings and run the worker process."""

    asyncio.run(supervise_worker(WorkerSettings()))


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
