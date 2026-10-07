"""Concurrent, explicitly supervised RabbitMQ task delivery intake."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Never, cast

from aio_pika.abc import (
    AbstractIncomingMessage,
    AbstractQueueIterator,
    AbstractRobustChannel,
    AbstractRobustConnection,
    AbstractRobustQueue,
)

type TaskDeliveryHandler = Callable[[AbstractIncomingMessage], Awaitable[None]]


async def open_consumer_channel(
    connection: AbstractRobustConnection,
    *,
    prefetch_count: int,
) -> AbstractRobustChannel:
    """Open a recoverable channel with bounded per-consumer delivery credit."""

    if prefetch_count < 1:
        raise ValueError("prefetch_count must be at least 1")

    channel = cast(
        AbstractRobustChannel,
        await connection.channel(
            publisher_confirms=False,
            on_return_raises=False,
        ),
    )
    try:
        await channel.set_qos(
            prefetch_count=prefetch_count,
            prefetch_size=0,
            global_=False,
        )
    except BaseException:
        await channel.close()
        raise

    return channel


class TaskConsumerExitedError(RuntimeError):
    """Raised when RabbitMQ delivery intake ends without a shutdown request."""


class TaskDeliveryCancelledError(RuntimeError):
    """Raised when a delivery handler cancels itself outside worker shutdown."""


class TaskConsumerCloseCancelledError(RuntimeError):
    """Raised when RabbitMQ consumer cancellation stops unexpectedly."""


class TaskDeliveryRequeueCancelledError(RuntimeError):
    """Raised when requeueing an unowned delivery stops unexpectedly."""


class _WaitOutcome(Enum):
    STOPPED = "stopped"


class _OperationCancelled(Enum):
    OUTCOME = "cancelled"


@dataclass(frozen=True, slots=True)
class _OperationCompleted[T]:
    result: T


@dataclass(frozen=True, slots=True)
class _OperationFailed:
    error: BaseException


@dataclass(frozen=True, slots=True)
class _InterruptedOperation[T]:
    result: T
    owner_error: BaseException


async def run_task_consumer(
    queue: AbstractRobustQueue,
    handler: TaskDeliveryHandler,
    *,
    concurrency: int,
    stop_event: asyncio.Event,
) -> None:
    """Consume deliveries with bounded concurrency until shutdown or failure."""

    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if stop_event.is_set():
        return

    slots = asyncio.BoundedSemaphore(concurrency)
    iterator = queue.iterator(no_ack=False)
    stop_waiter = asyncio.create_task(
        stop_event.wait(),
        name="task-consumer-stop-waiter",
    )

    try:
        try:
            entry_result = await _await_operation_or_stop(
                iterator.__aenter__(),
                stop_waiter=stop_waiter,
            )
        except BaseException as entry_error:
            await _raise_after_iterator_close(
                iterator,
                entry_error,
                group_message="task consumer entry and iterator close failed",
            )

        if isinstance(entry_result, _InterruptedOperation):
            await _raise_after_iterator_close(
                iterator,
                entry_result.owner_error,
                group_message="task consumer entry cancellation and close failed",
            )
        if entry_result is _WaitOutcome.STOPPED:
            await _close_iterator(iterator)
            return
        if stop_event.is_set():
            await _close_iterator(iterator)
            return

        singleton_failure: BaseException | None = None
        try:
            async with asyncio.TaskGroup() as handlers:
                try:
                    await _consume_deliveries(
                        iterator,
                        handler,
                        handlers=handlers,
                        slots=slots,
                        stop_waiter=stop_waiter,
                    )
                except BaseException as consume_error:
                    await _raise_after_iterator_close(
                        iterator,
                        consume_error,
                        group_message="task consumer intake and iterator close failed",
                    )
                else:
                    await _close_iterator(iterator)
        except BaseExceptionGroup as failures:
            if len(failures.exceptions) != 1:
                raise
            singleton_failure = failures.exceptions[0]

        if singleton_failure is not None:
            raise singleton_failure
    finally:
        stop_waiter.cancel()
        await asyncio.gather(stop_waiter, return_exceptions=True)


async def _consume_deliveries(
    iterator: AbstractQueueIterator,
    handler: TaskDeliveryHandler,
    *,
    handlers: asyncio.TaskGroup,
    slots: asyncio.BoundedSemaphore,
    stop_waiter: asyncio.Task[bool],
) -> None:
    delivery_number = 0
    while not stop_waiter.done():
        slot_result = await _await_operation_or_stop(
            slots.acquire(),
            stop_waiter=stop_waiter,
        )
        if isinstance(slot_result, _InterruptedOperation):
            slots.release()
            raise slot_result.owner_error
        if slot_result is _WaitOutcome.STOPPED:
            return
        if stop_waiter.done():
            slots.release()
            return

        try:
            delivery_result = await _await_operation_or_stop(
                anext(iterator),
                stop_waiter=stop_waiter,
            )
        except StopAsyncIteration as error:
            slots.release()
            if stop_waiter.done():
                return
            raise TaskConsumerExitedError("RabbitMQ task consumer stopped unexpectedly") from error
        except BaseException:
            slots.release()
            raise

        if isinstance(delivery_result, _InterruptedOperation):
            try:
                requeue_error = await _requeue_unowned_delivery(delivery_result.result)
            finally:
                slots.release()
            if requeue_error is not None:
                raise BaseExceptionGroup(
                    "task consumer cancellation and delivery requeue failed",
                    [delivery_result.owner_error, requeue_error],
                ) from None
            raise delivery_result.owner_error
        if delivery_result is _WaitOutcome.STOPPED:
            slots.release()
            return

        delivery_number += 1
        handlers.create_task(
            _handle_with_slot(delivery_result, handler=handler, slots=slots),
            name=f"task-delivery-{delivery_number}",
        )


async def _handle_with_slot(
    delivery: AbstractIncomingMessage,
    *,
    handler: TaskDeliveryHandler,
    slots: asyncio.BoundedSemaphore,
) -> None:
    handler_task = asyncio.create_task(
        _await_operation(handler(delivery)),
        name="task-delivery-handler",
    )
    try:
        await asyncio.shield(handler_task)
    except asyncio.CancelledError as error:
        current_task = asyncio.current_task()
        if current_task is not None and current_task.cancelling():
            cleanup_error = await _cancel_and_reap_handler(handler_task)
            if cleanup_error is not None:
                raise BaseExceptionGroup(
                    "task delivery cancellation and handler cleanup failed",
                    [error, cleanup_error],
                ) from None
            raise
        raise TaskDeliveryCancelledError(
            "task delivery handler cancelled outside worker shutdown"
        ) from error
    finally:
        slots.release()


async def _cancel_and_reap_handler(
    handler_task: asyncio.Task[None],
) -> BaseException | None:
    handler_task.cancel()
    while not handler_task.done():
        try:
            await asyncio.shield(handler_task)
        except asyncio.CancelledError:
            continue
        except BaseException:
            break

    if handler_task.cancelled():
        return None
    return handler_task.exception()


async def _close_iterator(iterator: AbstractQueueIterator) -> None:
    close_task = asyncio.create_task(
        _await_iterator_close(iterator),
        name="task-consumer-close",
    )
    current_task = asyncio.current_task()
    initial_cancellation_count = current_task.cancelling() if current_task else 0
    try:
        await asyncio.shield(close_task)
    except asyncio.CancelledError as error:
        cancellation_requested = (
            current_task is not None and current_task.cancelling() > initial_cancellation_count
        )
        if not cancellation_requested:
            raise TaskConsumerCloseCancelledError(
                "RabbitMQ task consumer close was cancelled unexpectedly"
            ) from error

        close_error = await _reap_iterator_close(close_task)
        if close_error is not None:
            raise BaseExceptionGroup(
                "task consumer cancellation and iterator close failed",
                [error, close_error],
            ) from None
        raise


async def _raise_after_iterator_close(
    iterator: AbstractQueueIterator,
    error: BaseException,
    *,
    group_message: str,
) -> Never:
    try:
        await _close_iterator(iterator)
    except BaseException as close_error:
        if isinstance(error, asyncio.CancelledError) and isinstance(
            close_error, asyncio.CancelledError
        ):
            raise error from None
        raise BaseExceptionGroup(
            group_message,
            [error, close_error],
        ) from None
    raise error


async def _await_iterator_close(iterator: AbstractQueueIterator) -> None:
    await iterator.close()


async def _reap_iterator_close(
    close_task: asyncio.Task[None],
) -> BaseException | None:
    while not close_task.done():
        try:
            await asyncio.shield(close_task)
        except asyncio.CancelledError:
            continue
        except BaseException:
            break

    if close_task.cancelled():
        return TaskConsumerCloseCancelledError(
            "RabbitMQ task consumer close was cancelled unexpectedly"
        )
    return close_task.exception()


async def _requeue_unowned_delivery(
    delivery: AbstractIncomingMessage,
) -> BaseException | None:
    requeue_task = asyncio.create_task(
        _nack_delivery(delivery),
        name="task-delivery-requeue",
    )
    while not requeue_task.done():
        try:
            await asyncio.shield(requeue_task)
        except asyncio.CancelledError:
            continue
        except BaseException:
            break

    if requeue_task.cancelled():
        return TaskDeliveryRequeueCancelledError(
            "RabbitMQ delivery requeue was cancelled unexpectedly"
        )
    return requeue_task.exception()


async def _nack_delivery(delivery: AbstractIncomingMessage) -> None:
    await delivery.nack(multiple=False, requeue=True)


async def _await_operation_or_stop[T](
    operation: Awaitable[T],
    *,
    stop_waiter: asyncio.Task[bool],
) -> T | _WaitOutcome | _InterruptedOperation[T]:
    operation_task = asyncio.create_task(
        _await_operation(operation),
        name="task-consumer-operation",
    )
    try:
        completed, _pending = await asyncio.wait(
            (operation_task, stop_waiter),
            return_when=asyncio.FIRST_COMPLETED,
        )
    except BaseException as error:
        operation_outcome = await _cancel_and_reap_operation(operation_task)
        if isinstance(operation_outcome, _OperationFailed):
            raise BaseExceptionGroup(
                "task consumer operation and cancellation failed",
                [error, operation_outcome.error],
            ) from None
        if isinstance(operation_outcome, _OperationCompleted):
            return _InterruptedOperation(
                result=operation_outcome.result,
                owner_error=error,
            )
        raise

    if operation_task in completed:
        return operation_task.result()

    return await _cancel_operation_for_stop(operation_task)


async def _await_operation[T](operation: Awaitable[T]) -> T:
    return await operation


async def _cancel_and_reap_operation[T](
    operation_task: asyncio.Task[T],
) -> _OperationCancelled | _OperationCompleted[T] | _OperationFailed:
    operation_task.cancel()
    while not operation_task.done():
        try:
            await asyncio.shield(operation_task)
        except asyncio.CancelledError:
            continue
        except BaseException:
            break

    if operation_task.cancelled():
        return _OperationCancelled.OUTCOME
    operation_error = operation_task.exception()
    if operation_error is not None:
        return _OperationFailed(operation_error)
    return _OperationCompleted(operation_task.result())


async def _cancel_operation_for_stop[T](
    operation_task: asyncio.Task[T],
) -> T | _WaitOutcome | _InterruptedOperation[T]:
    current_task = asyncio.current_task()
    observed_cancellation_count = current_task.cancelling() if current_task else 0
    external_cancellation: asyncio.CancelledError | None = None
    operation_task.cancel()

    while not operation_task.done():
        try:
            await asyncio.shield(operation_task)
        except asyncio.CancelledError as error:
            cancellation_count = current_task.cancelling() if current_task else 0
            if cancellation_count > observed_cancellation_count:
                external_cancellation = external_cancellation or error
                observed_cancellation_count = cancellation_count
        except BaseException:
            break

    operation_error = None if operation_task.cancelled() else operation_task.exception()
    if external_cancellation is not None:
        if operation_error is not None:
            raise BaseExceptionGroup(
                "task consumer cancellation and operation cleanup failed",
                [external_cancellation, operation_error],
            ) from None
        if not operation_task.cancelled():
            return _InterruptedOperation(
                result=operation_task.result(),
                owner_error=external_cancellation,
            )
        raise external_cancellation
    if operation_error is not None:
        raise operation_error
    if operation_task.cancelled():
        return _WaitOutcome.STOPPED
    return operation_task.result()
