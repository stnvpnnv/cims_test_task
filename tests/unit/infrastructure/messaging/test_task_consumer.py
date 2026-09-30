"""Tests for concurrent RabbitMQ task delivery intake."""

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from aio_pika.abc import (
    AbstractIncomingMessage,
    AbstractQueueIterator,
    AbstractRobustChannel,
    AbstractRobustConnection,
    AbstractRobustQueue,
)

from cims_task_service.infrastructure.messaging.task_consumer import (
    TaskConsumerCloseCancelledError,
    TaskConsumerExitedError,
    TaskDeliveryCancelledError,
    TaskDeliveryHandler,
    TaskDeliveryRequeueCancelledError,
    open_consumer_channel,
    run_task_consumer,
)

_END_OF_STREAM = object()


class _FakeQueueIterator:
    def __init__(self) -> None:
        self._items: asyncio.Queue[object] = asyncio.Queue()
        self._is_closed = False
        self.entered = asyncio.Event()
        self.closed = asyncio.Event()
        self.next_started = asyncio.Event()
        self.next_cancelled = asyncio.Event()
        self.close_was_concurrent = False
        self.requeued: list[AbstractIncomingMessage] = []
        self.close_error: BaseException | None = None
        self.close_calls = 0
        self._next_active = False

    def put(self, delivery: AbstractIncomingMessage) -> None:
        self._items.put_nowait(delivery)

    def finish(self) -> None:
        self._items.put_nowait(_END_OF_STREAM)

    def fail(self, error: BaseException) -> None:
        self._items.put_nowait(error)

    async def __aenter__(self) -> AbstractQueueIterator:
        self.entered.set()
        return cast(AbstractQueueIterator, self)

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: object,
    ) -> None:
        await self.close()

    async def __anext__(self) -> AbstractIncomingMessage:
        self._next_active = True
        self.next_started.set()
        try:
            item = await self._items.get()
        except asyncio.CancelledError:
            self.next_cancelled.set()
            raise
        finally:
            self._next_active = False

        if item is _END_OF_STREAM:
            raise StopAsyncIteration
        if isinstance(item, BaseException):
            raise item
        return cast(AbstractIncomingMessage, item)

    async def close(self) -> None:
        self.close_calls += 1
        if self._next_active:
            self.close_was_concurrent = True
        if self._is_closed:
            return

        self._is_closed = True
        while not self._items.empty():
            item = self._items.get_nowait()
            if item is not _END_OF_STREAM and not isinstance(item, BaseException):
                self.requeued.append(cast(AbstractIncomingMessage, item))
        self._items.put_nowait(_END_OF_STREAM)
        self.closed.set()

        if self.close_error is not None:
            raise self.close_error


class _GatedQueueIterator(_FakeQueueIterator):
    def __init__(self) -> None:
        super().__init__()
        self.release_delivery = asyncio.Event()
        self.delivery_returned = asyncio.Event()

    async def __anext__(self) -> AbstractIncomingMessage:
        self._next_active = True
        self.next_started.set()
        try:
            item = await self._items.get()
            await self.release_delivery.wait()
            self.delivery_returned.set()
        except asyncio.CancelledError:
            self.next_cancelled.set()
            raise
        finally:
            self._next_active = False

        if item is _END_OF_STREAM:
            raise StopAsyncIteration
        return cast(AbstractIncomingMessage, item)


class _GatedEntryQueueIterator(_FakeQueueIterator):
    def __init__(self) -> None:
        super().__init__()
        self.entry_started = asyncio.Event()
        self.release_entry = asyncio.Event()
        self.entry_cancelled = asyncio.Event()

    async def __aenter__(self) -> AbstractQueueIterator:
        self.entry_started.set()
        try:
            await self.release_entry.wait()
        except asyncio.CancelledError:
            self.entry_cancelled.set()
            raise
        self.entered.set()
        return cast(AbstractQueueIterator, self)


class _ReturningEntryAfterCancellationQueueIterator(_FakeQueueIterator):
    def __init__(self) -> None:
        super().__init__()
        self.entry_started = asyncio.Event()
        self.entry_cancelled = asyncio.Event()
        self.release_entry = asyncio.Event()

    async def __aenter__(self) -> AbstractQueueIterator:
        self.entry_started.set()
        try:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")
        except asyncio.CancelledError:
            self.entry_cancelled.set()
            await self.release_entry.wait()
            self.entered.set()
            return cast(AbstractQueueIterator, self)


class _FailingEntryQueueIterator(_FakeQueueIterator):
    def __init__(self, entry_error: BaseException) -> None:
        super().__init__()
        self._entry_error = entry_error

    async def __aenter__(self) -> AbstractQueueIterator:
        raise self._entry_error


class _StopOnEntryQueueIterator(_FakeQueueIterator):
    def __init__(self, stop_event: asyncio.Event) -> None:
        super().__init__()
        self._stop_event = stop_event

    async def __aenter__(self) -> AbstractQueueIterator:
        self.entered.set()
        self._stop_event.set()
        return cast(AbstractQueueIterator, self)


class _GatedCloseQueueIterator(_FakeQueueIterator):
    def __init__(self) -> None:
        super().__init__()
        self.close_started = asyncio.Event()
        self.release_close = asyncio.Event()
        self.close_cancelled = asyncio.Event()
        self._close_claimed = False

    async def close(self) -> None:
        self.close_calls += 1
        if self._next_active:
            self.close_was_concurrent = True
        if self._close_claimed:
            return

        self._close_claimed = True
        self.close_started.set()
        try:
            await self.release_close.wait()
        except asyncio.CancelledError:
            self.close_cancelled.set()
            raise

        self._is_closed = True
        while not self._items.empty():
            item = self._items.get_nowait()
            if item is not _END_OF_STREAM and not isinstance(item, BaseException):
                self.requeued.append(cast(AbstractIncomingMessage, item))
        self._items.put_nowait(_END_OF_STREAM)
        self.closed.set()

        if self.close_error is not None:
            raise self.close_error


class _ClosingOnCancellationQueueIterator(_GatedCloseQueueIterator):
    def __init__(self) -> None:
        super().__init__()
        self.idle_next_started = asyncio.Event()

    async def __anext__(self) -> AbstractIncomingMessage:
        self._next_active = True
        self.next_started.set()
        if self._items.empty():
            self.idle_next_started.set()
        try:
            item = await self._items.get()
        except asyncio.CancelledError:
            self.next_cancelled.set()
            close_task = asyncio.create_task(self.close())
            await close_task
            raise
        finally:
            self._next_active = False

        if item is _END_OF_STREAM:
            raise StopAsyncIteration
        if isinstance(item, BaseException):
            raise item
        return cast(AbstractIncomingMessage, item)


class _OwnerCancellingQueueIterator(_FakeQueueIterator):
    def __init__(self) -> None:
        super().__init__()
        self.owner_task: asyncio.Task[None] | None = None

    async def __anext__(self) -> AbstractIncomingMessage:
        delivery = await super().__anext__()
        assert self.owner_task is not None
        self.owner_task.cancel()
        return delivery


class _ReturningAfterCancellationQueueIterator(_FakeQueueIterator):
    def __init__(self, delivery: AbstractIncomingMessage) -> None:
        super().__init__()
        self._delivery = delivery
        self.release_delivery = asyncio.Event()

    async def __anext__(self) -> AbstractIncomingMessage:
        self._next_active = True
        self.next_started.set()
        try:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")
        except asyncio.CancelledError:
            self.next_cancelled.set()
            await self.release_delivery.wait()
            return self._delivery
        finally:
            self._next_active = False


@dataclass(frozen=True, slots=True)
class _QueueHarness:
    queue: AbstractRobustQueue
    iterator: _FakeQueueIterator
    iterator_factory: Mock


def _queue(iterator: _FakeQueueIterator | None = None) -> _QueueHarness:
    fake_iterator = _FakeQueueIterator() if iterator is None else iterator
    iterator_factory = Mock(return_value=cast(AbstractQueueIterator, fake_iterator))
    queue = cast(AbstractRobustQueue, SimpleNamespace(iterator=iterator_factory))
    return _QueueHarness(
        queue=queue,
        iterator=fake_iterator,
        iterator_factory=iterator_factory,
    )


def _delivery(identifier: str) -> AbstractIncomingMessage:
    return cast(AbstractIncomingMessage, SimpleNamespace(message_id=identifier))


def _handler(handler: AsyncMock) -> TaskDeliveryHandler:
    return cast(TaskDeliveryHandler, handler)


def _leaf_exceptions(error: BaseException) -> tuple[BaseException, ...]:
    if isinstance(error, BaseExceptionGroup):
        return tuple(
            leaf for nested_error in error.exceptions for leaf in _leaf_exceptions(nested_error)
        )
    return (error,)


async def _wait(event: asyncio.Event) -> None:
    async with asyncio.timeout(1):
        await event.wait()


async def _finish(task: asyncio.Task[None]) -> None:
    async with asyncio.timeout(1):
        await task


@pytest.mark.asyncio
async def test_consumer_channel_disables_confirms_and_limits_delivery_credit() -> None:
    """The dedicated robust channel exposes only the configured worker capacity."""

    set_qos = AsyncMock()
    close = AsyncMock()
    expected_channel = cast(
        AbstractRobustChannel,
        SimpleNamespace(set_qos=set_qos, close=close),
    )
    channel_result: asyncio.Future[AbstractRobustChannel] = (
        asyncio.get_running_loop().create_future()
    )
    channel_result.set_result(expected_channel)
    channel = Mock(return_value=channel_result)
    connection = cast(AbstractRobustConnection, SimpleNamespace(channel=channel))

    opened = await open_consumer_channel(connection, prefetch_count=7)

    assert opened is expected_channel
    channel.assert_called_once_with(
        publisher_confirms=False,
        on_return_raises=False,
    )
    set_qos.assert_awaited_once_with(
        prefetch_count=7,
        prefetch_size=0,
        global_=False,
    )
    close.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("prefetch_count", [0, -1])
async def test_consumer_channel_rejects_non_positive_prefetch_before_broker_access(
    prefetch_count: int,
) -> None:
    """Invalid delivery credit cannot allocate a broker channel."""

    channel = Mock()
    connection = cast(AbstractRobustConnection, SimpleNamespace(channel=channel))

    with pytest.raises(ValueError, match=r"^prefetch_count must be at least 1$"):
        await open_consumer_channel(connection, prefetch_count=prefetch_count)

    channel.assert_not_called()


@pytest.mark.asyncio
async def test_consumer_channel_closes_after_qos_base_exception() -> None:
    """Failed QoS setup cannot leak a channel or replace the original failure."""

    class FatalQosError(BaseException):
        pass

    expected_error = FatalQosError()
    set_qos = AsyncMock(side_effect=expected_error)
    close = AsyncMock()
    expected_channel = cast(
        AbstractRobustChannel,
        SimpleNamespace(set_qos=set_qos, close=close),
    )
    channel_result: asyncio.Future[AbstractRobustChannel] = (
        asyncio.get_running_loop().create_future()
    )
    channel_result.set_result(expected_channel)
    channel = Mock(return_value=channel_result)
    connection = cast(AbstractRobustConnection, SimpleNamespace(channel=channel))

    with pytest.raises(FatalQosError) as error_info:
        await open_consumer_channel(connection, prefetch_count=3)

    assert error_info.value is expected_error
    channel.assert_called_once_with(
        publisher_confirms=False,
        on_return_raises=False,
    )
    set_qos.assert_awaited_once_with(
        prefetch_count=3,
        prefetch_size=0,
        global_=False,
    )
    close.assert_awaited_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrency", [0, -1])
async def test_consumer_rejects_non_positive_concurrency(concurrency: int) -> None:
    """Invalid local capacity fails before the broker consumer is created."""

    queue = _queue()

    with pytest.raises(ValueError, match=r"^concurrency must be at least 1$"):
        await run_task_consumer(
            queue.queue,
            _handler(AsyncMock()),
            concurrency=concurrency,
            stop_event=asyncio.Event(),
        )

    queue.iterator_factory.assert_not_called()


@pytest.mark.asyncio
async def test_preexisting_stop_does_not_create_a_broker_consumer() -> None:
    """A worker stopped during startup never begins RabbitMQ intake."""

    queue = _queue()
    stop_event = asyncio.Event()
    stop_event.set()

    await run_task_consumer(
        queue.queue,
        _handler(AsyncMock()),
        concurrency=1,
        stop_event=stop_event,
    )

    queue.iterator_factory.assert_not_called()


@pytest.mark.asyncio
async def test_stop_during_blocked_entry_cancels_and_closes_iterator() -> None:
    """Shutdown interrupts RabbitMQ registration without leaking its task."""

    iterator = _GatedEntryQueueIterator()
    queue = _queue(iterator)
    stop_event = asyncio.Event()
    handle = AsyncMock()
    consumer = asyncio.create_task(
        run_task_consumer(
            queue.queue,
            _handler(handle),
            concurrency=1,
            stop_event=stop_event,
        )
    )
    await _wait(iterator.entry_started)

    stop_event.set()
    await _finish(consumer)

    assert iterator.entry_cancelled.is_set()
    assert iterator.closed.is_set()
    assert iterator.next_started.is_set() is False
    handle.assert_not_awaited()


@pytest.mark.asyncio
async def test_entry_completing_with_stop_never_starts_delivery_intake() -> None:
    """A registration and shutdown tie closes before requesting a delivery."""

    stop_event = asyncio.Event()
    iterator = _StopOnEntryQueueIterator(stop_event)
    queue = _queue(iterator)
    handle = AsyncMock()

    await run_task_consumer(
        queue.queue,
        _handler(handle),
        concurrency=1,
        stop_event=stop_event,
    )

    assert iterator.entered.is_set()
    assert iterator.closed.is_set()
    assert iterator.next_started.is_set() is False
    handle.assert_not_awaited()


@pytest.mark.asyncio
async def test_external_cancellation_during_entry_reaps_and_closes_iterator() -> None:
    """Forced shutdown cannot leave a RabbitMQ registration task running."""

    iterator = _GatedEntryQueueIterator()
    queue = _queue(iterator)
    consumer = asyncio.create_task(
        run_task_consumer(
            queue.queue,
            _handler(AsyncMock()),
            concurrency=1,
            stop_event=asyncio.Event(),
        )
    )
    await _wait(iterator.entry_started)

    consumer.cancel()

    with pytest.raises(asyncio.CancelledError):
        await consumer

    assert iterator.entry_cancelled.is_set()
    assert iterator.closed.is_set()
    assert iterator.next_started.is_set() is False


@pytest.mark.asyncio
async def test_entry_returned_during_cancellation_is_closed() -> None:
    """A late successful registration is closed before cancellation escapes."""

    iterator = _ReturningEntryAfterCancellationQueueIterator()
    queue = _queue(iterator)
    consumer = asyncio.create_task(
        run_task_consumer(
            queue.queue,
            _handler(AsyncMock()),
            concurrency=1,
            stop_event=asyncio.Event(),
        )
    )
    await _wait(iterator.entry_started)

    consumer.cancel()
    await _wait(iterator.entry_cancelled)
    iterator.release_entry.set()

    with pytest.raises(asyncio.CancelledError):
        await consumer

    assert iterator.entered.is_set()
    assert iterator.closed.is_set()
    assert iterator.next_started.is_set() is False


@pytest.mark.asyncio
async def test_entry_and_iterator_close_failures_are_preserved() -> None:
    """Registration and cleanup errors remain independently observable."""

    entry_error = ConnectionError("consumer registration failed")
    close_error = RuntimeError("iterator close failed")
    iterator = _FailingEntryQueueIterator(entry_error)
    iterator.close_error = close_error
    queue = _queue(iterator)

    with pytest.raises(BaseExceptionGroup) as error_info:
        await run_task_consumer(
            queue.queue,
            _handler(AsyncMock()),
            concurrency=1,
            stop_event=asyncio.Event(),
        )

    failures = _leaf_exceptions(error_info.value)
    assert len(failures) == 2
    assert any(error is entry_error for error in failures)
    assert any(error is close_error for error in failures)
    assert iterator.closed.is_set()


@pytest.mark.asyncio
async def test_entry_cancellation_and_iterator_close_failure_are_preserved() -> None:
    """A close failure cannot replace cancellation during registration."""

    iterator = _GatedEntryQueueIterator()
    close_error = ConnectionError("iterator close failed")
    iterator.close_error = close_error
    queue = _queue(iterator)
    consumer = asyncio.create_task(
        run_task_consumer(
            queue.queue,
            _handler(AsyncMock()),
            concurrency=1,
            stop_event=asyncio.Event(),
        )
    )
    await _wait(iterator.entry_started)

    consumer.cancel()

    with pytest.raises(BaseExceptionGroup) as error_info:
        await consumer

    failures = _leaf_exceptions(error_info.value)
    assert len(failures) == 2
    assert sum(isinstance(error, asyncio.CancelledError) for error in failures) == 1
    assert any(error is close_error for error in failures)
    assert iterator.entry_cancelled.is_set()
    assert iterator.closed.is_set()


@pytest.mark.asyncio
async def test_idle_stop_cancels_next_before_closing_the_iterator() -> None:
    """Shutdown wakes an idle consumer without racing iterator close against anext."""

    queue = _queue()
    stop_event = asyncio.Event()
    consumer = asyncio.create_task(
        run_task_consumer(
            queue.queue,
            _handler(AsyncMock()),
            concurrency=1,
            stop_event=stop_event,
        )
    )
    await _wait(queue.iterator.next_started)

    stop_event.set()
    await _finish(consumer)

    queue.iterator_factory.assert_called_once_with(no_ack=False)
    assert queue.iterator.next_cancelled.is_set()
    assert queue.iterator.closed.is_set()
    assert queue.iterator.close_was_concurrent is False


@pytest.mark.asyncio
async def test_concurrency_limit_reuses_released_slots() -> None:
    """At most the configured handlers run while completed slots remain reusable."""

    queue = _queue()
    deliveries = tuple(_delivery(str(index)) for index in range(3))
    for delivery in deliveries:
        queue.iterator.put(delivery)

    stop_event = asyncio.Event()
    release_first_batch = asyncio.Event()
    first_batch_started = asyncio.Event()
    third_started = asyncio.Event()
    started: list[AbstractIncomingMessage] = []
    active = 0
    maximum_active = 0

    async def handle(delivery: AbstractIncomingMessage) -> None:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        started.append(delivery)
        if len(started) == 2:
            first_batch_started.set()
        try:
            if delivery in deliveries[:2]:
                await release_first_batch.wait()
            else:
                third_started.set()
        finally:
            active -= 1

    consumer = asyncio.create_task(
        run_task_consumer(
            queue.queue,
            handle,
            concurrency=2,
            stop_event=stop_event,
        )
    )
    await _wait(first_batch_started)
    await asyncio.sleep(0)
    assert len(started) == 2

    release_first_batch.set()
    await _wait(third_started)
    stop_event.set()
    await _finish(consumer)

    assert started == list(deliveries)
    assert maximum_active == 2
    assert queue.iterator.close_was_concurrent is False


@pytest.mark.asyncio
async def test_stop_at_capacity_closes_intake_before_draining_handlers() -> None:
    """Shutdown requeues buffered work before waiting for active handlers."""

    queue = _queue()
    deliveries = tuple(_delivery(str(index)) for index in range(3))
    for delivery in deliveries:
        queue.iterator.put(delivery)

    stop_event = asyncio.Event()
    handlers_started = asyncio.Event()
    release_handlers = asyncio.Event()
    handled: list[AbstractIncomingMessage] = []

    async def handle(delivery: AbstractIncomingMessage) -> None:
        handled.append(delivery)
        if len(handled) == 2:
            handlers_started.set()
        await release_handlers.wait()

    consumer = asyncio.create_task(
        run_task_consumer(
            queue.queue,
            handle,
            concurrency=2,
            stop_event=stop_event,
        )
    )
    await _wait(handlers_started)

    stop_event.set()
    await _wait(queue.iterator.closed)

    assert consumer.done() is False
    assert handled == list(deliveries[:2])
    assert queue.iterator.requeued == [deliveries[2]]

    release_handlers.set()
    await _finish(consumer)
    assert queue.iterator.close_was_concurrent is False


@pytest.mark.asyncio
async def test_handler_failure_cannot_interrupt_iterator_close() -> None:
    """TaskGroup cancellation cannot interrupt an in-flight Basic.Cancel."""

    iterator = _GatedCloseQueueIterator()
    queue = _queue(iterator)
    iterator.put(_delivery("failing"))
    stop_event = asyncio.Event()
    handler_started = asyncio.Event()
    expected_error = RuntimeError("delivery failed during consumer close")

    async def handle(_delivery: AbstractIncomingMessage) -> None:
        handler_started.set()
        await iterator.close_started.wait()
        raise expected_error

    consumer = asyncio.create_task(
        run_task_consumer(
            queue.queue,
            handle,
            concurrency=1,
            stop_event=stop_event,
        )
    )
    await _wait(handler_started)
    stop_event.set()
    await _wait(iterator.close_started)
    async with asyncio.timeout(1):
        while consumer.cancelling() == 0:  # noqa: ASYNC110
            await asyncio.sleep(0)

    assert iterator.close_cancelled.is_set() is False
    iterator.release_close.set()

    with pytest.raises(RuntimeError) as error_info:
        await consumer

    assert error_info.value is expected_error
    assert iterator.closed.is_set()
    assert iterator.close_cancelled.is_set() is False


@pytest.mark.asyncio
async def test_repeated_external_cancellation_cannot_interrupt_iterator_close() -> None:
    """Repeated forced shutdown still waits for the owned close operation."""

    iterator = _GatedCloseQueueIterator()
    queue = _queue(iterator)
    iterator.put(_delivery("running"))
    stop_event = asyncio.Event()
    handler_started = asyncio.Event()
    handler_reaped = asyncio.Event()

    async def handle(_delivery: AbstractIncomingMessage) -> None:
        handler_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            handler_reaped.set()

    consumer = asyncio.create_task(
        run_task_consumer(
            queue.queue,
            handle,
            concurrency=1,
            stop_event=stop_event,
        )
    )
    await _wait(handler_started)
    stop_event.set()
    await _wait(iterator.close_started)

    consumer.cancel()
    await asyncio.sleep(0)
    consumer.cancel()
    await asyncio.sleep(0)

    assert iterator.close_cancelled.is_set() is False
    iterator.release_close.set()

    with pytest.raises(asyncio.CancelledError):
        await consumer

    assert iterator.closed.is_set()
    assert iterator.close_cancelled.is_set() is False
    assert handler_reaped.is_set()


@pytest.mark.asyncio
async def test_intake_cancellation_and_iterator_close_failure_are_preserved() -> None:
    """A close failure cannot replace an already propagating cancellation."""

    iterator = _GatedCloseQueueIterator()
    queue = _queue(iterator)
    expected_error = ConnectionError("consumer cancel failed")
    iterator.close_error = expected_error
    consumer = asyncio.create_task(
        run_task_consumer(
            queue.queue,
            _handler(AsyncMock()),
            concurrency=1,
            stop_event=asyncio.Event(),
        )
    )
    await _wait(iterator.next_started)

    consumer.cancel()
    await _wait(iterator.close_started)
    iterator.release_close.set()

    with pytest.raises(BaseExceptionGroup) as error_info:
        await consumer

    failures = _leaf_exceptions(error_info.value)
    assert len(failures) == 2
    assert sum(isinstance(error, asyncio.CancelledError) for error in failures) == 1
    assert any(error is expected_error for error in failures)
    assert iterator.closed.is_set()
    assert iterator.close_cancelled.is_set() is False


@pytest.mark.asyncio
async def test_handler_failure_cannot_interrupt_anext_internal_close() -> None:
    """TaskGroup cancellation cannot interrupt aio-pika's anext cleanup."""

    iterator = _ClosingOnCancellationQueueIterator()
    queue = _queue(iterator)
    iterator.put(_delivery("failing"))
    stop_event = asyncio.Event()
    handler_started = asyncio.Event()
    expected_error = RuntimeError("delivery failed during anext cleanup")

    async def handle(_delivery: AbstractIncomingMessage) -> None:
        handler_started.set()
        await iterator.close_started.wait()
        raise expected_error

    consumer = asyncio.create_task(
        run_task_consumer(
            queue.queue,
            handle,
            concurrency=2,
            stop_event=stop_event,
        )
    )
    await _wait(handler_started)
    await _wait(iterator.idle_next_started)
    stop_event.set()
    await _wait(iterator.close_started)
    async with asyncio.timeout(1):
        while consumer.cancelling() == 0:  # noqa: ASYNC110
            await asyncio.sleep(0)

    assert iterator.close_cancelled.is_set() is False
    iterator.release_close.set()

    with pytest.raises(RuntimeError) as error_info:
        await consumer

    assert error_info.value is expected_error
    assert iterator.closed.is_set()
    assert iterator.close_cancelled.is_set() is False


@pytest.mark.asyncio
async def test_repeated_cancellation_cannot_interrupt_anext_internal_close() -> None:
    """Repeated forced shutdown cannot create a ghost aio-pika consumer."""

    iterator = _ClosingOnCancellationQueueIterator()
    queue = _queue(iterator)
    stop_event = asyncio.Event()
    consumer = asyncio.create_task(
        run_task_consumer(
            queue.queue,
            _handler(AsyncMock()),
            concurrency=1,
            stop_event=stop_event,
        )
    )
    await _wait(iterator.idle_next_started)
    stop_event.set()
    await _wait(iterator.close_started)

    consumer.cancel()
    await asyncio.sleep(0)
    consumer.cancel()
    await asyncio.sleep(0)

    assert iterator.close_cancelled.is_set() is False
    iterator.release_close.set()

    with pytest.raises(asyncio.CancelledError):
        await consumer

    assert iterator.closed.is_set()
    assert iterator.close_cancelled.is_set() is False


@pytest.mark.asyncio
async def test_anext_cleanup_failure_and_cancellation_are_preserved() -> None:
    """Forced shutdown retains an error from aio-pika's internal close."""

    iterator = _ClosingOnCancellationQueueIterator()
    queue = _queue(iterator)
    expected_error = ConnectionError("anext consumer cancel failed")
    iterator.close_error = expected_error
    stop_event = asyncio.Event()
    consumer = asyncio.create_task(
        run_task_consumer(
            queue.queue,
            _handler(AsyncMock()),
            concurrency=1,
            stop_event=stop_event,
        )
    )
    await _wait(iterator.idle_next_started)
    stop_event.set()
    await _wait(iterator.close_started)

    consumer.cancel()
    await asyncio.sleep(0)
    iterator.release_close.set()

    with pytest.raises(BaseExceptionGroup) as error_info:
        await consumer

    failures = _leaf_exceptions(error_info.value)
    assert len(failures) == 2
    assert sum(isinstance(error, asyncio.CancelledError) for error in failures) == 1
    assert any(error is expected_error for error in failures)
    assert iterator.closed.is_set()
    assert iterator.close_cancelled.is_set() is False


@pytest.mark.asyncio
async def test_delivery_completed_during_cancellation_is_requeued() -> None:
    """A delivery returned with owner cancellation is explicitly requeued."""

    iterator = _OwnerCancellingQueueIterator()
    queue = _queue(iterator)
    nack = AsyncMock()
    delivery = cast(
        AbstractIncomingMessage,
        SimpleNamespace(message_id="interrupted", nack=nack),
    )
    iterator.put(delivery)
    handle = AsyncMock()
    consumer = asyncio.create_task(
        run_task_consumer(
            queue.queue,
            _handler(handle),
            concurrency=1,
            stop_event=asyncio.Event(),
        )
    )
    iterator.owner_task = consumer

    with pytest.raises(asyncio.CancelledError):
        await consumer

    handle.assert_not_awaited()
    nack.assert_awaited_once_with(multiple=False, requeue=True)
    assert iterator.closed.is_set()


@pytest.mark.asyncio
async def test_delivery_returned_after_stop_and_cancellation_is_requeued() -> None:
    """A stop-cancelled anext result remains owned during forced shutdown."""

    nack = AsyncMock()
    delivery = cast(
        AbstractIncomingMessage,
        SimpleNamespace(message_id="stop-interrupted", nack=nack),
    )
    iterator = _ReturningAfterCancellationQueueIterator(delivery)
    queue = _queue(iterator)
    stop_event = asyncio.Event()
    handle = AsyncMock()
    consumer = asyncio.create_task(
        run_task_consumer(
            queue.queue,
            _handler(handle),
            concurrency=1,
            stop_event=stop_event,
        )
    )
    await _wait(iterator.next_started)
    stop_event.set()
    await _wait(iterator.next_cancelled)

    consumer.cancel()
    await asyncio.sleep(0)
    iterator.release_delivery.set()

    with pytest.raises(asyncio.CancelledError):
        await consumer

    handle.assert_not_awaited()
    nack.assert_awaited_once_with(multiple=False, requeue=True)
    assert iterator.closed.is_set()


@pytest.mark.asyncio
async def test_repeated_cancellation_cannot_interrupt_delivery_requeue() -> None:
    """Repeated forced shutdown fully reaps a compensating nack operation."""

    iterator = _OwnerCancellingQueueIterator()
    queue = _queue(iterator)
    nack_started = asyncio.Event()
    release_nack = asyncio.Event()
    nack_cancelled = asyncio.Event()

    async def nack(*, multiple: bool, requeue: bool) -> None:
        assert multiple is False
        assert requeue is True
        nack_started.set()
        try:
            await release_nack.wait()
        except asyncio.CancelledError:
            nack_cancelled.set()
            raise

    nack_mock = AsyncMock(side_effect=nack)
    iterator.put(
        cast(
            AbstractIncomingMessage,
            SimpleNamespace(message_id="interrupted", nack=nack_mock),
        )
    )
    consumer = asyncio.create_task(
        run_task_consumer(
            queue.queue,
            _handler(AsyncMock()),
            concurrency=1,
            stop_event=asyncio.Event(),
        )
    )
    iterator.owner_task = consumer
    await _wait(nack_started)

    consumer.cancel()
    await asyncio.sleep(0)
    consumer.cancel()
    await asyncio.sleep(0)

    assert nack_cancelled.is_set() is False
    release_nack.set()

    with pytest.raises(asyncio.CancelledError):
        await consumer

    nack_mock.assert_awaited_once_with(multiple=False, requeue=True)
    assert nack_cancelled.is_set() is False
    assert iterator.closed.is_set()


@pytest.mark.asyncio
async def test_delivery_requeue_failure_and_cancellation_are_preserved() -> None:
    """A compensating nack failure cannot replace owner cancellation."""

    iterator = _OwnerCancellingQueueIterator()
    queue = _queue(iterator)
    expected_error = ConnectionError("delivery requeue failed")
    nack = AsyncMock(side_effect=expected_error)
    iterator.put(
        cast(
            AbstractIncomingMessage,
            SimpleNamespace(message_id="interrupted", nack=nack),
        )
    )
    consumer = asyncio.create_task(
        run_task_consumer(
            queue.queue,
            _handler(AsyncMock()),
            concurrency=1,
            stop_event=asyncio.Event(),
        )
    )
    iterator.owner_task = consumer

    with pytest.raises(BaseExceptionGroup) as error_info:
        await consumer

    failures = _leaf_exceptions(error_info.value)
    assert len(failures) == 2
    assert sum(isinstance(error, asyncio.CancelledError) for error in failures) == 1
    assert any(error is expected_error for error in failures)
    nack.assert_awaited_once_with(multiple=False, requeue=True)
    assert iterator.closed.is_set()


@pytest.mark.asyncio
async def test_delivery_requeue_self_cancellation_is_a_failure() -> None:
    """A nack task cancelling itself cannot be mistaken for clean shutdown."""

    iterator = _OwnerCancellingQueueIterator()
    queue = _queue(iterator)
    nack = AsyncMock(side_effect=asyncio.CancelledError())
    iterator.put(
        cast(
            AbstractIncomingMessage,
            SimpleNamespace(message_id="interrupted", nack=nack),
        )
    )
    consumer = asyncio.create_task(
        run_task_consumer(
            queue.queue,
            _handler(AsyncMock()),
            concurrency=1,
            stop_event=asyncio.Event(),
        )
    )
    iterator.owner_task = consumer

    with pytest.raises(BaseExceptionGroup) as error_info:
        await consumer

    failures = _leaf_exceptions(error_info.value)
    assert len(failures) == 2
    assert sum(isinstance(error, asyncio.CancelledError) for error in failures) == 1
    assert sum(isinstance(error, TaskDeliveryRequeueCancelledError) for error in failures) == 1
    nack.assert_awaited_once_with(multiple=False, requeue=True)
    assert iterator.closed.is_set()


@pytest.mark.asyncio
async def test_delivery_completed_with_stop_is_still_owned_by_a_handler() -> None:
    """A message already returned by anext is processed even when stop also wins."""

    iterator = _GatedQueueIterator()
    queue = _queue(iterator)
    delivery = _delivery("simultaneous")
    iterator.put(delivery)
    stop_event = asyncio.Event()
    handle = AsyncMock()
    consumer = asyncio.create_task(
        run_task_consumer(
            queue.queue,
            _handler(handle),
            concurrency=1,
            stop_event=stop_event,
        )
    )
    await _wait(iterator.next_started)

    stop_event.set()
    iterator.release_delivery.set()
    await _wait(iterator.delivery_returned)
    await _finish(consumer)

    handle.assert_awaited_once_with(delivery)
    assert iterator.requeued == []
    assert iterator.close_was_concurrent is False


@pytest.mark.asyncio
async def test_unexpected_iterator_end_is_a_process_failure() -> None:
    """A consumer ending without a stop request cannot look like clean shutdown."""

    queue = _queue()
    queue.iterator.finish()

    with pytest.raises(
        TaskConsumerExitedError,
        match=r"^RabbitMQ task consumer stopped unexpectedly$",
    ):
        await run_task_consumer(
            queue.queue,
            _handler(AsyncMock()),
            concurrency=1,
            stop_event=asyncio.Event(),
        )

    assert queue.iterator.closed.is_set()


@pytest.mark.asyncio
async def test_iterator_failure_is_propagated_by_identity() -> None:
    """Broker intake failures remain visible to the process supervisor."""

    expected_error = ConnectionError("consumer channel failed")
    queue = _queue()
    queue.iterator.fail(expected_error)

    with pytest.raises(ConnectionError) as error_info:
        await run_task_consumer(
            queue.queue,
            _handler(AsyncMock()),
            concurrency=1,
            stop_event=asyncio.Event(),
        )

    assert error_info.value is expected_error
    assert queue.iterator.closed.is_set()


@pytest.mark.asyncio
async def test_handler_failure_cancels_and_reaps_its_sibling() -> None:
    """One failed delivery terminates intake and every sibling handler."""

    queue = _queue()
    failing_delivery = _delivery("failing")
    sibling_delivery = _delivery("sibling")
    queue.iterator.put(failing_delivery)
    queue.iterator.put(sibling_delivery)
    sibling_started = asyncio.Event()
    sibling_reaped = asyncio.Event()
    expected_error = RuntimeError("delivery failed")

    async def handle(delivery: AbstractIncomingMessage) -> None:
        if delivery is failing_delivery:
            await sibling_started.wait()
            raise expected_error
        sibling_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            sibling_reaped.set()

    with pytest.raises(RuntimeError) as error_info:
        await run_task_consumer(
            queue.queue,
            handle,
            concurrency=2,
            stop_event=asyncio.Event(),
        )

    assert error_info.value is expected_error
    assert sibling_reaped.is_set()
    assert queue.iterator.closed.is_set()
    assert queue.iterator.close_was_concurrent is False


@pytest.mark.asyncio
async def test_handler_originated_cancellation_fails_the_consumer() -> None:
    """A processor cancellation cannot silently consume one prefetch slot forever."""

    queue = _queue()
    queue.iterator.put(_delivery("self-cancelled"))

    async def handle(_delivery: AbstractIncomingMessage) -> None:
        raise asyncio.CancelledError

    with pytest.raises(
        TaskDeliveryCancelledError,
        match=r"^task delivery handler cancelled outside worker shutdown$",
    ) as error_info:
        async with asyncio.timeout(1):
            await run_task_consumer(
                queue.queue,
                handle,
                concurrency=1,
                stop_event=asyncio.Event(),
            )

    assert isinstance(error_info.value.__cause__, asyncio.CancelledError)
    assert queue.iterator.closed.is_set()
    assert queue.iterator.close_was_concurrent is False


@pytest.mark.asyncio
async def test_handler_task_self_cancellation_fails_the_consumer() -> None:
    """A canonical task self-cancel cannot look like worker shutdown."""

    queue = _queue()
    queue.iterator.put(_delivery("self-cancelled-task"))

    async def handle(_delivery: AbstractIncomingMessage) -> None:
        current_task = asyncio.current_task()
        assert current_task is not None
        current_task.cancel()
        await asyncio.sleep(0)

    with pytest.raises(
        TaskDeliveryCancelledError,
        match=r"^task delivery handler cancelled outside worker shutdown$",
    ) as error_info:
        async with asyncio.timeout(1):
            await run_task_consumer(
                queue.queue,
                handle,
                concurrency=1,
                stop_event=asyncio.Event(),
            )

    assert isinstance(error_info.value.__cause__, asyncio.CancelledError)
    assert queue.iterator.closed.is_set()
    assert queue.iterator.close_was_concurrent is False


@pytest.mark.asyncio
async def test_multiple_handler_failures_remain_grouped() -> None:
    """Concurrent independent failures are preserved for diagnostics."""

    queue = _queue()
    first_delivery = _delivery("first")
    second_delivery = _delivery("second")
    queue.iterator.put(first_delivery)
    queue.iterator.put(second_delivery)
    second_started = asyncio.Event()
    first_error = RuntimeError("first failed")
    second_error = ValueError("second failed during cancellation")

    async def handle(delivery: AbstractIncomingMessage) -> None:
        if delivery is first_delivery:
            await second_started.wait()
            raise first_error
        second_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise second_error from None

    with pytest.raises(BaseExceptionGroup) as error_info:
        await run_task_consumer(
            queue.queue,
            handle,
            concurrency=2,
            stop_event=asyncio.Event(),
        )

    failures = _leaf_exceptions(error_info.value)
    assert len(failures) == 3
    assert any(error is first_error for error in failures)
    assert any(error is second_error for error in failures)
    assert sum(isinstance(error, asyncio.CancelledError) for error in failures) == 1


@pytest.mark.asyncio
async def test_external_cancellation_reaps_handlers_and_intake() -> None:
    """Forced shutdown leaves no handler or iterator operation running."""

    queue = _queue()
    deliveries = (_delivery("first"), _delivery("second"))
    for delivery in deliveries:
        queue.iterator.put(delivery)

    handlers_started = asyncio.Event()
    started_count = 0
    reaped_count = 0
    all_reaped = asyncio.Event()

    async def handle(_delivery: AbstractIncomingMessage) -> None:
        nonlocal reaped_count, started_count
        started_count += 1
        if started_count == 2:
            handlers_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            reaped_count += 1
            if reaped_count == 2:
                all_reaped.set()

    consumer = asyncio.create_task(
        run_task_consumer(
            queue.queue,
            handle,
            concurrency=2,
            stop_event=asyncio.Event(),
        )
    )
    await _wait(handlers_started)
    consumer.cancel()

    with pytest.raises(asyncio.CancelledError):
        await consumer

    assert all_reaped.is_set()
    assert reaped_count == 2
    assert queue.iterator.closed.is_set()
    assert queue.iterator.close_was_concurrent is False


@pytest.mark.asyncio
async def test_external_cancellation_reaps_idle_iterator_operation() -> None:
    """Forced shutdown reaps an idle anext before closing its iterator."""

    queue = _queue()
    consumer = asyncio.create_task(
        run_task_consumer(
            queue.queue,
            _handler(AsyncMock()),
            concurrency=1,
            stop_event=asyncio.Event(),
        )
    )
    await _wait(queue.iterator.next_started)

    consumer.cancel()

    with pytest.raises(asyncio.CancelledError):
        await consumer

    assert queue.iterator.next_cancelled.is_set()
    assert queue.iterator.closed.is_set()
    assert queue.iterator.close_was_concurrent is False


@pytest.mark.asyncio
async def test_iterator_close_failure_is_propagated() -> None:
    """A failed Basic.Cancel remains a process-level infrastructure error."""

    expected_error = ConnectionError("consumer cancel failed")
    queue = _queue()
    queue.iterator.close_error = expected_error
    stop_event = asyncio.Event()
    consumer = asyncio.create_task(
        run_task_consumer(
            queue.queue,
            _handler(AsyncMock()),
            concurrency=1,
            stop_event=stop_event,
        )
    )
    await _wait(queue.iterator.next_started)
    stop_event.set()

    with pytest.raises(ConnectionError) as error_info:
        await consumer

    assert error_info.value is expected_error
    assert queue.iterator.close_was_concurrent is False


@pytest.mark.asyncio
async def test_unexpected_iterator_close_cancellation_is_a_failure() -> None:
    """A close task cancelling itself cannot look like clean shutdown."""

    queue = _queue()
    queue.iterator.close_error = asyncio.CancelledError()
    stop_event = asyncio.Event()
    consumer = asyncio.create_task(
        run_task_consumer(
            queue.queue,
            _handler(AsyncMock()),
            concurrency=1,
            stop_event=stop_event,
        )
    )
    await _wait(queue.iterator.next_started)
    stop_event.set()

    with pytest.raises(
        TaskConsumerCloseCancelledError,
        match=r"^RabbitMQ task consumer close was cancelled unexpectedly$",
    ) as error_info:
        await consumer

    assert isinstance(error_info.value.__cause__, asyncio.CancelledError)
    assert queue.iterator.closed.is_set()
    assert queue.iterator.close_was_concurrent is False
