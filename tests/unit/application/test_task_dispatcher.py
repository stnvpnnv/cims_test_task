"""Tests for one-batch task outbox dispatch orchestration."""

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace, TracebackType
from typing import cast
from unittest.mock import AsyncMock, Mock, call
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from cims_task_service.application import task_dispatcher as task_dispatcher_module
from cims_task_service.application.task_dispatcher import (
    MAX_DISPATCH_BATCH_SIZE,
    DispatchBatchResult,
    TaskEventPublisher,
    TaskOutboxDispatcher,
    calculate_publish_retry_delay,
    summarize_publication_failure,
)
from cims_task_service.infrastructure.database.outbox_repository import ClaimedOutboxEvent
from cims_task_service.infrastructure.database.session import AsyncSessionFactory
from cims_task_service.infrastructure.messaging.topology import TASK_ROUTING_KEY

_CREATED_AT = datetime(2026, 9, 15, 3, 30, tzinfo=UTC)
_LEASE_DURATION = timedelta(seconds=30)
_RETRY_INITIAL_DELAY = timedelta(seconds=2)
_RETRY_MAXIMUM_DELAY = timedelta(seconds=10)


@dataclass(frozen=True, slots=True)
class _DispatcherHarness:
    dispatcher: TaskOutboxDispatcher
    session_factory: AsyncSessionFactory
    begin: Mock
    sessions: tuple[AsyncSession, ...]
    transactions: tuple[AsyncMock, ...]
    repository_factory: Mock
    claim_batch: AsyncMock
    publish: AsyncMock
    mark_published: AsyncMock
    reschedule: AsyncMock


def _event(index: int, *, publish_attempts: int = 1) -> ClaimedOutboxEvent:
    event_id = UUID(int=index)
    task_id = UUID(int=100 + index)
    publisher_token = UUID(int=200 + index)
    return ClaimedOutboxEvent(
        id=event_id,
        task_id=task_id,
        event_type=TASK_ROUTING_KEY,
        payload={"task_id": str(task_id), "dispatch_token": str(UUID(int=300 + index))},
        message_priority=3,
        created_at=_CREATED_AT,
        available_at=_CREATED_AT,
        publish_attempts=publish_attempts,
        publisher_token=publisher_token,
        lease_expires_at=_CREATED_AT + _LEASE_DURATION,
    )


def _half_jitter() -> float:
    return 0.5


def _transaction(
    session: AsyncSession,
    *,
    on_exit: Callable[[], None] | None = None,
    exit_error: BaseException | None = None,
) -> AsyncMock:
    transaction = AsyncMock()
    transaction.__aenter__.return_value = session

    def exit_transaction(
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> bool:
        if on_exit is not None:
            on_exit()
        if exit_error is not None:
            raise exit_error
        return False

    transaction.__aexit__.side_effect = exit_transaction
    return transaction


def _dispatcher_harness(
    monkeypatch: pytest.MonkeyPatch,
    events: Sequence[ClaimedOutboxEvent],
    *,
    claim_batch: AsyncMock | None = None,
    publish: AsyncMock | None = None,
    mark_published: AsyncMock | None = None,
    reschedule: AsyncMock | None = None,
    claim_on_exit: Callable[[], None] | None = None,
    claim_exit_error: BaseException | None = None,
    final_exit_errors: Sequence[BaseException | None] | None = None,
    jitter_factor_factory: Callable[[], float] | None = _half_jitter,
) -> _DispatcherHarness:
    claimed_events = tuple(events)
    claim_session = cast(AsyncSession, object())
    final_sessions = tuple(cast(AsyncSession, object()) for _event_item in claimed_events)
    sessions = (claim_session, *final_sessions)

    supplied_exit_errors = tuple(final_exit_errors or ())
    padded_exit_errors = supplied_exit_errors + (None,) * (
        len(claimed_events) - len(supplied_exit_errors)
    )
    transactions = (
        _transaction(
            claim_session,
            on_exit=claim_on_exit,
            exit_error=claim_exit_error,
        ),
        *(
            _transaction(session, exit_error=exit_error)
            for session, exit_error in zip(
                final_sessions,
                padded_exit_errors,
                strict=True,
            )
        ),
    )
    begin = Mock(side_effect=transactions)
    session_factory = cast(
        AsyncSessionFactory,
        SimpleNamespace(begin=begin),
    )

    claim_batch_mock = claim_batch or AsyncMock(return_value=claimed_events)
    publish_mock = publish or AsyncMock(return_value=None)
    mark_published_mock = mark_published or AsyncMock(return_value=True)
    reschedule_mock = reschedule or AsyncMock(return_value=True)
    claim_repository = SimpleNamespace(claim_batch=claim_batch_mock)
    final_repository = SimpleNamespace(
        mark_published=mark_published_mock,
        reschedule=reschedule_mock,
    )
    repository_factory = Mock(
        side_effect=(claim_repository, *(final_repository for _event_item in claimed_events))
    )
    monkeypatch.setattr(
        task_dispatcher_module,
        "OutboxRepository",
        repository_factory,
    )

    publisher = cast(TaskEventPublisher, SimpleNamespace(publish=publish_mock))
    dispatcher = TaskOutboxDispatcher(
        session_factory,
        publisher,
        batch_size=10,
        lease_duration=_LEASE_DURATION,
        retry_initial_delay=_RETRY_INITIAL_DELAY,
        retry_maximum_delay=_RETRY_MAXIMUM_DELAY,
        jitter_factor_factory=jitter_factor_factory,
    )
    return _DispatcherHarness(
        dispatcher=dispatcher,
        session_factory=session_factory,
        begin=begin,
        sessions=sessions,
        transactions=transactions,
        repository_factory=repository_factory,
        claim_batch=claim_batch_mock,
        publish=publish_mock,
        mark_published=mark_published_mock,
        reschedule=reschedule_mock,
    )


@pytest.mark.parametrize(
    (
        "batch_size",
        "lease_duration",
        "retry_initial_delay",
        "retry_maximum_delay",
        "message",
    ),
    [
        (
            0,
            timedelta(seconds=1),
            timedelta(seconds=1),
            timedelta(seconds=2),
            "batch_size must be at least 1",
        ),
        (
            1,
            timedelta(0),
            timedelta(seconds=1),
            timedelta(seconds=2),
            "lease_duration must be positive",
        ),
        (
            MAX_DISPATCH_BATCH_SIZE + 1,
            timedelta(seconds=1),
            timedelta(seconds=1),
            timedelta(seconds=2),
            f"batch_size must be at most {MAX_DISPATCH_BATCH_SIZE}",
        ),
        (
            1,
            timedelta(seconds=1),
            timedelta(0),
            timedelta(seconds=2),
            "retry_initial_delay must be positive",
        ),
        (
            1,
            timedelta(seconds=1),
            timedelta(seconds=2),
            timedelta(seconds=1),
            "retry_maximum_delay must be at least retry_initial_delay",
        ),
    ],
)
def test_dispatcher_rejects_invalid_batch_and_retry_policy(
    batch_size: int,
    lease_duration: timedelta,
    retry_initial_delay: timedelta,
    retry_maximum_delay: timedelta,
    message: str,
) -> None:
    """Invalid operational settings fail before database or broker access."""

    begin = Mock()
    publish = AsyncMock()

    with pytest.raises(ValueError, match=f"^{message}$"):
        TaskOutboxDispatcher(
            cast(AsyncSessionFactory, SimpleNamespace(begin=begin)),
            cast(TaskEventPublisher, SimpleNamespace(publish=publish)),
            batch_size=batch_size,
            lease_duration=lease_duration,
            retry_initial_delay=retry_initial_delay,
            retry_maximum_delay=retry_maximum_delay,
        )

    begin.assert_not_called()
    publish.assert_not_called()


@pytest.mark.parametrize(
    ("publish_attempts", "jitter_factor", "expected_delay"),
    [
        (1, 0.5, timedelta(seconds=1)),
        (1, 1.0, timedelta(seconds=2)),
        (2, 0.75, timedelta(seconds=3)),
        (4, 1.0, timedelta(seconds=10)),
        (10**100, 1.0, timedelta(seconds=10)),
        (1, 0.5, timedelta(microseconds=1)),
        (1, 1.0, timedelta.max),
    ],
)
def test_retry_delay_uses_equal_jitter_and_saturates_without_overflow(
    publish_attempts: int,
    jitter_factor: float,
    expected_delay: timedelta,
) -> None:
    """The retry schedule doubles, caps, jitters, and handles huge attempts."""

    if expected_delay in {timedelta(microseconds=1), timedelta.max}:
        initial_delay = expected_delay
        maximum_delay = expected_delay
    else:
        initial_delay = _RETRY_INITIAL_DELAY
        maximum_delay = _RETRY_MAXIMUM_DELAY

    assert (
        calculate_publish_retry_delay(
            publish_attempts,
            initial_delay=initial_delay,
            maximum_delay=maximum_delay,
            jitter_factor=jitter_factor,
        )
        == expected_delay
    )


@pytest.mark.parametrize(
    (
        "publish_attempts",
        "initial_delay",
        "maximum_delay",
        "jitter_factor",
        "message",
    ),
    [
        (0, timedelta(seconds=1), timedelta(seconds=2), 0.5, "publish_attempts"),
        (1, timedelta(0), timedelta(seconds=2), 0.5, "initial_delay"),
        (1, timedelta(seconds=2), timedelta(seconds=1), 0.5, "maximum_delay"),
        (1, timedelta(seconds=1), timedelta(seconds=2), 0.49, "jitter_factor"),
        (1, timedelta(seconds=1), timedelta(seconds=2), 1.01, "jitter_factor"),
        (1, timedelta(seconds=1), timedelta(seconds=2), float("inf"), "jitter_factor"),
        (1, timedelta(seconds=1), timedelta(seconds=2), float("nan"), "jitter_factor"),
    ],
)
def test_retry_delay_rejects_invalid_inputs(
    publish_attempts: int,
    initial_delay: timedelta,
    maximum_delay: timedelta,
    jitter_factor: float,
    message: str,
) -> None:
    """Broken retry inputs cannot silently create an unsafe schedule."""

    with pytest.raises(ValueError, match=message):
        calculate_publish_retry_delay(
            publish_attempts,
            initial_delay=initial_delay,
            maximum_delay=maximum_delay,
            jitter_factor=jitter_factor,
        )


def test_failure_summary_never_persists_arbitrary_exception_data() -> None:
    """Persisted diagnostics retain a useful type without any secret-bearing text."""

    error = ConnectionError(
        "broker unavailable\n"
        "at amqp://dispatcher:top-secret@rabbitmq/cims "
        "{'password': 'hunter two'} Authorization: Bearer access-token"
    )

    assert summarize_publication_failure(error) == "ConnectionError"


def test_failure_summary_is_stable_for_empty_and_oversized_messages() -> None:
    """The category does not depend on exception message presence or size."""

    assert summarize_publication_failure(RuntimeError()) == "RuntimeError"
    assert summarize_publication_failure(RuntimeError("x" * 3_000)) == "RuntimeError"


@pytest.mark.asyncio
async def test_empty_claim_commits_without_publishing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An idle pass performs one short read transaction and no broker work."""

    harness = _dispatcher_harness(monkeypatch, ())

    result = await harness.dispatcher.dispatch_once()

    assert result == DispatchBatchResult(
        claimed=0,
        published=0,
        rescheduled=0,
        lost_ownership=0,
    )
    harness.begin.assert_called_once_with()
    harness.repository_factory.assert_called_once_with(harness.sessions[0])
    harness.claim_batch.assert_awaited_once_with(
        event_type=TASK_ROUTING_KEY,
        batch_size=10,
        lease_duration=_LEASE_DURATION,
    )
    harness.transactions[0].__aexit__.assert_awaited_once_with(None, None, None)
    harness.publish.assert_not_awaited()
    harness.mark_published.assert_not_awaited()
    harness.reschedule.assert_not_awaited()


@pytest.mark.asyncio
async def test_confirmed_event_is_marked_after_the_claim_transaction_commits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RabbitMQ I/O starts only after reservation commit and uses a fresh session."""

    event = _event(1)
    claim_committed = False

    def record_claim_commit() -> None:
        nonlocal claim_committed
        claim_committed = True

    def assert_claim_committed(published_event: ClaimedOutboxEvent) -> None:
        assert claim_committed is True
        assert published_event is event

    publish = AsyncMock(side_effect=assert_claim_committed)
    harness = _dispatcher_harness(
        monkeypatch,
        (event,),
        publish=publish,
        claim_on_exit=record_claim_commit,
    )

    result = await harness.dispatcher.dispatch_once()

    assert result == DispatchBatchResult(1, 1, 0, 0)
    harness.repository_factory.assert_has_calls(
        [call(harness.sessions[0]), call(harness.sessions[1])]
    )
    assert harness.sessions[0] is not harness.sessions[1]
    harness.mark_published.assert_awaited_once_with(
        event.id,
        publisher_token=event.publisher_token,
    )
    harness.reschedule.assert_not_awaited()
    for transaction in harness.transactions:
        transaction.__aexit__.assert_awaited_once_with(None, None, None)


@pytest.mark.asyncio
async def test_publish_failure_is_rescheduled_with_backoff_and_safe_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broker failure releases the lease with deterministic retry metadata."""

    event = _event(2, publish_attempts=3)
    expected_error = ConnectionError(
        "cannot reach amqp://user:password@rabbitmq/cims\npassword=visible"
    )
    publish = AsyncMock(side_effect=expected_error)
    random_uniform = Mock(return_value=0.75)
    monkeypatch.setattr(
        task_dispatcher_module,
        "_SYSTEM_RANDOM",
        SimpleNamespace(uniform=random_uniform),
    )
    harness = _dispatcher_harness(
        monkeypatch,
        (event,),
        publish=publish,
        jitter_factor_factory=None,
    )

    result = await harness.dispatcher.dispatch_once()

    assert result == DispatchBatchResult(1, 0, 1, 0)
    random_uniform.assert_called_once_with(0.5, 1.0)
    harness.reschedule.assert_awaited_once_with(
        event.id,
        publisher_token=event.publisher_token,
        retry_delay=timedelta(seconds=6),
        failure_summary="ConnectionError",
    )
    harness.mark_published.assert_not_awaited()


@pytest.mark.parametrize("publisher_fails", [False, True])
@pytest.mark.asyncio
async def test_lost_fencing_token_is_a_normal_outcome(
    monkeypatch: pytest.MonkeyPatch,
    publisher_fails: bool,
) -> None:
    """Cancellation or lease takeover causes no stale-owner retry writes."""

    event = _event(3)
    publish_error = ConnectionError("broker unavailable") if publisher_fails else None
    harness = _dispatcher_harness(
        monkeypatch,
        (event,),
        publish=AsyncMock(return_value=None, side_effect=publish_error),
        mark_published=AsyncMock(return_value=False),
        reschedule=AsyncMock(return_value=False),
    )

    result = await harness.dispatcher.dispatch_once()

    assert result == DispatchBatchResult(1, 0, 0, 1)
    if publisher_fails:
        harness.reschedule.assert_awaited_once()
        harness.mark_published.assert_not_awaited()
    else:
        harness.mark_published.assert_awaited_once()
        harness.reschedule.assert_not_awaited()


@pytest.mark.asyncio
async def test_batch_is_published_concurrently_and_failures_do_not_cancel_siblings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The finite batch is one bounded wave with independent event outcomes."""

    events = (_event(4), _event(5), _event(6))
    failed_event = events[1]
    started_ids: set[UUID] = set()
    all_started = asyncio.Event()
    release = asyncio.Event()

    async def publish_event(event: ClaimedOutboxEvent) -> None:
        started_ids.add(event.id)
        if len(started_ids) == len(events):
            all_started.set()
        await release.wait()
        if event is failed_event:
            raise ConnectionError("broker rejected message")

    harness = _dispatcher_harness(
        monkeypatch,
        events,
        publish=AsyncMock(side_effect=publish_event),
    )
    dispatch_task = asyncio.create_task(harness.dispatcher.dispatch_once())

    try:
        safety_timeout = asyncio.timeout(1)
        async with safety_timeout:
            await all_started.wait()
            assert started_ids == {event.id for event in events}
            harness.begin.assert_called_once_with()
            harness.repository_factory.assert_called_once_with(harness.sessions[0])
            harness.mark_published.assert_not_awaited()
            harness.reschedule.assert_not_awaited()
            release.set()
            result = await dispatch_task
        assert safety_timeout.expired() is False
    finally:
        release.set()
        if not dispatch_task.done():
            dispatch_task.cancel()
        await asyncio.gather(dispatch_task, return_exceptions=True)

    assert result == DispatchBatchResult(3, 2, 1, 0)
    assert harness.publish.await_count == 3
    assert harness.mark_published.await_count == 2
    harness.reschedule.assert_awaited_once_with(
        failed_event.id,
        publisher_token=failed_event.publisher_token,
        retry_delay=timedelta(seconds=1),
        failure_summary="ConnectionError",
    )
    finalization_sessions = [
        repository_call.args[0] for repository_call in harness.repository_factory.call_args_list[1:]
    ]
    assert len({id(session) for session in finalization_sessions}) == len(events)


@pytest.mark.asyncio
async def test_cancellation_is_not_converted_into_a_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cooperative shutdown cancels publication and leaves lease recovery to PostgreSQL."""

    event = _event(7)
    publish_started = asyncio.Event()
    never_released = asyncio.Event()

    async def stalled_publish(_event_item: ClaimedOutboxEvent) -> None:
        publish_started.set()
        await never_released.wait()

    harness = _dispatcher_harness(
        monkeypatch,
        (event,),
        publish=AsyncMock(side_effect=stalled_publish),
    )
    dispatch_task = asyncio.create_task(harness.dispatcher.dispatch_once())
    safety_timeout = asyncio.timeout(1)

    async with safety_timeout:
        await publish_started.wait()
        dispatch_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await dispatch_task

    assert safety_timeout.expired() is False
    harness.mark_published.assert_not_awaited()
    harness.reschedule.assert_not_awaited()
    harness.begin.assert_called_once_with()


@pytest.mark.asyncio
async def test_publisher_originated_cancellation_is_propagated_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TaskGroup child cancellation remains a shutdown signal, not a retry error."""

    event = _event(15)
    expected_error = asyncio.CancelledError()
    harness = _dispatcher_harness(
        monkeypatch,
        (event,),
        publish=AsyncMock(side_effect=expected_error),
    )

    with pytest.raises(asyncio.CancelledError) as error_info:
        await harness.dispatcher.dispatch_once()

    assert error_info.value is expected_error
    harness.mark_published.assert_not_awaited()
    harness.reschedule.assert_not_awaited()
    harness.begin.assert_called_once_with()


@pytest.mark.asyncio
async def test_claim_failure_rolls_back_and_never_reaches_the_broker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reservation failure propagates through its transaction owner."""

    expected_error = OSError("database unavailable")
    harness = _dispatcher_harness(
        monkeypatch,
        (_event(8),),
        claim_batch=AsyncMock(side_effect=expected_error),
    )

    with pytest.raises(OSError, match=r"^database unavailable$") as error_info:
        await harness.dispatcher.dispatch_once()

    assert error_info.value is expected_error
    exit_call = harness.transactions[0].__aexit__.await_args
    assert exit_call is not None
    assert exit_call.args[0] is OSError
    assert exit_call.args[1] is expected_error
    assert exit_call.args[2] is not None
    harness.publish.assert_not_awaited()
    harness.mark_published.assert_not_awaited()
    harness.reschedule.assert_not_awaited()


@pytest.mark.asyncio
async def test_claim_commit_failure_never_reaches_the_broker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Events are not published when reservation commit has an unknown outcome."""

    event = _event(16)
    expected_error = OSError("reservation commit outcome unknown")
    harness = _dispatcher_harness(
        monkeypatch,
        (event,),
        claim_exit_error=expected_error,
    )

    with pytest.raises(
        OSError,
        match=r"^reservation commit outcome unknown$",
    ) as error_info:
        await harness.dispatcher.dispatch_once()

    assert error_info.value is expected_error
    harness.claim_batch.assert_awaited_once()
    harness.publish.assert_not_awaited()
    harness.mark_published.assert_not_awaited()
    harness.reschedule.assert_not_awaited()
    harness.begin.assert_called_once_with()
    harness.repository_factory.assert_called_once_with(harness.sessions[0])


@pytest.mark.asyncio
async def test_finalization_failure_waits_for_siblings_then_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One database failure cannot abandon another event in the same wave."""

    events = (_event(9), _event(10))
    expected_error = OSError("write failed")
    sibling_started = asyncio.Event()
    failure_raised = asyncio.Event()
    release_sibling = asyncio.Event()

    async def finalize_event(event_id: UUID, *, publisher_token: UUID) -> bool:
        assert publisher_token in {event.publisher_token for event in events}
        if event_id == events[0].id:
            await sibling_started.wait()
            failure_raised.set()
            raise expected_error
        sibling_started.set()
        await release_sibling.wait()
        return True

    harness = _dispatcher_harness(
        monkeypatch,
        events,
        mark_published=AsyncMock(side_effect=finalize_event),
    )
    dispatch_task = asyncio.create_task(harness.dispatcher.dispatch_once())

    try:
        safety_timeout = asyncio.timeout(1)
        async with safety_timeout:
            await failure_raised.wait()
            assert dispatch_task.done() is False
            release_sibling.set()
            with pytest.raises(OSError, match=r"^write failed$") as error_info:
                await dispatch_task
        assert safety_timeout.expired() is False
    finally:
        release_sibling.set()
        if not dispatch_task.done():
            dispatch_task.cancel()
        await asyncio.gather(dispatch_task, return_exceptions=True)

    assert error_info.value is expected_error
    assert harness.publish.await_count == 2
    assert harness.mark_published.await_count == 2
    harness.reschedule.assert_not_awaited()


@pytest.mark.asyncio
async def test_confirm_commit_failure_is_not_rescheduled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An uncertain mark commit leaves the lease intact instead of duplicating a write."""

    event = _event(11)
    expected_error = OSError("commit outcome unknown")
    harness = _dispatcher_harness(
        monkeypatch,
        (event,),
        final_exit_errors=(expected_error,),
    )

    with pytest.raises(OSError, match=r"^commit outcome unknown$") as error_info:
        await harness.dispatcher.dispatch_once()

    assert error_info.value is expected_error
    harness.publish.assert_awaited_once_with(event)
    harness.mark_published.assert_awaited_once_with(
        event.id,
        publisher_token=event.publisher_token,
    )
    harness.reschedule.assert_not_awaited()


@pytest.mark.asyncio
async def test_reschedule_commit_failure_is_not_retried_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An uncertain retry commit also relies on lease expiry for recovery."""

    event = _event(12)
    expected_error = OSError("commit outcome unknown")
    harness = _dispatcher_harness(
        monkeypatch,
        (event,),
        publish=AsyncMock(side_effect=ConnectionError("broker unavailable")),
        final_exit_errors=(expected_error,),
    )

    with pytest.raises(OSError, match=r"^commit outcome unknown$") as error_info:
        await harness.dispatcher.dispatch_once()

    assert error_info.value is expected_error
    harness.reschedule.assert_awaited_once()
    harness.mark_published.assert_not_awaited()


@pytest.mark.asyncio
async def test_multiple_finalization_failures_are_reported_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Independent infrastructure failures remain visible after the complete wave."""

    events = (_event(13), _event(14))
    failures = (
        OSError("first database write failed"),
        RuntimeError("second database write failed"),
    )

    def fail_finalization(event_id: UUID, *, publisher_token: UUID) -> bool:
        assert publisher_token in {event.publisher_token for event in events}
        if event_id == events[0].id:
            raise failures[0]
        raise failures[1]

    harness = _dispatcher_harness(
        monkeypatch,
        events,
        mark_published=AsyncMock(side_effect=fail_finalization),
    )

    with pytest.raises(ExceptionGroup, match="outbox event dispatch failed") as error_info:
        await harness.dispatcher.dispatch_once()

    assert error_info.value.exceptions == failures
    assert harness.mark_published.await_count == 2
    harness.reschedule.assert_not_awaited()
