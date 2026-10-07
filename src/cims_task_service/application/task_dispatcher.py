"""Reliable task outbox publication orchestration."""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from math import isfinite
from random import SystemRandom
from typing import Final, Protocol

from cims_task_service.application.retry_backoff import (
    calculate_capped_exponential_retry_delay,
)
from cims_task_service.infrastructure.database.outbox_repository import (
    ClaimedOutboxEvent,
    OutboxRepository,
)
from cims_task_service.infrastructure.database.session import AsyncSessionFactory
from cims_task_service.infrastructure.messaging.topology import TASK_ROUTING_KEY

_MINIMUM_JITTER_FACTOR: Final = 0.5
_MAXIMUM_JITTER_FACTOR: Final = 1.0
_SYSTEM_RANDOM: Final = SystemRandom()
MAX_DISPATCH_BATCH_SIZE: Final = 100


class TaskEventPublisher(Protocol):
    """Port implemented by a confirm-aware task event publisher."""

    async def publish(self, event: ClaimedOutboxEvent) -> None:
        """Publish one claimed event or raise when delivery is not confirmed."""


@dataclass(frozen=True, slots=True)
class DispatchBatchResult:
    """Observable outcome of one finite dispatcher pass."""

    claimed: int
    published: int
    rescheduled: int
    lost_ownership: int


type DispatchOnce = Callable[[], Awaitable[DispatchBatchResult]]


class _DispatchOutcome(Enum):
    PUBLISHED = "published"
    RESCHEDULED = "rescheduled"
    LOST_OWNERSHIP = "lost_ownership"


def calculate_publish_retry_delay(
    publish_attempts: int,
    *,
    initial_delay: timedelta,
    maximum_delay: timedelta,
    jitter_factor: float,
) -> timedelta:
    """Return overflow-safe capped exponential backoff with equal jitter."""

    if publish_attempts < 1:
        raise ValueError("publish_attempts must be at least 1")
    return calculate_capped_exponential_retry_delay(
        publish_attempts,
        initial_delay=initial_delay,
        maximum_delay=maximum_delay,
        jitter_factor=jitter_factor,
    )


def summarize_publication_failure(error: Exception) -> str:
    """Return a stable diagnostic category without persisting exception data."""

    return type(error).__name__


class TaskOutboxDispatcher:
    """Claim, publish, and independently finalize one task event batch."""

    def __init__(
        self,
        session_factory: AsyncSessionFactory,
        publisher: TaskEventPublisher,
        *,
        batch_size: int,
        lease_duration: timedelta,
        retry_initial_delay: timedelta,
        retry_maximum_delay: timedelta,
        jitter_factor_factory: Callable[[], float] | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if batch_size > MAX_DISPATCH_BATCH_SIZE:
            raise ValueError(f"batch_size must be at most {MAX_DISPATCH_BATCH_SIZE}")
        if lease_duration <= timedelta(0):
            raise ValueError("lease_duration must be positive")
        if retry_initial_delay <= timedelta(0):
            raise ValueError("retry_initial_delay must be positive")
        if retry_maximum_delay < retry_initial_delay:
            raise ValueError("retry_maximum_delay must be at least retry_initial_delay")

        self._session_factory = session_factory
        self._publisher = publisher
        self._batch_size = batch_size
        self._lease_duration = lease_duration
        self._retry_initial_delay = retry_initial_delay
        self._retry_maximum_delay = retry_maximum_delay
        self._jitter_factor_factory = (
            _equal_jitter_factor if jitter_factor_factory is None else jitter_factor_factory
        )

    async def dispatch_once(self) -> DispatchBatchResult:
        """Dispatch at most one claimed batch and wait for every finalization."""

        claimed_events = await self._claim_batch()
        if not claimed_events:
            return DispatchBatchResult(
                claimed=0,
                published=0,
                rescheduled=0,
                lost_ownership=0,
            )

        tasks: list[asyncio.Task[_DispatchOutcome | Exception]] = []
        async with asyncio.TaskGroup() as task_group:
            tasks.extend(
                task_group.create_task(self._capture_dispatch_result(event))
                for event in claimed_events
            )

        raw_results = tuple(task.result() for task in tasks)
        outcomes = _unwrap_dispatch_outcomes(raw_results)
        return DispatchBatchResult(
            claimed=len(claimed_events),
            published=outcomes.count(_DispatchOutcome.PUBLISHED),
            rescheduled=outcomes.count(_DispatchOutcome.RESCHEDULED),
            lost_ownership=outcomes.count(_DispatchOutcome.LOST_OWNERSHIP),
        )

    async def _claim_batch(self) -> tuple[ClaimedOutboxEvent, ...]:
        async with self._session_factory.begin() as session:
            claimed_events = await OutboxRepository(session).claim_batch(
                event_type=TASK_ROUTING_KEY,
                batch_size=self._batch_size,
                lease_duration=self._lease_duration,
            )

        return claimed_events

    async def _dispatch_event(self, event: ClaimedOutboxEvent) -> _DispatchOutcome:
        try:
            await self._publisher.publish(event)
        except Exception as error:
            retry_delay = calculate_publish_retry_delay(
                event.publish_attempts,
                initial_delay=self._retry_initial_delay,
                maximum_delay=self._retry_maximum_delay,
                jitter_factor=self._jitter_factor_factory(),
            )
            rescheduled = await self._reschedule(
                event,
                retry_delay=retry_delay,
                failure_summary=summarize_publication_failure(error),
            )
            return _DispatchOutcome.RESCHEDULED if rescheduled else _DispatchOutcome.LOST_OWNERSHIP

        published = await self._mark_published(event)
        return _DispatchOutcome.PUBLISHED if published else _DispatchOutcome.LOST_OWNERSHIP

    async def _capture_dispatch_result(
        self,
        event: ClaimedOutboxEvent,
    ) -> _DispatchOutcome | Exception:
        try:
            return await self._dispatch_event(event)
        except Exception as error:
            return error

    async def _mark_published(self, event: ClaimedOutboxEvent) -> bool:
        async with self._session_factory.begin() as session:
            published = await OutboxRepository(session).mark_published(
                event.id,
                publisher_token=event.publisher_token,
            )

        return published

    async def _reschedule(
        self,
        event: ClaimedOutboxEvent,
        *,
        retry_delay: timedelta,
        failure_summary: str,
    ) -> bool:
        async with self._session_factory.begin() as session:
            rescheduled = await OutboxRepository(session).reschedule(
                event.id,
                publisher_token=event.publisher_token,
                retry_delay=retry_delay,
                failure_summary=failure_summary,
            )

        return rescheduled


async def run_dispatcher_loop(
    dispatch_once: DispatchOnce,
    *,
    stop_event: asyncio.Event,
    poll_interval_seconds: float,
) -> None:
    """Drain ready batches and wait interruptibly whenever the outbox is idle."""

    if not isfinite(poll_interval_seconds) or poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be finite and positive")

    while not stop_event.is_set():
        result = await dispatch_once()
        if stop_event.is_set():
            return
        if result.claimed > 0:
            continue
        await _wait_for_stop(
            stop_event,
            poll_interval_seconds=poll_interval_seconds,
        )


def _equal_jitter_factor() -> float:
    return _SYSTEM_RANDOM.uniform(_MINIMUM_JITTER_FACTOR, _MAXIMUM_JITTER_FACTOR)


async def _wait_for_stop(
    stop_event: asyncio.Event,
    *,
    poll_interval_seconds: float,
) -> None:
    idle_timeout = asyncio.timeout(poll_interval_seconds)
    try:
        async with idle_timeout:
            await stop_event.wait()
    except TimeoutError:
        if not idle_timeout.expired():
            raise


def _unwrap_dispatch_outcomes(
    raw_results: Sequence[_DispatchOutcome | Exception],
) -> tuple[_DispatchOutcome, ...]:
    failures = [result for result in raw_results if isinstance(result, Exception)]
    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise ExceptionGroup("outbox event dispatch failed", failures)

    return tuple(result for result in raw_results if isinstance(result, _DispatchOutcome))
