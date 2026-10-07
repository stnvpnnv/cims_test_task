"""Backoff policy for retryable task execution failures."""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from random import SystemRandom
from typing import Final

from cims_task_service.application.retry_backoff import (
    calculate_capped_exponential_retry_delay,
)

_SYSTEM_RANDOM: Final = SystemRandom()


def _equal_jitter_factor() -> float:
    return _SYSTEM_RANDOM.uniform(0.5, 1.0)


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionRetryDelayPolicy:
    """Calculate a fresh delay after each consumed execution attempt.

    Attempt one starts with initial_delay before jitter; subsequent attempts
    double that base up to maximum_delay. Equal jitter scales the capped base
    to 50-100 percent, so initial_delay is not a minimum wait. The caller owns
    the attempt limit and persists the delay; this policy does not schedule work.
    """

    initial_delay: timedelta
    maximum_delay: timedelta
    jitter_factor_factory: Callable[[], float] = field(
        default=_equal_jitter_factor,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.initial_delay <= timedelta(0):
            raise ValueError("initial_delay must be positive")
        if self.maximum_delay < self.initial_delay:
            raise ValueError("maximum_delay must be at least initial_delay")

    def __call__(self, attempt_count: int) -> timedelta:
        """Return backoff for the already consumed attempt, without incrementing it."""

        if attempt_count < 1:
            raise ValueError("attempt_count must be at least 1")
        return calculate_capped_exponential_retry_delay(
            attempt_count,
            initial_delay=self.initial_delay,
            maximum_delay=self.maximum_delay,
            jitter_factor=self.jitter_factor_factory(),
        )
