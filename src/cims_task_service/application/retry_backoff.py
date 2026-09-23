"""Shared, deterministic retry-delay arithmetic for application policies."""

from datetime import timedelta
from math import isfinite
from typing import Final

_MICROSECONDS_PER_SECOND: Final = 1_000_000
_SECONDS_PER_DAY: Final = 86_400


def calculate_capped_exponential_retry_delay(
    attempt_number: int,
    *,
    initial_delay: timedelta,
    maximum_delay: timedelta,
    jitter_factor: float,
) -> timedelta:
    """Double from attempt one, cap, then apply equal jitter in microseconds.

    The caller supplies a factor in [0.5, 1.0]. Fractional microseconds round
    down, with a one-microsecond floor so a retry never becomes immediate.
    """

    if attempt_number < 1:
        raise ValueError("attempt_number must be at least 1")
    if initial_delay <= timedelta(0):
        raise ValueError("initial_delay must be positive")
    if maximum_delay < initial_delay:
        raise ValueError("maximum_delay must be at least initial_delay")
    if not isfinite(jitter_factor) or not (0.5 <= jitter_factor <= 1.0):
        raise ValueError("jitter_factor must be finite and between 0.5 and 1.0")

    initial_microseconds = _timedelta_microseconds(initial_delay)
    maximum_microseconds = _timedelta_microseconds(maximum_delay)
    doublings = attempt_number - 1
    if doublings >= maximum_microseconds.bit_length():
        capped_microseconds = maximum_microseconds
    else:
        capped_microseconds = min(
            initial_microseconds << doublings,
            maximum_microseconds,
        )

    jitter_numerator, jitter_denominator = jitter_factor.as_integer_ratio()
    jittered_microseconds = min(
        capped_microseconds,
        max(1, capped_microseconds * jitter_numerator // jitter_denominator),
    )
    return timedelta(microseconds=jittered_microseconds)


def _timedelta_microseconds(value: timedelta) -> int:
    return (
        value.days * _SECONDS_PER_DAY + value.seconds
    ) * _MICROSECONDS_PER_SECOND + value.microseconds
