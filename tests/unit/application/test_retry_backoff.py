"""Tests for bounded exponential retry delays with equal jitter."""

from datetime import timedelta

import pytest

from cims_task_service.application.retry_backoff import calculate_capped_exponential_retry_delay


@pytest.mark.parametrize(
    ("attempt_number", "expected_seconds"),
    [(1, 2), (2, 4), (3, 8), (4, 10), (5, 10), (10**100, 10)],
)
def test_retry_delay_doubles_from_first_attempt_until_cap(
    attempt_number: int,
    expected_seconds: int,
) -> None:
    """The first consumed attempt uses the initial delay without an extra doubling."""

    assert calculate_capped_exponential_retry_delay(
        attempt_number,
        initial_delay=timedelta(seconds=2),
        maximum_delay=timedelta(seconds=10),
        jitter_factor=1.0,
    ) == timedelta(seconds=expected_seconds)


@pytest.mark.parametrize(
    ("jitter_factor", "expected_delay"),
    [
        (0.5, timedelta(seconds=5)),
        (0.75, timedelta(seconds=7, milliseconds=500)),
        (1.0, timedelta(seconds=10)),
    ],
)
def test_retry_delay_applies_equal_jitter_after_capping(
    jitter_factor: float,
    expected_delay: timedelta,
) -> None:
    """A saturated retry still spreads across the lower and upper jitter bounds."""

    assert (
        calculate_capped_exponential_retry_delay(
            4,
            initial_delay=timedelta(seconds=2),
            maximum_delay=timedelta(seconds=10),
            jitter_factor=jitter_factor,
        )
        == expected_delay
    )


@pytest.mark.parametrize("attempt_number", [1, 2, 10**100])
def test_equal_initial_and_maximum_delays_use_a_constant_base(attempt_number: int) -> None:
    """A fixed-delay policy remains valid for every consumed attempt."""

    assert calculate_capped_exponential_retry_delay(
        attempt_number,
        initial_delay=timedelta(seconds=8),
        maximum_delay=timedelta(seconds=8),
        jitter_factor=0.75,
    ) == timedelta(seconds=6)


@pytest.mark.parametrize(
    ("attempt_number", "initial_delay", "maximum_delay", "jitter_factor", "expected_delay"),
    [
        pytest.param(
            1,
            timedelta(microseconds=1),
            timedelta(microseconds=1),
            0.5,
            timedelta(microseconds=1),
            id="positive-minimum",
        ),
        pytest.param(
            1,
            timedelta(microseconds=3),
            timedelta(seconds=1),
            0.5,
            timedelta(microseconds=1),
            id="floor-odd-microseconds",
        ),
        pytest.param(
            1,
            timedelta(microseconds=5),
            timedelta(seconds=1),
            0.6,
            timedelta(microseconds=2),
            id="floor-exact-float-rational",
        ),
        pytest.param(
            4,
            timedelta(microseconds=3),
            timedelta(microseconds=24),
            1.0,
            timedelta(microseconds=24),
            id="exact-power-of-two-cap",
        ),
        pytest.param(
            2,
            timedelta(days=1, microseconds=1),
            timedelta(days=3),
            1.0,
            timedelta(days=2, microseconds=2),
            id="duration-including-days",
        ),
        pytest.param(
            1,
            timedelta.max,
            timedelta.max,
            1.0,
            timedelta.max,
            id="maximum-duration-without-rounding",
        ),
        pytest.param(
            1,
            timedelta.max,
            timedelta.max,
            0.5,
            timedelta(days=500_000_000, microseconds=-1),
            id="maximum-duration-half-floor",
        ),
        pytest.param(
            10**100,
            timedelta(microseconds=1),
            timedelta.max,
            0.75,
            timedelta(days=750_000_000, microseconds=-1),
            id="maximum-duration-huge-attempt",
        ),
    ],
)
def test_retry_delay_preserves_bounds_and_microsecond_precision(
    attempt_number: int,
    initial_delay: timedelta,
    maximum_delay: timedelta,
    jitter_factor: float,
    expected_delay: timedelta,
) -> None:
    """Extreme inputs avoid overflow, floating-point rounding, and zero delays."""

    assert (
        calculate_capped_exponential_retry_delay(
            attempt_number,
            initial_delay=initial_delay,
            maximum_delay=maximum_delay,
            jitter_factor=jitter_factor,
        )
        == expected_delay
    )


@pytest.mark.parametrize(
    ("attempt_number", "initial_delay", "maximum_delay", "message"),
    [
        (0, timedelta(seconds=1), timedelta(seconds=2), "attempt_number"),
        (-1, timedelta(seconds=1), timedelta(seconds=2), "attempt_number"),
        (1, timedelta(0), timedelta(seconds=2), "initial_delay"),
        (1, timedelta(microseconds=-1), timedelta(seconds=2), "initial_delay"),
        (1, timedelta(seconds=1), timedelta(0), "maximum_delay"),
        (1, timedelta(seconds=1), timedelta(microseconds=-1), "maximum_delay"),
        (1, timedelta(seconds=2), timedelta(seconds=1), "maximum_delay"),
    ],
)
def test_retry_delay_rejects_invalid_attempts_and_durations(
    attempt_number: int,
    initial_delay: timedelta,
    maximum_delay: timedelta,
    message: str,
) -> None:
    """Invalid attempt numbers and delay ranges cannot produce a retry schedule."""

    with pytest.raises(ValueError, match=message):
        calculate_capped_exponential_retry_delay(
            attempt_number,
            initial_delay=initial_delay,
            maximum_delay=maximum_delay,
            jitter_factor=0.75,
        )


@pytest.mark.parametrize("jitter_factor", [0.49, 1.01, float("nan"), float("inf"), float("-inf")])
def test_retry_delay_rejects_invalid_jitter(jitter_factor: float) -> None:
    """Jitter must be a finite factor within the equal-jitter interval."""

    with pytest.raises(ValueError, match="jitter_factor"):
        calculate_capped_exponential_retry_delay(
            1,
            initial_delay=timedelta(seconds=1),
            maximum_delay=timedelta(seconds=2),
            jitter_factor=jitter_factor,
        )
