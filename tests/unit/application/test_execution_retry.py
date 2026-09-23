"""Tests for the callable execution retry policy."""

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cims_task_service.application import execution_retry as retry_module
from cims_task_service.application.execution_retry import ExecutionRetryDelayPolicy


def test_policy_uses_consumed_attempt_and_draws_fresh_jitter_on_every_call() -> None:
    """Repeated attempt numbers on different tasks must not share cached jitter."""

    jitter = Mock(side_effect=[0.5, 1.0, 0.75, 0.5])
    policy = ExecutionRetryDelayPolicy(
        initial_delay=timedelta(seconds=4),
        maximum_delay=timedelta(seconds=10),
        jitter_factor_factory=jitter,
    )
    jitter.assert_not_called()

    assert [policy(attempt) for attempt in (1, 1, 2, 3)] == [
        timedelta(seconds=2),
        timedelta(seconds=4),
        timedelta(seconds=6),
        timedelta(seconds=5),
    ]
    assert jitter.call_count == 4


def test_policy_default_draws_equal_jitter_from_system_random(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime defaults use the agreed jitter range without nondeterministic assertions."""

    uniform = Mock(return_value=0.75)
    monkeypatch.setattr(retry_module, "_SYSTEM_RANDOM", SimpleNamespace(uniform=uniform))
    policy = ExecutionRetryDelayPolicy(
        initial_delay=timedelta(seconds=4),
        maximum_delay=timedelta(seconds=10),
    )

    assert policy(1) == timedelta(seconds=3)
    uniform.assert_called_once_with(0.5, 1.0)


@pytest.mark.parametrize(
    ("initial", "maximum", "message"),
    [
        (timedelta(0), timedelta(seconds=1), "initial_delay must be positive"),
        (timedelta(microseconds=-1), timedelta(seconds=1), "initial_delay must be positive"),
        (
            timedelta(seconds=2),
            timedelta(seconds=1),
            "maximum_delay must be at least initial_delay",
        ),
    ],
)
def test_policy_rejects_invalid_durations_at_construction(
    initial: timedelta,
    maximum: timedelta,
    message: str,
) -> None:
    """Static policy errors fail before any task reaches execution recovery."""

    jitter = Mock()
    with pytest.raises(ValueError, match=f"^{message}$"):
        ExecutionRetryDelayPolicy(
            initial_delay=initial,
            maximum_delay=maximum,
            jitter_factor_factory=jitter,
        )
    jitter.assert_not_called()


@pytest.mark.parametrize("attempt_count", [0, -1])
def test_policy_rejects_unconsumed_attempts_before_drawing_jitter(attempt_count: int) -> None:
    """A retry delay belongs to a started execution, not a future one."""

    jitter = Mock()
    policy = ExecutionRetryDelayPolicy(
        initial_delay=timedelta(seconds=1),
        maximum_delay=timedelta(seconds=2),
        jitter_factor_factory=jitter,
    )

    with pytest.raises(ValueError, match=r"^attempt_count must be at least 1$"):
        policy(attempt_count)
    jitter.assert_not_called()


@pytest.mark.parametrize("invalid_factor", [0.49, 1.01, float("nan"), float("inf"), -float("inf")])
def test_policy_validates_each_new_jitter_factor(invalid_factor: float) -> None:
    """A successful previous sample cannot hide a later invalid factor."""

    jitter = Mock(side_effect=[1.0, invalid_factor])
    policy = ExecutionRetryDelayPolicy(
        initial_delay=timedelta(seconds=1),
        maximum_delay=timedelta(seconds=2),
        jitter_factor_factory=jitter,
    )

    assert policy(1) == timedelta(seconds=1)
    with pytest.raises(ValueError, match="jitter_factor must be finite and between"):
        policy(1)
    assert jitter.call_count == 2
