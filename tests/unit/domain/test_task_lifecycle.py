"""Tests for the task lifecycle policy."""

from itertools import product

import pytest

from cims_task_service.domain.task import TaskStatus
from cims_task_service.domain.task_lifecycle import (
    InvalidTaskTransitionError,
    can_transition,
    ensure_task_transition,
    is_terminal,
)

_EXPECTED_TRANSITIONS = {
    (TaskStatus.NEW, TaskStatus.PENDING),
    (TaskStatus.NEW, TaskStatus.CANCELLED),
    (TaskStatus.PENDING, TaskStatus.IN_PROGRESS),
    (TaskStatus.PENDING, TaskStatus.CANCELLED),
    (TaskStatus.IN_PROGRESS, TaskStatus.PENDING),
    (TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED),
    (TaskStatus.IN_PROGRESS, TaskStatus.FAILED),
    (TaskStatus.IN_PROGRESS, TaskStatus.CANCELLED),
}


def test_transition_matrix_matches_the_architecture() -> None:
    """Only the documented status transitions are permitted."""

    actual_transitions = {
        (current_status, target_status)
        for current_status, target_status in product(TaskStatus, repeat=2)
        if can_transition(current_status, target_status)
    }

    assert actual_transitions == _EXPECTED_TRANSITIONS


def test_terminal_statuses_have_no_outgoing_transitions() -> None:
    """Successful, failed, and cancelled tasks are terminal."""

    terminal_statuses = {status for status in TaskStatus if is_terminal(status)}

    assert terminal_statuses == {
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    }


def test_ensure_task_transition_accepts_every_documented_transition() -> None:
    """The guard accepts the complete allowed transition set."""

    for current_status, target_status in _EXPECTED_TRANSITIONS:
        ensure_task_transition(current_status, target_status)


def test_ensure_task_transition_exposes_invalid_transition_context() -> None:
    """The domain error identifies both sides of a rejected transition."""

    with pytest.raises(
        InvalidTaskTransitionError,
        match=r"NEW -> COMPLETED",
    ) as error_info:
        ensure_task_transition(TaskStatus.NEW, TaskStatus.COMPLETED)

    assert error_info.value.current_status is TaskStatus.NEW
    assert error_info.value.target_status is TaskStatus.COMPLETED


def test_repeated_cancellation_is_not_a_second_domain_transition() -> None:
    """Cancellation idempotency is handled before requesting a transition."""

    assert not can_transition(TaskStatus.CANCELLED, TaskStatus.CANCELLED)
