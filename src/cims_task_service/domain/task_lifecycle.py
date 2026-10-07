"""Task status transition policy."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from cims_task_service.domain.task import TaskStatus

_ALLOWED_TRANSITIONS: Final[Mapping[TaskStatus, frozenset[TaskStatus]]] = MappingProxyType(
    {
        TaskStatus.NEW: frozenset(
            {
                TaskStatus.PENDING,
                TaskStatus.CANCELLED,
            }
        ),
        TaskStatus.PENDING: frozenset(
            {
                TaskStatus.IN_PROGRESS,
                TaskStatus.CANCELLED,
            }
        ),
        TaskStatus.IN_PROGRESS: frozenset(
            {
                TaskStatus.PENDING,
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
            }
        ),
        TaskStatus.COMPLETED: frozenset(),
        TaskStatus.FAILED: frozenset(),
        TaskStatus.CANCELLED: frozenset(),
    }
)


class InvalidTaskTransitionError(ValueError):
    """Raised when a task status transition violates the lifecycle."""

    def __init__(
        self,
        current_status: TaskStatus,
        target_status: TaskStatus,
    ) -> None:
        self.current_status = current_status
        self.target_status = target_status
        super().__init__(
            f"Invalid task status transition: {current_status.value} -> {target_status.value}"
        )


def can_transition(
    current_status: TaskStatus,
    target_status: TaskStatus,
) -> bool:
    """Return whether the lifecycle permits a status transition."""

    return target_status in _ALLOWED_TRANSITIONS[current_status]


def is_terminal(status: TaskStatus) -> bool:
    """Return whether a task status has no outgoing transitions."""

    return not _ALLOWED_TRANSITIONS[status]


def ensure_task_transition(
    current_status: TaskStatus,
    target_status: TaskStatus,
) -> None:
    """Raise when a requested task status transition is not permitted."""

    if not can_transition(current_status, target_status):
        raise InvalidTaskTransitionError(current_status, target_status)
