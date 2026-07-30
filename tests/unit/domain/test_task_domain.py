"""Contract tests for task domain values."""

from cims_task_service.domain.task import TaskPriority, TaskStatus


def test_task_priority_values_match_the_contract() -> None:
    """Priorities retain their documented names and wire values."""

    assert {item.name: item.value for item in TaskPriority} == {
        "LOW": "LOW",
        "MEDIUM": "MEDIUM",
        "HIGH": "HIGH",
    }


def test_task_status_values_match_the_contract() -> None:
    """Statuses retain their documented names and wire values."""

    assert {item.name: item.value for item in TaskStatus} == {
        "NEW": "NEW",
        "PENDING": "PENDING",
        "IN_PROGRESS": "IN_PROGRESS",
        "COMPLETED": "COMPLETED",
        "FAILED": "FAILED",
        "CANCELLED": "CANCELLED",
    }
