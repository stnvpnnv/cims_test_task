"""Tests for the task-processing contract and default processor."""

from dataclasses import FrozenInstanceError
from uuid import UUID

import pytest

from cims_task_service.application.task_processor import (
    MAX_PROCESSING_ERROR_CODE_LENGTH,
    TaskProcessingError,
    TaskProcessingInput,
    TaskProcessor,
    TextStatisticsProcessor,
)
from cims_task_service.domain.task import TaskPriority

_TASK_ID = UUID("12345678-1234-4678-9234-567812345678")


def test_processing_input_is_a_frozen_slotted_detached_value() -> None:
    """A processor receives all execution data without mutable model state."""

    task = TaskProcessingInput(
        task_id=_TASK_ID,
        name="Quarterly report",
        description="Aggregate all regions",
        priority=TaskPriority.HIGH,
        attempt_count=2,
        max_attempts=5,
    )

    assert task.task_id == _TASK_ID
    assert task.name == "Quarterly report"
    assert task.description == "Aggregate all regions"
    assert task.priority is TaskPriority.HIGH
    assert task.attempt_count == 2
    assert task.max_attempts == 5
    assert not hasattr(task, "__dict__")

    with pytest.raises(FrozenInstanceError, match="cannot assign to field 'name'"):
        task.name = "Changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("name", "description", "expected_name_length", "expected_description_length"),
    [
        ("Задача🙂", "данныеé", 7, 7),
        (" \t ", "\n\r\n", 3, 3),
        ("", "", 0, 0),
    ],
)
@pytest.mark.asyncio
async def test_text_statistics_processor_returns_fresh_deterministic_results(
    name: str,
    description: str,
    expected_name_length: int,
    expected_description_length: int,
) -> None:
    """Counts use Python Unicode semantics and never reuse mutable result objects."""

    processor: TaskProcessor = TextStatisticsProcessor()
    task = TaskProcessingInput(
        task_id=_TASK_ID,
        name=name,
        description=description,
        priority=TaskPriority.LOW,
        attempt_count=1,
        max_attempts=3,
    )
    expected = {
        "name_length": expected_name_length,
        "description_length": expected_description_length,
    }

    first = await processor.process(task)
    second = await processor.process(task)

    assert first == expected
    assert second == expected
    assert first is not second

    first["name_length"] = -1
    assert second == expected


@pytest.mark.parametrize(
    ("code", "retryable"),
    [
        ("UPSTREAM_UNAVAILABLE", True),
        ("INVALID_TASK_INPUT", False),
        ("ERROR_2", True),
        ("A" * MAX_PROCESSING_ERROR_CODE_LENGTH, False),
    ],
)
def test_processing_error_exposes_only_safe_classification(
    code: str,
    retryable: bool,
) -> None:
    """Expected failures expose their bounded code and retry decision exactly."""

    error = TaskProcessingError(code, retryable=retryable)

    assert error.code == code
    assert error.retryable is retryable
    assert str(error) == code
    assert error.args == (code,)


@pytest.mark.parametrize(
    "code",
    [
        "",
        " ",
        "lowercase",
        "MIXED_Case",
        "HAS-DASH",
        "HAS SPACE",
        "CODE:DETAILS",
        "_LEADING",
        "TRAILING_",
        "DOUBLE__UNDERSCORE",
        "9_STARTS_WITH_DIGIT",
    ],
)
def test_processing_error_rejects_unsafe_codes(code: str) -> None:
    """Free-form details cannot cross the persistence and API boundary as a code."""

    with pytest.raises(ValueError, match=r"^code must be uppercase snake case$"):
        TaskProcessingError(code, retryable=False)


def test_processing_error_rejects_codes_above_the_public_limit() -> None:
    """Even syntactically safe codes stay within the documented storage bound."""

    code = "A" * (MAX_PROCESSING_ERROR_CODE_LENGTH + 1)

    with pytest.raises(
        ValueError,
        match=rf"^code must be at most {MAX_PROCESSING_ERROR_CODE_LENGTH} characters$",
    ):
        TaskProcessingError(code, retryable=True)


@pytest.mark.parametrize("code", [None, b"ERROR", 42])
def test_processing_error_rejects_non_string_codes(code: object) -> None:
    """Invalid caller types fail with a stable error instead of leaking implementation errors."""

    with pytest.raises(TypeError, match=r"^code must be a string$"):
        TaskProcessingError(code, retryable=True)  # type: ignore[arg-type]


@pytest.mark.parametrize("retryable", [None, 0, 1, "true"])
def test_processing_error_requires_a_real_boolean(retryable: object) -> None:
    """Truthiness must not silently change permanent and retryable classifications."""

    with pytest.raises(TypeError, match=r"^retryable must be a bool$"):
        TaskProcessingError(
            "PROCESSING_FAILED",
            retryable=retryable,  # type: ignore[arg-type]
        )
