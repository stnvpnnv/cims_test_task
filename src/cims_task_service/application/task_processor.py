"""Application contract and default implementation for task processing."""

import re
from dataclasses import dataclass
from typing import Final, Protocol
from uuid import UUID

from cims_task_service.domain.task import TaskPriority
from cims_task_service.infrastructure.database.models import JsonObject

MAX_PROCESSING_ERROR_CODE_LENGTH: Final = 64
_PROCESSING_ERROR_CODE_PATTERN: Final = re.compile(
    r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*",
)


@dataclass(frozen=True, slots=True)
class TaskProcessingInput:
    """Detached task data supplied to a processor for one claimed attempt."""

    task_id: UUID
    name: str
    description: str
    priority: TaskPriority
    attempt_count: int
    max_attempts: int


class TaskProcessor(Protocol):
    """Port for replaceable asynchronous task-processing implementations."""

    async def process(self, task: TaskProcessingInput) -> JsonObject:
        """Process one claimed task attempt and return its JSON-compatible result."""


class TaskProcessingError(Exception):
    """Expected processor failure safe to persist and expose as structured data."""

    def __init__(self, code: str, *, retryable: bool) -> None:
        if not isinstance(code, str):
            raise TypeError("code must be a string")
        if len(code) > MAX_PROCESSING_ERROR_CODE_LENGTH:
            raise ValueError(
                f"code must be at most {MAX_PROCESSING_ERROR_CODE_LENGTH} characters",
            )
        if _PROCESSING_ERROR_CODE_PATTERN.fullmatch(code) is None:
            raise ValueError("code must be uppercase snake case")
        if type(retryable) is not bool:
            raise TypeError("retryable must be a bool")

        self.code = code
        self.retryable = retryable
        super().__init__(code)


class TextStatisticsProcessor:
    """Return deterministic character counts for the task text fields."""

    async def process(self, task: TaskProcessingInput) -> JsonObject:
        """Count Unicode code points in the task name and description."""

        return {
            "name_length": len(task.name),
            "description_length": len(task.description),
        }
