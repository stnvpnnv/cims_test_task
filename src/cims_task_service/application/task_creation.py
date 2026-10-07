"""Transactional task creation application service."""

from dataclasses import dataclass
from uuid import UUID

from cims_task_service.application.idempotency import (
    fingerprint_task_creation_request,
    hash_task_creation_idempotency_key,
)
from cims_task_service.domain.task import TaskPriority
from cims_task_service.infrastructure.database.models import TaskModel
from cims_task_service.infrastructure.database.session import AsyncSessionFactory
from cims_task_service.infrastructure.database.task_repository import TaskRepository
from cims_task_service.infrastructure.messaging.topology import (
    TASK_ROUTING_KEY,
    task_message_priority,
)


@dataclass(frozen=True, slots=True)
class CreateTaskCommand:
    """Validated client-controlled fields for a task creation request."""

    name: str
    description: str
    priority: TaskPriority
    idempotency_key: str | None = None


@dataclass(frozen=True, slots=True)
class CreateTaskResult:
    """Created or replayed task together with the operation outcome."""

    task: TaskModel
    created: bool


class IdempotencyKeyConflictError(ValueError):
    """Raised when an idempotency key is reused for a different request."""

    def __init__(self, task_id: UUID) -> None:
        self.task_id = task_id
        super().__init__("Idempotency key was already used for a different task request")


async def create_task(
    command: CreateTaskCommand,
    *,
    session_factory: AsyncSessionFactory,
    max_attempts: int,
) -> CreateTaskResult:
    """Create a task and outbox event atomically, or replay a keyed request."""

    if max_attempts < 1:
        message = "max_attempts must be at least 1"
        raise ValueError(message)

    idempotency_key_hash: bytes | None = None
    request_fingerprint: bytes | None = None
    if command.idempotency_key is not None:
        idempotency_key_hash = hash_task_creation_idempotency_key(command.idempotency_key)
        request_fingerprint = fingerprint_task_creation_request(
            name=command.name,
            description=command.description,
            priority=command.priority,
        )

    async with session_factory.begin() as session:
        stored = await TaskRepository(session).create_with_outbox(
            name=command.name,
            description=command.description,
            priority=command.priority,
            max_attempts=max_attempts,
            idempotency_key_hash=idempotency_key_hash,
            request_fingerprint=request_fingerprint,
            event_type=TASK_ROUTING_KEY,
            message_priority=task_message_priority(command.priority),
        )
        if not stored.created and stored.task.request_fingerprint != request_fingerprint:
            raise IdempotencyKeyConflictError(stored.task.id)

        result = CreateTaskResult(task=stored.task, created=stored.created)

    return result
