"""Application-level errors shared by task use cases."""

from uuid import UUID

from cims_task_service.domain.task import TaskStatus


class TaskNotFoundError(LookupError):
    """Raised when a requested task does not exist."""

    def __init__(self, task_id: UUID) -> None:
        self.task_id = task_id
        super().__init__(f"Task {task_id} was not found")


class TaskNotCancellableError(RuntimeError):
    """Raised when a completed or failed task cannot be cancelled."""

    def __init__(self, task_id: UUID, current_status: TaskStatus) -> None:
        self.task_id = task_id
        self.current_status = current_status
        super().__init__(f"Task {task_id} cannot be cancelled from status {current_status.value}")
