"""Atomic recovery of task executions abandoned by workers."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Final
from uuid import UUID

from cims_task_service.infrastructure.database.models import JsonObject
from cims_task_service.infrastructure.database.session import AsyncSessionFactory
from cims_task_service.infrastructure.database.task_execution_repository import (
    LockedExpiredTaskExecution,
    TaskExecutionRepository,
)
from cims_task_service.infrastructure.messaging.topology import (
    TASK_ROUTING_KEY,
    task_message_priority,
)

MAX_RECOVERY_BATCH_SIZE: Final = 100

type RetryDelayForAttempt = Callable[[int], timedelta]


@dataclass(frozen=True, slots=True)
class RecoveryBatchResult:
    """Observable outcome of one finite execution-recovery pass."""

    locked: int
    retried: int
    failed: int


class ExecutionRecoveryInvariantError(RuntimeError):
    """Raised when a row locked for recovery unexpectedly loses ownership."""

    def __init__(self, task_id: UUID) -> None:
        self.task_id = task_id
        super().__init__(f"locked execution lost ownership for task {task_id}")


class TaskExecutionRecovery:
    """Recover one bounded batch inside a single PostgreSQL transaction."""

    def __init__(
        self,
        session_factory: AsyncSessionFactory,
        *,
        batch_size: int,
        retry_delay_for_attempt: RetryDelayForAttempt,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if batch_size > MAX_RECOVERY_BATCH_SIZE:
            raise ValueError(f"batch_size must be at most {MAX_RECOVERY_BATCH_SIZE}")

        self._session_factory = session_factory
        self._batch_size = batch_size
        self._retry_delay_for_attempt = retry_delay_for_attempt

    async def recover_once(self) -> RecoveryBatchResult:
        """Retry or fail expired executions before committing their row locks."""

        async with self._session_factory.begin() as session:
            repository = TaskExecutionRepository(session)
            locked_executions = await repository.lock_expired_execution_batch(
                batch_size=self._batch_size
            )
            retry_delays = self._retry_delays(locked_executions)

            retried = 0
            failed = 0
            for execution in locked_executions:
                if execution.attempt_count < execution.max_attempts:
                    changed = await repository.schedule_execution_retry(
                        execution.task_id,
                        execution_token=execution.execution_token,
                        retry_delay=retry_delays[execution.task_id],
                        event_type=TASK_ROUTING_KEY,
                        message_priority=task_message_priority(execution.priority),
                    )
                    retried += 1
                else:
                    changed = await repository.fail_execution(
                        execution.task_id,
                        execution_token=execution.execution_token,
                        error=_execution_lease_expired_error(),
                    )
                    failed += 1

                if not changed:
                    raise ExecutionRecoveryInvariantError(execution.task_id)

        return RecoveryBatchResult(
            locked=len(locked_executions),
            retried=retried,
            failed=failed,
        )

    def _retry_delays(
        self,
        locked_executions: tuple[LockedExpiredTaskExecution, ...],
    ) -> dict[UUID, timedelta]:
        retry_delays: dict[UUID, timedelta] = {}
        for execution in locked_executions:
            if execution.attempt_count >= execution.max_attempts:
                continue

            retry_delay = self._retry_delay_for_attempt(execution.attempt_count)
            if retry_delay <= timedelta(0):
                raise ValueError("retry delay must be positive")
            retry_delays[execution.task_id] = retry_delay

        return retry_delays


def _execution_lease_expired_error() -> JsonObject:
    return {"code": "EXECUTION_LEASE_EXPIRED", "retryable": False}
