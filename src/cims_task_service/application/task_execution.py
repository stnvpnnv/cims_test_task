"""Orchestrate one fenced task execution outside database transactions."""

import json
from collections.abc import Callable
from datetime import timedelta
from enum import StrEnum
from typing import Final
from uuid import UUID

from pydantic import ConfigDict, TypeAdapter

from cims_task_service.application.task_processor import (
    TaskProcessingError,
    TaskProcessingInput,
    TaskProcessor,
)
from cims_task_service.infrastructure.database.models import JsonObject, JsonValue
from cims_task_service.infrastructure.database.session import AsyncSessionFactory
from cims_task_service.infrastructure.database.task_execution_repository import (
    ClaimedTaskExecution,
    TaskExecutionRepository,
)
from cims_task_service.infrastructure.messaging.topology import (
    TASK_ROUTING_KEY,
    task_message_priority,
)

INVALID_PROCESSOR_RESULT_CODE: Final = "INVALID_PROCESSOR_RESULT"
_PROCESSOR_RESULT_ADAPTER: Final[TypeAdapter[JsonObject]] = TypeAdapter(
    JsonObject,
    config=ConfigDict(strict=True, allow_inf_nan=False),
)

type RetryDelayForAttempt = Callable[[int], timedelta]


class TaskExecutionOutcome(StrEnum):
    """Durable outcome used by the message consumer after one delivery."""

    NOT_CLAIMED = "not_claimed"
    COMPLETED = "completed"
    RETRY_SCHEDULED = "retry_scheduled"
    FAILED = "failed"
    LOST_OWNERSHIP = "lost_ownership"


class _InvalidProcessorResultError(ValueError):
    """Internal marker for values that cannot cross the JSON result boundary."""


class TaskExecutor:
    """Claim, process, and durably finalize one task delivery."""

    def __init__(
        self,
        session_factory: AsyncSessionFactory,
        processor: TaskProcessor,
        *,
        lease_duration: timedelta,
        retry_delay_for_attempt: RetryDelayForAttempt,
    ) -> None:
        if lease_duration <= timedelta(0):
            raise ValueError("lease_duration must be positive")

        self._session_factory = session_factory
        self._processor = processor
        self._lease_duration = lease_duration
        self._retry_delay_for_attempt = retry_delay_for_attempt

    async def execute(
        self,
        task_id: UUID,
        *,
        dispatch_token: UUID,
    ) -> TaskExecutionOutcome:
        """Process a current delivery and return only after its database commit."""

        claimed = await self._claim(task_id, dispatch_token=dispatch_token)
        if claimed is None:
            return TaskExecutionOutcome.NOT_CLAIMED

        processing_input = TaskProcessingInput(
            task_id=claimed.task_id,
            name=claimed.name,
            description=claimed.description,
            priority=claimed.priority,
            attempt_count=claimed.attempt_count,
            max_attempts=claimed.max_attempts,
        )
        try:
            raw_result = await self._processor.process(processing_input)
        except TaskProcessingError as error:
            return await self._finalize_processing_error(claimed, error)

        try:
            result = _validate_processor_result(raw_result)
        except _InvalidProcessorResultError:
            return await self._fail(
                claimed,
                error={
                    "code": INVALID_PROCESSOR_RESULT_CODE,
                    "retryable": False,
                },
            )

        return await self._complete(claimed, result=result)

    async def _claim(
        self,
        task_id: UUID,
        *,
        dispatch_token: UUID,
    ) -> ClaimedTaskExecution | None:
        async with self._session_factory.begin() as session:
            claimed = await TaskExecutionRepository(session).claim_for_execution(
                task_id,
                dispatch_token=dispatch_token,
                lease_duration=self._lease_duration,
            )

        return claimed

    async def _finalize_processing_error(
        self,
        claimed: ClaimedTaskExecution,
        error: TaskProcessingError,
    ) -> TaskExecutionOutcome:
        if error.retryable and claimed.attempt_count < claimed.max_attempts:
            retry_delay = self._retry_delay_for_attempt(claimed.attempt_count)
            if retry_delay <= timedelta(0):
                raise ValueError("retry delay must be positive")
            return await self._schedule_retry(claimed, retry_delay=retry_delay)

        return await self._fail(
            claimed,
            error={"code": error.code, "retryable": error.retryable},
        )

    async def _complete(
        self,
        claimed: ClaimedTaskExecution,
        *,
        result: JsonObject,
    ) -> TaskExecutionOutcome:
        async with self._session_factory.begin() as session:
            completed = await TaskExecutionRepository(session).complete_execution(
                claimed.task_id,
                execution_token=claimed.execution_token,
                result=result,
            )

        if completed:
            return TaskExecutionOutcome.COMPLETED
        return TaskExecutionOutcome.LOST_OWNERSHIP

    async def _schedule_retry(
        self,
        claimed: ClaimedTaskExecution,
        *,
        retry_delay: timedelta,
    ) -> TaskExecutionOutcome:
        async with self._session_factory.begin() as session:
            scheduled = await TaskExecutionRepository(session).schedule_execution_retry(
                claimed.task_id,
                execution_token=claimed.execution_token,
                retry_delay=retry_delay,
                event_type=TASK_ROUTING_KEY,
                message_priority=task_message_priority(claimed.priority),
            )

        if scheduled:
            return TaskExecutionOutcome.RETRY_SCHEDULED
        return TaskExecutionOutcome.LOST_OWNERSHIP

    async def _fail(
        self,
        claimed: ClaimedTaskExecution,
        *,
        error: JsonObject,
    ) -> TaskExecutionOutcome:
        async with self._session_factory.begin() as session:
            failed = await TaskExecutionRepository(session).fail_execution(
                claimed.task_id,
                execution_token=claimed.execution_token,
                error=error,
            )

        if failed:
            return TaskExecutionOutcome.FAILED
        return TaskExecutionOutcome.LOST_OWNERSHIP


def _validate_processor_result(result: object) -> JsonObject:
    try:
        validated = _PROCESSOR_RESULT_ADAPTER.validate_python(result)
        if _contains_unsupported_jsonb_text(validated):
            raise ValueError
        json.dumps(validated, allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise _InvalidProcessorResultError from None

    return validated


def _contains_unsupported_jsonb_text(value: JsonValue) -> bool:
    if isinstance(value, str):
        return "\x00" in value or any("\ud800" <= character <= "\udfff" for character in value)
    if isinstance(value, list):
        return any(_contains_unsupported_jsonb_text(item) for item in value)
    if isinstance(value, dict):
        return any(
            _contains_unsupported_jsonb_text(key) or _contains_unsupported_jsonb_text(item)
            for key, item in value.items()
        )
    return False
