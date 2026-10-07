"""Republish pending tasks whose confirmed broker delivery may have been lost."""

from datetime import timedelta

from cims_task_service.application.task_execution_recovery import MAX_RECOVERY_BATCH_SIZE
from cims_task_service.config import MAX_PENDING_DELIVERY_TIMEOUT_SECONDS
from cims_task_service.infrastructure.database.outbox_repository import OutboxRepository
from cims_task_service.infrastructure.database.session import AsyncSessionFactory
from cims_task_service.infrastructure.messaging.topology import TASK_ROUTING_KEY


class PendingTaskDeliveryRecovery:
    """Reopen a bounded outbox batch without spending execution attempts."""

    def __init__(
        self,
        session_factory: AsyncSessionFactory,
        *,
        batch_size: int,
        delivery_timeout: timedelta,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if batch_size > MAX_RECOVERY_BATCH_SIZE:
            raise ValueError(f"batch_size must be at most {MAX_RECOVERY_BATCH_SIZE}")
        if delivery_timeout <= timedelta(0):
            raise ValueError("delivery_timeout must be positive")
        if delivery_timeout > timedelta(seconds=MAX_PENDING_DELIVERY_TIMEOUT_SECONDS):
            raise ValueError("delivery_timeout must be at most 7 days")

        self._session_factory = session_factory
        self._batch_size = batch_size
        self._delivery_timeout = delivery_timeout

    async def recover_once(self) -> int:
        """Commit replay eligibility while retaining the original token and event ID."""

        async with self._session_factory.begin() as session:
            return await OutboxRepository(session).recover_pending_deliveries(
                event_type=TASK_ROUTING_KEY,
                batch_size=self._batch_size,
                delivery_timeout=self._delivery_timeout,
            )
