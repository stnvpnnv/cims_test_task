"""Tests for transactional recovery of unclaimed task deliveries."""

import asyncio
from dataclasses import dataclass
from datetime import timedelta
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from cims_task_service.application import pending_delivery_recovery as recovery_module
from cims_task_service.application.pending_delivery_recovery import PendingTaskDeliveryRecovery
from cims_task_service.infrastructure.database.session import AsyncSessionFactory
from cims_task_service.infrastructure.messaging.topology import TASK_ROUTING_KEY


@dataclass(frozen=True, slots=True)
class _RecoveryHarness:
    recovery: PendingTaskDeliveryRecovery
    session: AsyncSession
    begin: Mock
    transaction: AsyncMock
    repository_factory: Mock
    recover_deliveries: AsyncMock


def _harness(monkeypatch: pytest.MonkeyPatch, *, recovered: int = 2) -> _RecoveryHarness:
    session = cast(AsyncSession, object())
    recover_deliveries = AsyncMock(return_value=recovered)
    repository_factory = Mock(
        return_value=SimpleNamespace(recover_pending_deliveries=recover_deliveries),
    )
    monkeypatch.setattr(recovery_module, "OutboxRepository", repository_factory)
    transaction = AsyncMock()
    transaction.__aenter__.return_value = session
    transaction.__aexit__.return_value = False
    begin = Mock(return_value=transaction)
    recovery = PendingTaskDeliveryRecovery(
        cast(AsyncSessionFactory, SimpleNamespace(begin=begin)),
        batch_size=7,
        delivery_timeout=timedelta(minutes=5),
    )
    return _RecoveryHarness(
        recovery,
        session,
        begin,
        transaction,
        repository_factory,
        recover_deliveries,
    )


@pytest.mark.parametrize("recovered", [0, 1, 7])
@pytest.mark.asyncio
async def test_pending_delivery_recovery_commits_one_finite_batch(
    monkeypatch: pytest.MonkeyPatch,
    recovered: int,
) -> None:
    """The service commits the repository's count using the execution route."""

    harness = _harness(monkeypatch, recovered=recovered)

    assert await harness.recovery.recover_once() == recovered

    harness.begin.assert_called_once_with()
    harness.transaction.__aenter__.assert_awaited_once_with()
    harness.repository_factory.assert_called_once_with(harness.session)
    harness.recover_deliveries.assert_awaited_once_with(
        event_type=TASK_ROUTING_KEY,
        batch_size=7,
        delivery_timeout=timedelta(minutes=5),
    )
    harness.transaction.__aexit__.assert_awaited_once_with(None, None, None)


@pytest.mark.parametrize(
    ("batch_size", "delivery_timeout", "message"),
    [
        (0, timedelta(seconds=1), "batch_size must be at least 1"),
        (-1, timedelta(seconds=1), "batch_size must be at least 1"),
        (101, timedelta(seconds=1), "batch_size must be at most 100"),
        (1, timedelta(0), "delivery_timeout must be positive"),
        (1, timedelta(microseconds=-1), "delivery_timeout must be positive"),
        (
            1,
            timedelta(days=7, microseconds=1),
            "delivery_timeout must be at most 7 days",
        ),
        (1, timedelta.max, "delivery_timeout must be at most 7 days"),
    ],
)
def test_pending_delivery_recovery_rejects_invalid_options_before_database_access(
    batch_size: int,
    delivery_timeout: timedelta,
    message: str,
) -> None:
    """Unsafe timers and unbounded batches cannot begin a transaction."""

    begin = Mock()

    with pytest.raises(ValueError, match=f"^{message}$"):
        PendingTaskDeliveryRecovery(
            cast(AsyncSessionFactory, SimpleNamespace(begin=begin)),
            batch_size=batch_size,
            delivery_timeout=delivery_timeout,
        )

    begin.assert_not_called()


@pytest.mark.parametrize("batch_size", [1, 100])
@pytest.mark.parametrize("delivery_timeout", [timedelta(microseconds=1), timedelta(days=7)])
def test_pending_delivery_recovery_accepts_batch_and_timeout_bounds(
    batch_size: int,
    delivery_timeout: timedelta,
) -> None:
    """Both batch bounds and the supported timeout extremes are accepted."""

    begin = Mock()
    PendingTaskDeliveryRecovery(
        cast(AsyncSessionFactory, SimpleNamespace(begin=begin)),
        batch_size=batch_size,
        delivery_timeout=delivery_timeout,
    )
    begin.assert_not_called()


@pytest.mark.parametrize("error_type", [RuntimeError, TimeoutError, asyncio.CancelledError])
@pytest.mark.asyncio
async def test_pending_delivery_failure_leaves_rollback_to_the_transaction(
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[BaseException],
) -> None:
    """Database failures and forced cancellation leave no unowned transaction."""

    harness = _harness(monkeypatch)
    expected_error = error_type("pending recovery failed")
    harness.recover_deliveries.side_effect = expected_error

    with pytest.raises(error_type) as error_info:
        await harness.recovery.recover_once()

    assert error_info.value is expected_error
    exit_arguments = harness.transaction.__aexit__.await_args.args
    assert exit_arguments[:2] == (error_type, expected_error)


@pytest.mark.asyncio
async def test_pending_delivery_transaction_start_failure_skips_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unavailable database cannot invoke the recovery repository."""

    harness = _harness(monkeypatch)
    expected_error = RuntimeError("transaction unavailable")
    harness.transaction.__aenter__.side_effect = expected_error

    with pytest.raises(RuntimeError) as error_info:
        await harness.recovery.recover_once()

    assert error_info.value is expected_error
    harness.repository_factory.assert_not_called()
    harness.recover_deliveries.assert_not_awaited()
    harness.transaction.__aexit__.assert_not_awaited()


@pytest.mark.asyncio
async def test_pending_delivery_commit_failure_is_not_reported_as_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A returned repository count does not mask failure to commit the batch."""

    harness = _harness(monkeypatch)
    expected_error = RuntimeError("commit failed")
    harness.transaction.__aexit__.side_effect = expected_error

    with pytest.raises(RuntimeError) as error_info:
        await harness.recovery.recover_once()

    assert error_info.value is expected_error
    harness.recover_deliveries.assert_awaited_once()
