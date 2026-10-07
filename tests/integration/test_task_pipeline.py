"""Task creation, publication, execution, and HTTP queries against real services."""

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from uuid import uuid4

import pytest
import pytest_asyncio
from aio_pika.abc import AbstractRobustConnection
from aio_pika.exceptions import ChannelNotFoundEntity
from httpx2 import ASGITransport, AsyncClient
from pydantic import SecretStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

import cims_task_service.dispatcher as dispatcher_module
import cims_task_service.main as main_module
import cims_task_service.worker as worker_module
from cims_task_service.api.schemas.task import TaskResponse, TaskStatusResponse
from cims_task_service.config import DispatcherSettings, Settings, WorkerSettings
from cims_task_service.domain.task import TaskPriority, TaskStatus
from cims_task_service.infrastructure.database.models import OutboxEventModel, TaskModel
from cims_task_service.infrastructure.database.session import AsyncSessionFactory
from cims_task_service.infrastructure.messaging import topology as topology_module
from cims_task_service.infrastructure.messaging.publisher import open_publisher_channel
from cims_task_service.infrastructure.messaging.topology import TaskTopology

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_OPERATION_TIMEOUT_SECONDS = 5.0
_PIPELINE_TIMEOUT_SECONDS = 15.0
_SHUTDOWN_TIMEOUT_SECONDS = 5.0
_POLL_INTERVAL_SECONDS = 0.025
_NAME = "Ежедневный отчёт 🚀"
_DESCRIPTION = "Посчитать символы\nв тексте 🌍"


@dataclass(frozen=True, slots=True)
class _Runtime:
    task: asyncio.Task[None]
    stop_event: asyncio.Event


@pytest_asyncio.fixture
async def pipeline_topology(
    monkeypatch: pytest.MonkeyPatch,
    rabbitmq_connection: AbstractRobustConnection,
) -> AsyncIterator[TaskTopology]:
    """Declare the production quorum/DLQ topology with names owned by this test."""

    prefix = f"cims.tests.pipeline.{uuid4().hex}"
    names = {
        "TASK_EXCHANGE_NAME": prefix,
        "TASK_QUEUE_NAME": f"{prefix}.execute.v1",
        "DEAD_LETTER_EXCHANGE_NAME": f"{prefix}.dead-letter",
        "DEAD_LETTER_QUEUE_NAME": f"{prefix}.dead-letter.v1",
    }
    for constant, name in names.items():
        monkeypatch.setattr(topology_module, constant, name)

    channel = await open_publisher_channel(rabbitmq_connection)
    try:
        async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
            topology = await topology_module.declare_task_topology(channel)
        yield topology
    finally:
        try:
            async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
                await channel.close()
        finally:
            # Delete by owned names even when declaration failed partway through.
            await _delete_owned_topology(rabbitmq_connection, names)


async def _delete_owned_topology(
    connection: AbstractRobustConnection,
    names: dict[str, str],
) -> None:
    failures: list[Exception] = []
    resources = (
        (True, names["TASK_QUEUE_NAME"]),
        (True, names["DEAD_LETTER_QUEUE_NAME"]),
        (False, names["TASK_EXCHANGE_NAME"]),
        (False, names["DEAD_LETTER_EXCHANGE_NAME"]),
    )
    for is_queue, name in resources:
        try:
            async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
                # A missing partial resource can close its channel with AMQP 404.
                channel = await open_publisher_channel(connection)
                try:
                    if is_queue:
                        await channel.queue_delete(
                            name,
                            if_unused=False,
                            if_empty=False,
                            timeout=_OPERATION_TIMEOUT_SECONDS,
                        )
                    else:
                        await channel.exchange_delete(
                            name,
                            if_unused=False,
                            timeout=_OPERATION_TIMEOUT_SECONDS,
                        )
                finally:
                    await channel.close()
        except ChannelNotFoundEntity:
            continue
        except Exception as error:
            failures.append(error)

    if failures:
        raise ExceptionGroup("pipeline topology cleanup failed", failures)


def _require_running(runtimes: list[_Runtime]) -> None:
    """Expose a runtime failure immediately instead of hiding it in a poll timeout."""

    for runtime in runtimes:
        if runtime.task.done():
            runtime.task.result()
            pytest.fail(f"{runtime.task.get_name()} exited before shutdown")


async def _stop_runtimes(runtimes: list[_Runtime]) -> None:
    """Request graceful shutdown, cancel overdue runtimes, and retrieve their results."""

    if not runtimes:
        return

    tasks = [runtime.task for runtime in runtimes]
    for runtime in runtimes:
        runtime.stop_event.set()

    completed, pending = await asyncio.wait(tasks, timeout=_SHUTDOWN_TIMEOUT_SECONDS)
    overdue = bool(pending)
    for task in pending:
        task.cancel()
    if pending:
        cancelled, pending = await asyncio.wait(pending, timeout=_SHUTDOWN_TIMEOUT_SECONDS)
        completed |= cancelled

    failures: list[Exception] = []
    for task in completed:
        try:
            task.result()
        except asyncio.CancelledError:
            if not overdue:
                failures.append(RuntimeError(f"{task.get_name()} cancelled itself"))
        except Exception as error:
            failures.append(error)

    if pending:
        names = ", ".join(task.get_name() for task in pending)
        failures.append(RuntimeError(f"pipeline runtimes did not cancel: {names}"))
    elif overdue:
        failures.append(RuntimeError("pipeline runtimes exceeded graceful shutdown timeout"))

    runtimes[:] = [runtime for runtime in runtimes if runtime.task in pending]
    if failures:
        raise ExceptionGroup("pipeline runtime shutdown failed", failures)


async def _read_rows(
    session_factory: AsyncSessionFactory,
) -> tuple[TaskModel, OutboxEventModel]:
    async with session_factory() as session:
        task = (await session.scalars(select(TaskModel))).one()
        event = (await session.scalars(select(OutboxEventModel))).one()
        return task, event


async def _wait_for_publication(
    client: AsyncClient,
    location: str,
    session_factory: AsyncSessionFactory,
    runtimes: list[_Runtime],
) -> None:
    async with asyncio.timeout(_PIPELINE_TIMEOUT_SECONDS):
        while True:
            _require_running(runtimes)
            response = await client.get(location)
            assert response.status_code == 200
            task = TaskResponse.model_validate(response.json())
            _stored_task, event = await _read_rows(session_factory)
            # PENDING alone precedes confirmation: it is set when outbox is claimed.
            if task.status is TaskStatus.PENDING and event.published_at is not None:
                return
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)


async def _wait_for_completion(
    client: AsyncClient,
    location: str,
    runtimes: list[_Runtime],
) -> TaskResponse:
    async with asyncio.timeout(_PIPELINE_TIMEOUT_SECONDS):
        while True:
            _require_running(runtimes)
            response = await client.get(location)
            assert response.status_code == 200
            task = TaskResponse.model_validate(response.json())
            if task.status is TaskStatus.COMPLETED:
                return task
            assert task.status in {TaskStatus.PENDING, TaskStatus.IN_PROGRESS}
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)


async def test_task_runs_from_http_creation_to_confirmed_publication_and_http_result(
    monkeypatch: pytest.MonkeyPatch,
    postgres_database_url: SecretStr,
    postgres_engine_factory: Callable[[Settings], AsyncEngine],
    postgres_session_factory: AsyncSessionFactory,
    rabbitmq_test_url: SecretStr,
    pipeline_topology: TaskTopology,
) -> None:
    """Exercise all production components using isolated schema and broker names."""

    for module in (main_module, dispatcher_module, worker_module):
        monkeypatch.setattr(module, "create_database_engine", postgres_engine_factory)

    settings = Settings(
        database_url=postgres_database_url,
        database_pool_size=3,
        database_max_overflow=0,
        database_pool_timeout_seconds=2.0,
        task_max_attempts=4,
        rabbitmq_url=rabbitmq_test_url,
        rabbitmq_connection_timeout_seconds=_OPERATION_TIMEOUT_SECONDS,
        rabbitmq_reconnect_interval_seconds=0.5,
    )
    dispatcher_settings = DispatcherSettings.model_validate(
        {
            **settings.model_dump(),
            "dispatcher_batch_size": 1,
            "dispatcher_poll_interval_seconds": _POLL_INTERVAL_SECONDS,
            "dispatcher_lease_duration_seconds": 10.0,
            "dispatcher_retry_initial_delay_seconds": 0.1,
            "dispatcher_retry_maximum_delay_seconds": 1.0,
            "rabbitmq_publish_timeout_seconds": 2.0,
            "execution_recovery_batch_size": 1,
            "execution_recovery_poll_interval_seconds": _POLL_INTERVAL_SECONDS,
            "execution_retry_initial_delay_seconds": 0.1,
            "execution_retry_maximum_delay_seconds": 1.0,
        }
    )
    worker_settings = WorkerSettings.model_validate(
        {
            **settings.model_dump(),
            "worker_concurrency": 2,
            "worker_lease_duration_seconds": 10.0,
            "worker_heartbeat_interval_seconds": 0.5,
            "worker_processing_timeout_seconds": 5.0,
            "execution_retry_initial_delay_seconds": 0.1,
            "execution_retry_maximum_delay_seconds": 1.0,
        }
    )
    application = main_module.create_app(settings)
    payload = {"name": _NAME, "description": _DESCRIPTION, "priority": TaskPriority.HIGH.value}
    headers = {"Idempotency-Key": str(uuid4())}
    runtimes: list[_Runtime] = []

    async with (
        application.router.lifespan_context(application),
        AsyncClient(transport=ASGITransport(app=application), base_url="http://test") as client,
    ):
        try:
            created = await client.post("/api/v1/tasks", json=payload, headers=headers)
            assert created.status_code == 201
            original = TaskResponse.model_validate(created.json())
            location = created.headers["Location"]
            assert location == f"/api/v1/tasks/{original.id}"
            assert original.status is TaskStatus.NEW
            assert original.result is original.error is None
            assert original.started_at is original.finished_at is None

            replay = await client.post("/api/v1/tasks", json=payload, headers=headers)
            assert replay.status_code == 200
            assert replay.headers["Location"] == location
            assert replay.json() == created.json()

            initial_task, initial_event = await _read_rows(postgres_session_factory)
            assert initial_task.id == original.id
            assert initial_task.attempt_count == 0
            assert initial_task.dispatch_token is not None
            assert initial_event.task_id == original.id
            assert initial_event.published_at is initial_event.discarded_at is None
            assert initial_event.publish_attempts == 0
            assert initial_event.payload == {
                "task_id": str(original.id),
                "dispatch_token": str(initial_task.dispatch_token),
            }

            dispatcher_stop = asyncio.Event()
            runtimes.append(
                _Runtime(
                    asyncio.create_task(
                        dispatcher_module.run_dispatcher(
                            dispatcher_settings, stop_event=dispatcher_stop
                        ),
                        name="pipeline-dispatcher",
                    ),
                    dispatcher_stop,
                )
            )
            await _wait_for_publication(client, location, postgres_session_factory, runtimes)

            worker_stop = asyncio.Event()
            runtimes.append(
                _Runtime(
                    asyncio.create_task(
                        worker_module.run_worker(worker_settings, stop_event=worker_stop),
                        name="pipeline-worker",
                    ),
                    worker_stop,
                )
            )
            completed = await _wait_for_completion(client, location, runtimes)
            assert completed.id == original.id
            assert completed.name == _NAME
            assert completed.description == _DESCRIPTION
            assert completed.priority is TaskPriority.HIGH
            assert completed.created_at == original.created_at
            assert completed.result == {
                "name_length": len(_NAME),
                "description_length": len(_DESCRIPTION),
            }
            assert completed.error is None
            assert completed.started_at is not None
            assert completed.finished_at is not None
            assert completed.created_at <= completed.started_at <= completed.finished_at

            status_response = await client.get(f"{location}/status")
            assert status_response.status_code == 200
            assert TaskStatusResponse.model_validate(status_response.json()) == TaskStatusResponse(
                id=original.id, status=TaskStatus.COMPLETED
            )

            await _stop_runtimes(runtimes)

            stored_task, stored_event = await _read_rows(postgres_session_factory)
            assert stored_task.id == original.id
            assert stored_task.status is TaskStatus.COMPLETED
            assert stored_task.attempt_count == 1
            assert stored_task.max_attempts == settings.task_max_attempts
            assert stored_task.result == completed.result
            assert stored_task.error is None
            assert stored_task.dispatch_token is stored_task.execution_token is None
            assert stored_task.lease_expires_at is None
            assert stored_task.started_at == completed.started_at
            assert stored_task.finished_at == completed.finished_at
            assert stored_event.id == initial_event.id
            assert stored_event.task_id == original.id
            assert stored_event.published_at is not None
            assert stored_event.discarded_at is None
            assert stored_event.publish_attempts == 1
            assert stored_event.publisher_token is None
            assert stored_event.lease_expires_at is None
            assert stored_event.last_error is None
            assert stored_event.payload == initial_event.payload
            assert stored_event.message_priority == 3

            # Worker channel is closed, so any unacknowledged delivery would reappear.
            async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
                assert await pipeline_topology.task_queue.get(fail=False) is None
                assert await pipeline_topology.dead_letter_queue.get(fail=False) is None
        finally:
            await _stop_runtimes(runtimes)
