"""Tests for stable task-creation idempotency hashes."""

import pytest

from cims_task_service.application.idempotency import (
    fingerprint_task_creation_request,
    hash_task_creation_idempotency_key,
)
from cims_task_service.domain.task import TaskPriority


def test_idempotency_key_hash_has_a_stable_scoped_representation() -> None:
    """The persisted digest format cannot change accidentally."""

    digest = hash_task_creation_idempotency_key("8e03978e-40d5-43e8-bc93-6894a57f9324")

    assert digest == bytes.fromhex(
        "cf9df451772168e92618032806dd96bfeb8d503770d952401762317f88cc17d7"
    )


def test_request_fingerprint_has_a_stable_canonical_representation() -> None:
    """A golden vector pins field names, serialization, scope, and algorithm."""

    digest = fingerprint_task_creation_request(
        name="  \u0421\u0432\u043e\u0434\u043a\u0430 \U0001f4ca  ",
        description=(
            "\u0421\u0442\u0440\u043e\u043a\u0430 1\n\u0421\u0442\u0440\u043e\u043a\u0430 2"
        ),
        priority=TaskPriority.HIGH,
    )

    assert digest == bytes.fromhex(
        "102bfee7ef49c41d63258cea406f2ef7346b4d2680fea703db81bb7879e7bb58"
    )


def test_hashes_are_deterministic_sha256_digests() -> None:
    """Identical logical inputs always produce fixed-size binary values."""

    key_hash = hash_task_creation_idempotency_key("same-key")
    repeated_key_hash = hash_task_creation_idempotency_key("same-key")
    request_fingerprint = fingerprint_task_creation_request(
        name="Task",
        description="Description",
        priority=TaskPriority.MEDIUM,
    )
    repeated_request_fingerprint = fingerprint_task_creation_request(
        name="Task",
        description="Description",
        priority=TaskPriority.MEDIUM,
    )

    assert key_hash == repeated_key_hash
    assert request_fingerprint == repeated_request_fingerprint
    assert len(key_hash) == 32
    assert len(request_fingerprint) == 32


@pytest.mark.parametrize(
    ("key", "different_key"),
    [
        ("Same-Key", "same-key"),
        ("same-key ", "same-key"),
        ("caf\u00e9", "cafe\u0301"),
    ],
)
def test_key_hash_preserves_the_logical_key_value(key: str, different_key: str) -> None:
    """Case, whitespace, and Unicode remain significant in the key."""

    assert hash_task_creation_idempotency_key(key) != hash_task_creation_idempotency_key(
        different_key
    )


@pytest.mark.parametrize(
    ("name", "description", "priority"),
    [
        ("Task ", "Description", TaskPriority.MEDIUM),
        ("Task", "Description ", TaskPriority.MEDIUM),
        ("Task", "Description", TaskPriority.LOW),
    ],
)
def test_fingerprint_changes_with_each_client_controlled_field(
    name: str,
    description: str,
    priority: TaskPriority,
) -> None:
    """Every behavior-affecting request field participates in the digest."""

    baseline = fingerprint_task_creation_request(
        name="Task",
        description="Description",
        priority=TaskPriority.MEDIUM,
    )

    assert (
        fingerprint_task_creation_request(
            name=name,
            description=description,
            priority=priority,
        )
        != baseline
    )


def test_fingerprint_does_not_normalize_unicode() -> None:
    """Distinct validated code-point sequences remain distinct requests."""

    composed = fingerprint_task_creation_request(
        name="Caf\u00e9",
        description="Description",
        priority=TaskPriority.MEDIUM,
    )
    decomposed = fingerprint_task_creation_request(
        name="Cafe\u0301",
        description="Description",
        priority=TaskPriority.MEDIUM,
    )

    assert composed != decomposed
