"""Stable idempotency hashes for task creation."""

import json
from hashlib import sha256
from typing import Final

from cims_task_service.domain.task import TaskPriority

_KEY_HASH_DOMAIN: Final = b"POST /tasks:v1:idempotency-key\x00"
_REQUEST_FINGERPRINT_DOMAIN: Final = b"POST /tasks:v1:request-fingerprint\x00"


def hash_task_creation_idempotency_key(key: str) -> bytes:
    """Hash a validated logical key within the versioned task-creation scope."""

    return sha256(_KEY_HASH_DOMAIN + key.encode("utf-8")).digest()


def fingerprint_task_creation_request(
    *,
    name: str,
    description: str,
    priority: TaskPriority,
) -> bytes:
    """Fingerprint the validated client-controlled task creation fields."""

    canonical_request = json.dumps(
        {
            "description": description,
            "name": name,
            "priority": priority.value,
        },
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(_REQUEST_FINGERPRINT_DOMAIN + canonical_request).digest()
