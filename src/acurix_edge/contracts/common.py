"""Shared contract validators: UTC timestamps, canonical UUIDs, SHA-256, measured latency.

Every versioned contract (capture, inference, decision, event) uses these so that the
same field means the same thing everywhere (AEX-CONTRACT-003).
"""

from __future__ import annotations

import math
import re
import uuid
from datetime import UTC, datetime
from typing import Any

from acurix_edge.contracts.errors import ContractError

_UTC_MS_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}Z")  # ASCII digits only
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def validate_utc(ts: Any, name: str = "captured_at_utc") -> None:
    if not isinstance(ts, str) or not _UTC_MS_RE.fullmatch(ts):
        raise ContractError(f"{name} must be UTC 'YYYY-MM-DDTHH:MM:SS.mmmZ'")
    try:
        datetime.fromisoformat(ts[:-1] + "+00:00")
    except ValueError as exc:
        raise ContractError(f"invalid {name} {ts!r}") from exc


def validate_canonical_uuid(value: Any, name: str) -> None:
    try:
        canonical = str(uuid.UUID(value))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ContractError(f"{name} must be a UUID") from exc
    if value != canonical:
        raise ContractError(f"{name} must be a canonical lowercase UUID")


def validate_sha256(value: Any, name: str) -> None:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ContractError(f"{name} must be 64 lowercase hex chars")


def validate_latency_ms(value: Any, name: str = "latency_ms") -> None:
    """A measured duration: finite, >= 0, numeric and not bool."""
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ContractError(f"{name} must be a finite number >= 0")


def validate_non_empty_str(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{name} must be a non-empty string")


def require_keys(data: Any, keys: tuple[str, ...], contract: str) -> None:
    """Strict deserialization: a missing field is a contract error, never a silent default."""
    if not isinstance(data, dict):
        raise ContractError(f"{contract} payload must be an object")
    missing = [k for k in keys if k not in data]
    if missing:
        raise ContractError(f"{contract} payload missing fields {missing}")
