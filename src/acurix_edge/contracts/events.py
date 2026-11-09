"""Inspection event envelope (Vol.02 §6.2, schema 1.0).

Rules: event_id is an immutable canonical UUID; timestamps are UTC ISO-8601
with millisecond precision and 'Z'; result is the versioned InspectionResult
enum; confidence is a technical value in [0, 1], not a calibrated probability.

Defense in depth: a conclusive result (PASS/DEFECT) is rejected at this layer
too unless it carries a valid model reference, the decision rule id/version and
inference status OK — independent of the DecisionEngine guard.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from acurix_edge.contracts.decision import InspectionResult
from acurix_edge.contracts.errors import ContractError
from acurix_edge.contracts.inference import InferenceStatus, ModelRef

EVENT_SCHEMA_VERSION = "1.0"
_UTC_MS_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}Z")  # ASCII digits only


class ReviewStatus(StrEnum):
    NOT_REQUIRED = "NOT_REQUIRED"
    PENDING = "PENDING"
    REVIEWED = "REVIEWED"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def validate_utc(ts: Any) -> None:
    if not isinstance(ts, str) or not _UTC_MS_RE.fullmatch(ts):
        raise ContractError("captured_at_utc must be UTC 'YYYY-MM-DDTHH:MM:SS.mmmZ'")
    try:
        datetime.fromisoformat(ts[:-1] + "+00:00")
    except ValueError as exc:
        raise ContractError(f"invalid captured_at_utc {ts!r}") from exc


def _validate_canonical_uuid(value: Any, name: str) -> None:
    try:
        canonical = str(uuid.UUID(value))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ContractError(f"{name} must be a UUID") from exc
    if value != canonical:
        raise ContractError(f"{name} must be a canonical lowercase UUID")


def _is_str(v: Any) -> bool:
    return isinstance(v, str) and bool(v)


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    event_id: str
    site_id: str
    station_id: str
    session_id: str
    inspection_id: str
    captured_at_utc: str
    capture_sequence: int
    result: InspectionResult
    reason: str
    inference_status: str
    model: dict[str, str] | None
    configuration: dict[str, str]
    rule: dict[str, str] | None = None
    class_id: str | None = None
    confidence: float | None = None
    media_ref: str | None = None
    review_status: ReviewStatus = ReviewStatus.NOT_REQUIRED
    software_version: str = ""
    schema_version: str = EVENT_SCHEMA_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != EVENT_SCHEMA_VERSION:
            raise ContractError(f"unsupported event schema_version {self.schema_version!r}")
        _validate_canonical_uuid(self.event_id, "event_id")
        for name in ("site_id", "station_id", "session_id", "inspection_id", "reason", "software_version", "inference_status"):
            if not _is_str(getattr(self, name)):
                raise ContractError(f"{name} must be a non-empty string")
        try:
            InferenceStatus(self.inference_status)
        except ValueError as exc:
            raise ContractError(f"unknown inference_status {self.inference_status!r}") from exc
        validate_utc(self.captured_at_utc)
        if not isinstance(self.capture_sequence, int) or isinstance(self.capture_sequence, bool) or self.capture_sequence < 0:
            raise ContractError("capture_sequence must be int >= 0")
        if not isinstance(self.result, InspectionResult):
            raise ContractError("result must be InspectionResult")
        if not isinstance(self.review_status, ReviewStatus):
            raise ContractError("review_status must be ReviewStatus")
        if not isinstance(self.configuration, dict) or not all(
            _is_str(self.configuration.get(k)) for k in ("version", "sha256")
        ):
            raise ContractError("configuration must carry non-empty version and sha256 strings")
        if self.class_id is not None and not _is_str(self.class_id):
            raise ContractError("class_id must be a non-empty string or null")
        if self.confidence is not None and (
            not isinstance(self.confidence, (int, float)) or isinstance(self.confidence, bool)
            or not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0
        ):
            raise ContractError("confidence must be within [0, 1] or null")
        if self.model is not None:
            if not isinstance(self.model, dict):
                raise ContractError("model must be an object or null")
            if not {"id", "version", "sha256"} <= set(self.model):
                raise ContractError("model must carry id, version, sha256")
            ModelRef.from_dict(self.model)  # validates sha256 format
        if self.rule is not None and (not isinstance(self.rule, dict) or not (_is_str(self.rule.get("id")) and _is_str(self.rule.get("version")))):
            raise ContractError("rule must carry non-empty id and version or be null")
        if self.result in (InspectionResult.PASS, InspectionResult.DEFECT):
            if self.model is None or self.rule is None or self.inference_status != InferenceStatus.OK.value:
                raise ContractError("conclusive result requires model, decision rule and inference_status OK")
        if self.media_ref is not None:
            norm = self.media_ref.replace("\\", "/") if isinstance(self.media_ref, str) else ""
            if not norm or ".." in norm.split("/") or norm.startswith("/") or ":" in norm:
                raise ContractError("media_ref must be a managed relative reference")
        if not isinstance(self.extra, dict):
            raise ContractError("extra must be an object")
        # Detach mutable containers from caller references (hash stability).
        for name in ("model", "configuration", "rule", "extra"):
            object.__setattr__(self, name, copy.deepcopy(getattr(self, name)))

    @staticmethod
    def new_id() -> str:
        return str(uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy({
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "site_id": self.site_id,
            "station_id": self.station_id,
            "session_id": self.session_id,
            "inspection_id": self.inspection_id,
            "captured_at_utc": self.captured_at_utc,
            "capture_sequence": self.capture_sequence,
            "result": self.result.value,
            "reason": self.reason,
            "inference_status": self.inference_status,
            "rule": self.rule,
            "class_id": self.class_id,
            "confidence": self.confidence,
            "model": self.model,
            "configuration": self.configuration,
            "media_ref": self.media_ref,
            "review_status": self.review_status.value,
            "software_version": self.software_version,
            "extra": self.extra,
        })

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)

    def payload_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EventEnvelope":
        try:
            return cls(
                schema_version=data["schema_version"],
                event_id=data["event_id"],
                site_id=data["site_id"],
                station_id=data["station_id"],
                session_id=data["session_id"],
                inspection_id=data["inspection_id"],
                captured_at_utc=data["captured_at_utc"],
                capture_sequence=data["capture_sequence"],
                result=InspectionResult(data["result"]),
                reason=data["reason"],
                inference_status=data["inference_status"],
                rule=data.get("rule"),
                class_id=data.get("class_id"),
                confidence=data.get("confidence"),
                model=data.get("model"),
                configuration=data["configuration"],
                media_ref=data.get("media_ref"),
                review_status=ReviewStatus(data.get("review_status", ReviewStatus.NOT_REQUIRED.value)),
                software_version=data["software_version"],
                extra=data.get("extra") or {},
            )
        except ContractError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"invalid event payload: {exc}") from exc
