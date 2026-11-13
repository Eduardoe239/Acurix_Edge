"""Inspection event envelope (Vol.02 §6.2, schema 1.1; 1.0 is read-only).

Rules: event_id is an immutable canonical UUID; timestamps are UTC ISO-8601
with millisecond precision and 'Z'; result is the versioned InspectionResult
enum; confidence is a technical value in [0, 1], not a calibrated probability.

Schema 1.1 (AEX-CONTRACT-003) adds, both mandatory:
- `latency_ms`: measured processing latency (inference + decision);
- `request_sha256`: digest of the inspection request, used to detect the same
  inspection_id re-submitted with different content (idempotency conflict).
Events persisted as 1.0 keep decoding unchanged (their stored bytes and hash are
never rewritten); new events are always written as 1.1.

Defense in depth: a conclusive result (PASS/DEFECT) is rejected at this layer
too unless it carries a valid model reference, the decision rule id/version and
inference status OK — independent of the DecisionEngine guard.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from acurix_edge.contracts.common import (
    utc_now_iso,
    validate_canonical_uuid,
    validate_latency_ms,
    validate_sha256,
    validate_utc,
)
from acurix_edge.contracts.decision import InspectionResult
from acurix_edge.contracts.errors import ContractError
from acurix_edge.contracts.inference import InferenceStatus, ModelRef

__all__ = ["EVENT_SCHEMA_VERSION", "SUPPORTED_EVENT_SCHEMAS", "EventEnvelope", "ReviewStatus", "utc_now_iso",
           "validate_utc"]

EVENT_SCHEMA_VERSION = "1.1"
LEGACY_EVENT_SCHEMA = "1.0"
SUPPORTED_EVENT_SCHEMAS = frozenset({LEGACY_EVENT_SCHEMA, EVENT_SCHEMA_VERSION})


class ReviewStatus(StrEnum):
    NOT_REQUIRED = "NOT_REQUIRED"
    PENDING = "PENDING"
    REVIEWED = "REVIEWED"


_validate_canonical_uuid = validate_canonical_uuid


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
    latency_ms: float | None = None
    request_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version not in SUPPORTED_EVENT_SCHEMAS:
            raise ContractError(f"unsupported event schema_version {self.schema_version!r}")
        if self.schema_version == EVENT_SCHEMA_VERSION:
            if self.latency_ms is None or self.request_sha256 is None:
                raise ContractError("event schema 1.1 requires latency_ms and request_sha256")
            validate_latency_ms(self.latency_ms)
            validate_sha256(self.request_sha256, "request_sha256")
        elif self.latency_ms is not None or self.request_sha256 is not None:
            raise ContractError("event schema 1.0 does not carry latency_ms/request_sha256")
        _validate_canonical_uuid(self.event_id, "event_id")
        for name in ("site_id", "station_id", "session_id", "inspection_id", "reason", "software_version",
                     "inference_status"):
            if not _is_str(getattr(self, name)):
                raise ContractError(f"{name} must be a non-empty string")
        try:
            InferenceStatus(self.inference_status)
        except ValueError as exc:
            raise ContractError(f"unknown inference_status {self.inference_status!r}") from exc
        validate_utc(self.captured_at_utc)
        if (not isinstance(self.capture_sequence, int) or isinstance(self.capture_sequence, bool)
                or self.capture_sequence < 0):
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
        if self.rule is not None and (not isinstance(self.rule, dict) or not (
                _is_str(self.rule.get("id")) and _is_str(self.rule.get("version")))):
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
        d = self._base_dict()
        if self.schema_version == EVENT_SCHEMA_VERSION:
            d["latency_ms"] = self.latency_ms
            d["request_sha256"] = self.request_sha256
        return d

    def _base_dict(self) -> dict[str, Any]:
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
    def from_dict(cls, data: dict[str, Any]) -> EventEnvelope:
        if not isinstance(data, dict):
            raise ContractError("event payload must be an object")
        if data.get("schema_version") == EVENT_SCHEMA_VERSION:
            for key in ("latency_ms", "request_sha256"):
                if key not in data:
                    raise ContractError(f"event payload missing fields ['{key}']")
        try:
            return cls(
                latency_ms=data.get("latency_ms"),
                request_sha256=data.get("request_sha256"),
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
