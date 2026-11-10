"""Decision contract (Vol.02 §4.3). Versioned enumeration of inspection results."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from acurix_edge import __version__
from acurix_edge.contracts.common import (
    require_keys,
    utc_now_iso,
    validate_latency_ms,
    validate_non_empty_str,
    validate_utc,
)
from acurix_edge.contracts.errors import ContractError

# 1.1 (AEX-CONTRACT-003): + decided_at_utc, latency_ms, software_version.
DECISION_SCHEMA_VERSION = "1.1"


class InspectionResult(StrEnum):
    PASS = "PASS"
    DEFECT = "DEFECT"
    REVIEW = "REVIEW"
    NO_DECISION = "NO_DECISION"


class DecisionReason(StrEnum):
    RULE_APPLIED = "RULE_APPLIED"
    NO_APPROVED_RULE = "NO_APPROVED_RULE"
    INFERENCE_NOT_OK = "INFERENCE_NOT_OK"
    INPUT_QUALITY = "INPUT_QUALITY"
    SYSTEM_NOT_OBSERVING = "SYSTEM_NOT_OBSERVING"
    HEALTH_NOT_OK = "HEALTH_NOT_OK"
    RULE_ERROR = "RULE_ERROR"


@dataclass(frozen=True, slots=True)
class Decision:
    result: InspectionResult
    reason: DecisionReason
    rule_id: str | None = None
    rule_version: str | None = None
    class_id: str | None = None
    confidence: float | None = None
    detail: str | None = None
    decided_at_utc: str = field(default_factory=utc_now_iso)
    # Measured by the DecisionEngine (it overwrites whatever a rule supplies); None only before
    # leaving the engine. The event contract requires the measured total latency.
    latency_ms: float | None = None
    software_version: str = __version__
    schema_version: str = DECISION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DECISION_SCHEMA_VERSION:
            raise ContractError(f"unsupported decision schema_version {self.schema_version!r}")
        validate_utc(self.decided_at_utc, "decided_at_utc")
        if self.latency_ms is not None:
            validate_latency_ms(self.latency_ms)
        validate_non_empty_str(self.software_version, "software_version")
        if not isinstance(self.result, InspectionResult) or not isinstance(self.reason, DecisionReason):
            raise ContractError("result/reason must be versioned enums")
        for name in ("rule_id", "rule_version", "class_id", "detail"):
            v = getattr(self, name)
            if v is not None and not isinstance(v, str):
                raise ContractError(f"{name} must be str or None")
        # R4-05: same contract as the event envelope — technical value in [0, 1], finite, not bool.
        c = self.confidence
        if c is not None and (
            not isinstance(c, (int, float)) or isinstance(c, bool) or not math.isfinite(c) or not 0.0 <= c <= 1.0
        ):
            raise ContractError("confidence must be a finite number within [0, 1] or None")
        if self.result is InspectionResult.PASS or self.result is InspectionResult.DEFECT:
            # A conclusive result can only come from an identified, versioned rule.
            if self.reason is not DecisionReason.RULE_APPLIED or not self.rule_id or not self.rule_version:
                raise ContractError(f"{self.result} requires RULE_APPLIED with rule_id and rule_version")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "result": self.result.value,
            "reason": self.reason.value,
            "rule_id": self.rule_id,
            "rule_version": self.rule_version,
            "class_id": self.class_id,
            "confidence": self.confidence,
            "detail": self.detail,
            "decided_at_utc": self.decided_at_utc,
            "latency_ms": self.latency_ms,
            "software_version": self.software_version,
        }

    _KEYS = ("schema_version", "result", "reason", "rule_id", "rule_version", "class_id", "confidence", "detail",
             "decided_at_utc", "latency_ms", "software_version")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Decision":
        require_keys(data, cls._KEYS, "decision")
        try:
            return cls(
                schema_version=data["schema_version"],
                result=InspectionResult(data["result"]),
                reason=DecisionReason(data["reason"]),
                rule_id=data["rule_id"],
                rule_version=data["rule_version"],
                class_id=data["class_id"],
                confidence=data["confidence"],
                detail=data["detail"],
                decided_at_utc=data["decided_at_utc"],
                latency_ms=data["latency_ms"],
                software_version=data["software_version"],
            )
        except ContractError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"invalid decision payload: {exc}") from exc
