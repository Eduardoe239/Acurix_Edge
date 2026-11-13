"""Image quality gate result (QG-001, QG-002), schema 1.0.

Every evaluated frame gets a QualityResult: VALID or INPUT_INVALID, the list of reasons (metric,
measured value, limit, comparison), all measured metrics, the envelope version used and the time
spent. `envelope_validated` is False while the envelope is provisional (Anexo 04A §3.2 not done):
a provisional envelope can reject images, but it never certifies them as fit for a decision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from acurix_edge.contracts.errors import ContractError

QUALITY_SCHEMA_VERSION = "1.0"


class QualityStatus(StrEnum):
    VALID = "VALID"
    INPUT_INVALID = "INPUT_INVALID"


class Comparison(StrEnum):
    BELOW_MIN = "BELOW_MIN"   # value < limit is the failure
    ABOVE_MAX = "ABOVE_MAX"   # value > limit is the failure
    NOT_FOUND = "NOT_FOUND"   # the thing measured was not found (value/limit describe the search)


def _finite(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ContractError(f"{name} must be a finite number")


@dataclass(frozen=True, slots=True)
class QualityReason:
    code: str          # e.g. FOCUS_LOW, UNDEREXPOSED, OVEREXPOSED, REFLECTION, OBJECT_OUTSIDE_ROI
    metric: str        # metric name, e.g. sharpness_laplacian_var
    value: float
    limit: float
    comparison: Comparison

    def __post_init__(self) -> None:
        if not self.code or not self.metric:
            raise ContractError("quality reason requires code and metric")
        _finite(self.value, "reason value")
        _finite(self.limit, "reason limit")
        if not isinstance(self.comparison, Comparison):
            raise ContractError("comparison must be Comparison")

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "metric": self.metric, "value": self.value, "limit": self.limit,
                "comparison": self.comparison.value}


@dataclass(frozen=True, slots=True)
class QualityResult:
    status: QualityStatus
    reasons: tuple[QualityReason, ...]
    metrics: dict[str, float]
    elapsed_ms: float
    envelope_version: str
    envelope_status: str
    envelope_validated: bool
    roi_px: tuple[int, int, int, int]  # x, y, w, h
    envelope_sha256: str = ""          # exact limits used (review L-b); "" only for the gate-failure fallback

    def __post_init__(self) -> None:
        if not isinstance(self.status, QualityStatus):
            raise ContractError("status must be QualityStatus")
        if self.status is QualityStatus.VALID and self.reasons:
            raise ContractError("a VALID result carries no reasons")
        if self.status is QualityStatus.INPUT_INVALID and not self.reasons:
            raise ContractError("an INPUT_INVALID result requires at least one reason")
        for k, v in self.metrics.items():
            _finite(v, f"metric {k}")
        _finite(self.elapsed_ms, "elapsed_ms")
        if self.elapsed_ms < 0:
            raise ContractError("elapsed_ms must be >= 0")
        if self.envelope_validated and self.envelope_status != "validated":
            raise ContractError("only a 'validated' envelope can be reported as validated")

    @property
    def is_valid(self) -> bool:
        return self.status is QualityStatus.VALID

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": QUALITY_SCHEMA_VERSION, "status": self.status.value,
                "reasons": [r.to_dict() for r in self.reasons], "metrics": dict(self.metrics),
                "elapsed_ms": self.elapsed_ms, "envelope_version": self.envelope_version,
                "envelope_sha256": self.envelope_sha256,
                "envelope_status": self.envelope_status, "envelope_validated": self.envelope_validated,
                "roi_px": list(self.roi_px)}
