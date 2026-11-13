"""Captured frame contract (schema 1.1) — Capture Adapter output (AEX-CONTRACT-003, AEX-CAPTURE-006).

Rules:
- status is explicit: OK, SOURCE_UNAVAILABLE or UNKNOWN (never empty, never implied OK);
- a non-OK frame carries an AEX capture code, no pixels and no quality metrics;
- image quality defaults to UNKNOWN (never assumed VALID): `quality_metrics` (mean brightness,
  simple sharpness) are measurements only; the quality gate is release 0.3.0;
- `monotonic_ns` (CAP-002) orders frames inside one process; `captured_at_utc` is wall time;
- pixels are opaque and never serialized.

Schema history: 1.0 (Ordem 2) had no `monotonic_ns`/`quality_metrics`. 1.0 frames were never
persisted or transmitted (no capture adapter existed), so only 1.1 is accepted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from acurix_edge import __version__
from acurix_edge.contracts.common import (
    require_keys,
    validate_canonical_uuid,
    validate_latency_ms,
    validate_non_empty_str,
    validate_utc,
)
from acurix_edge.contracts.errors import ContractError, ErrorCode
from acurix_edge.contracts.inference import InputDescriptor

CAPTURE_SCHEMA_VERSION = "1.1"
CAPTURE_ERROR_CODES = frozenset({ErrorCode.CAP_001, ErrorCode.CAP_002})
MAX_BRIGHTNESS = 255.0


class CaptureStatus(StrEnum):
    OK = "OK"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    UNKNOWN = "UNKNOWN"


def _metric(value: Any, name: str, upper: float | None = None) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0 \
            or (upper is not None and value > upper):
        bound = f"0..{upper:g}" if upper is not None else ">= 0"
        raise ContractError(f"quality_metrics.{name} must be a finite number {bound}")


@dataclass(frozen=True, slots=True)
class FrameQualityMetrics:
    """Basic image measurements (not a quality verdict).

    mean_brightness: mean of the 8-bit grayscale image (0..255).
    sharpness: variance of the Laplacian of the grayscale image (>= 0; higher = sharper).
    """

    mean_brightness: float
    sharpness: float

    def __post_init__(self) -> None:
        _metric(self.mean_brightness, "mean_brightness", MAX_BRIGHTNESS)
        _metric(self.sharpness, "sharpness")

    def to_dict(self) -> dict[str, float]:
        return {"mean_brightness": self.mean_brightness, "sharpness": self.sharpness}

    @classmethod
    def from_dict(cls, data: Any) -> "FrameQualityMetrics":
        require_keys(data, ("mean_brightness", "sharpness"), "quality_metrics")
        if set(data) != {"mean_brightness", "sharpness"}:
            raise ContractError("quality_metrics has unknown fields")
        return cls(mean_brightness=data["mean_brightness"], sharpness=data["sharpness"])


_KEYS = ("schema_version", "frame_id", "source_id", "sequence", "monotonic_ns", "captured_at_utc",
         "capture_latency_ms", "input", "quality_metrics", "status", "error_code", "software_version")


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    frame_id: str
    source_id: str
    sequence: int
    monotonic_ns: int
    captured_at_utc: str
    capture_latency_ms: float
    input: InputDescriptor
    status: CaptureStatus
    quality_metrics: FrameQualityMetrics | None = None
    error_code: ErrorCode | None = None
    software_version: str = __version__
    schema_version: str = CAPTURE_SCHEMA_VERSION
    pixels: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != CAPTURE_SCHEMA_VERSION:
            raise ContractError(f"unsupported capture schema_version {self.schema_version!r}")
        validate_canonical_uuid(self.frame_id, "frame_id")
        validate_non_empty_str(self.source_id, "source_id")
        validate_non_empty_str(self.software_version, "software_version")
        if not isinstance(self.sequence, int) or isinstance(self.sequence, bool) or self.sequence < 0:
            raise ContractError("sequence must be int >= 0")
        if not isinstance(self.monotonic_ns, int) or isinstance(self.monotonic_ns, bool) or self.monotonic_ns < 0:
            raise ContractError("monotonic_ns must be int >= 0")
        validate_utc(self.captured_at_utc, "captured_at_utc")
        validate_latency_ms(self.capture_latency_ms, "capture_latency_ms")
        if not isinstance(self.input, InputDescriptor):
            raise ContractError("input must be InputDescriptor")
        self.input.validate()
        if not isinstance(self.status, CaptureStatus):
            raise ContractError("status must be CaptureStatus")
        if self.status is CaptureStatus.OK:
            if self.error_code is not None:
                raise ContractError("status OK must not carry error_code")
            if not isinstance(self.quality_metrics, FrameQualityMetrics):
                raise ContractError("status OK requires quality_metrics")
        else:
            if self.error_code not in CAPTURE_ERROR_CODES:
                raise ContractError(f"non-OK capture requires an AEX capture code {sorted(CAPTURE_ERROR_CODES)}")
            if self.pixels is not None:
                raise ContractError("non-OK capture must not carry pixels")
            if self.quality_metrics is not None:
                raise ContractError("non-OK capture must not carry quality_metrics")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "frame_id": self.frame_id,
            "source_id": self.source_id,
            "sequence": self.sequence,
            "monotonic_ns": self.monotonic_ns,
            "captured_at_utc": self.captured_at_utc,
            "capture_latency_ms": self.capture_latency_ms,
            "input": self.input.to_dict(),
            "quality_metrics": self.quality_metrics.to_dict() if self.quality_metrics else None,
            "status": self.status.value,
            "error_code": self.error_code.value if self.error_code else None,
            "software_version": self.software_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CapturedFrame":
        require_keys(data, _KEYS, "capture")
        try:
            return cls(
                schema_version=data["schema_version"],
                frame_id=data["frame_id"],
                source_id=data["source_id"],
                sequence=data["sequence"],
                monotonic_ns=data["monotonic_ns"],
                captured_at_utc=data["captured_at_utc"],
                capture_latency_ms=data["capture_latency_ms"],
                input=InputDescriptor.from_dict(data["input"]),
                quality_metrics=(FrameQualityMetrics.from_dict(data["quality_metrics"])
                                 if data["quality_metrics"] is not None else None),
                status=CaptureStatus(data["status"]),
                error_code=ErrorCode(data["error_code"]) if data["error_code"] is not None else None,
                software_version=data["software_version"],
            )
        except ContractError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"invalid capture payload: {exc}") from exc
