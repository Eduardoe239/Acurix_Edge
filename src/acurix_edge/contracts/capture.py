"""Captured frame contract (schema 1.0) — Capture Adapter output (AEX-CONTRACT-003).

0.1.0 has no real capture adapter; this contract fixes the boundary that 0.2.0
(AEX-CAPTURE-006) must satisfy. Rules:
- status is explicit: OK, SOURCE_UNAVAILABLE or UNKNOWN (never empty, never implied OK);
- a non-OK frame carries an AEX capture code and no pixels;
- image quality defaults to UNKNOWN (never assumed VALID);
- pixels are opaque and never serialized.
"""

from __future__ import annotations

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

CAPTURE_SCHEMA_VERSION = "1.0"
CAPTURE_ERROR_CODES = frozenset({ErrorCode.CAP_001, ErrorCode.CAP_002})


class CaptureStatus(StrEnum):
    OK = "OK"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    UNKNOWN = "UNKNOWN"


_KEYS = ("schema_version", "frame_id", "source_id", "sequence", "captured_at_utc", "capture_latency_ms",
         "input", "status", "error_code", "software_version")


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    frame_id: str
    source_id: str
    sequence: int
    captured_at_utc: str
    capture_latency_ms: float
    input: InputDescriptor
    status: CaptureStatus
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
        else:
            if self.error_code not in CAPTURE_ERROR_CODES:
                raise ContractError(f"non-OK capture requires an AEX capture code {sorted(CAPTURE_ERROR_CODES)}")
            if self.pixels is not None:
                raise ContractError("non-OK capture must not carry pixels")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "frame_id": self.frame_id,
            "source_id": self.source_id,
            "sequence": self.sequence,
            "captured_at_utc": self.captured_at_utc,
            "capture_latency_ms": self.capture_latency_ms,
            "input": self.input.to_dict(),
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
                captured_at_utc=data["captured_at_utc"],
                capture_latency_ms=data["capture_latency_ms"],
                input=InputDescriptor.from_dict(data["input"]),
                status=CaptureStatus(data["status"]),
                error_code=ErrorCode(data["error_code"]) if data["error_code"] is not None else None,
                software_version=data["software_version"],
            )
        except ContractError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"invalid capture payload: {exc}") from exc
