"""Normalized Inference Adapter contract (Vol.02 §17, schema 1.0).

The adapter invokes a runtime/model and normalizes its output. It NEVER decides
PASS/DEFECT: that belongs to the DecisionEngine. Any output without a valid
model, with invalid input or an incompatible runtime must carry a non-OK
status so that the decision layer returns NO_DECISION.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, cast, runtime_checkable

from acurix_edge import __version__
from acurix_edge.contracts.common import require_keys, utc_now_iso, validate_non_empty_str, validate_utc
from acurix_edge.contracts.errors import ContractError

# 1.1 (AEX-CONTRACT-003): + completed_at_utc, software_version. Not persisted as a whole, so no 1.0 reader.
# 1.2 (AEX-CV-007): + status INCONCLUSIVE; + Prediction.attributes (measurements such as an angle).
INFERENCE_SCHEMA_VERSION = "1.2"
AttributeValue = float | int | bool | str
_ATTR_KEY_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")
MAX_ATTRIBUTES = 32
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
# Caller-supplied identifier: ASCII only, bounded, no whitespace/control characters (R4-02).
_INSPECTION_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
MAX_IMAGE_SIDE = 65535


class InferenceStatus(StrEnum):
    OK = "OK"
    NO_MODEL = "NO_MODEL"
    INPUT_INVALID = "INPUT_INVALID"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    INCOMPATIBLE = "INCOMPATIBLE"
    TIMEOUT = "TIMEOUT"
    # The stage ran correctly but could not measure what it is for (e.g. label not found). Never conclusive.
    INCONCLUSIVE = "INCONCLUSIVE"


class ImageQuality(StrEnum):
    VALID = "VALID"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ContractError(msg)


def _finite_non_negative(value: float, name: str) -> None:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{name} must be numeric")
    _require(math.isfinite(value) and value >= 0, f"{name} must be finite and >= 0")


@dataclass(frozen=True, slots=True)
class ModelRef:
    id: str
    version: str
    sha256: str

    def __post_init__(self) -> None:
        _require(
            isinstance(self.id, str) and bool(self.id) and isinstance(self.version, str) and bool(self.version),
            "model id/version must be non-empty strings",
        )
        _require(bool(isinstance(self.sha256, str) and _SHA256_RE.fullmatch(self.sha256)),
                 "model sha256 must be 64 lowercase hex chars")

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "version": self.version, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelRef:
        if not isinstance(data, dict):
            raise ContractError("model must be an object")
        # No str() coercion: wrong types (or None) are rejected by __post_init__; the casts are static only.
        return cls(id=cast(str, data.get("id")), version=cast(str, data.get("version")),
                   sha256=cast(str, data.get("sha256")))


@dataclass(frozen=True, slots=True)
class RuntimeRef:
    name: str
    version: str
    device: str = "cpu"

    def __post_init__(self) -> None:
        _require(bool(self.name) and bool(self.version), "runtime name/version required")
        _require(bool(self.device), "runtime device required")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "device": self.device}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuntimeRef:
        return cls(name=str(data["name"]), version=str(data["version"]), device=str(data["device"]))


@dataclass(frozen=True, slots=True)
class InputDescriptor:
    width: int
    height: int
    quality: ImageQuality = ImageQuality.UNKNOWN

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name in ("width", "height"):
            v = getattr(self, name)
            _require(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= MAX_IMAGE_SIDE,
                     f"input {name} must be int in 0..{MAX_IMAGE_SIDE}")
        _require(isinstance(self.quality, ImageQuality), "input quality must be ImageQuality")
        if self.quality is ImageQuality.VALID:
            _require(self.width > 0 and self.height > 0, "a VALID input must have non-zero width and height")

    def to_dict(self) -> dict[str, Any]:
        return {"width": self.width, "height": self.height, "quality": self.quality.value}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InputDescriptor:
        try:
            quality = ImageQuality(data["quality"])
        except ValueError as exc:
            raise ContractError(f"unknown image quality {data.get('quality')!r}") from exc
        return cls(width=data["width"], height=data["height"], quality=quality)


def _check_attributes(attrs: Any) -> None:
    _require(isinstance(attrs, dict), "prediction attributes must be an object")
    _require(len(attrs) <= MAX_ATTRIBUTES, f"at most {MAX_ATTRIBUTES} prediction attributes")
    for key, value in attrs.items():
        _require(isinstance(key, str) and bool(_ATTR_KEY_RE.fullmatch(key)), f"invalid attribute name {key!r}")
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            _require(math.isfinite(value), f"attribute {key} must be finite")
        else:
            _require(isinstance(value, str) and len(value) <= 200, f"attribute {key} must be number, bool or str")


@dataclass(frozen=True, slots=True)
class Prediction:
    """Technical model output. `confidence` is a TECHNICAL confidence (DEC-004), NOT a probability.
    `attributes`: named measurements of the prediction (e.g. `angle_deg`, `borderline`)."""

    class_id: str
    confidence: float
    bbox: tuple[float, float, float, float] | None = None
    attributes: dict[str, AttributeValue] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        _check_attributes(self.attributes)
        _require(bool(self.class_id), "prediction class_id required")
        _require(
            isinstance(self.confidence, (int, float))
            and not isinstance(self.confidence, bool)
            and math.isfinite(self.confidence)
            and 0.0 <= self.confidence <= 1.0,
            "prediction confidence must be within [0, 1]",
        )
        if self.bbox is not None:
            _require(len(self.bbox) == 4, "bbox must have 4 values (x, y, w, h)")
            for v in self.bbox:
                _finite_non_negative(v, "bbox value")

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "confidence": self.confidence,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Prediction:
        bbox = data.get("bbox")
        return cls(
            class_id=str(data["class_id"]),
            confidence=data["confidence"],
            bbox=tuple(bbox) if bbox is not None else None,
            attributes=dict(data.get("attributes") or {}),
        )


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    inspection_id: str
    input: InputDescriptor
    frame: Any = field(default=None, repr=False, compare=False)  # opaque pixel payload, never serialized

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Full re-validation (R4-02). Called again by the application before any processing,
        so a request mutated after construction (object.__setattr__) cannot slip through."""
        _require(isinstance(self.inspection_id, str) and bool(_INSPECTION_ID_RE.fullmatch(self.inspection_id)),
                 "inspection_id must match " + _INSPECTION_ID_RE.pattern)
        _require(isinstance(self.input, InputDescriptor), "input must be InputDescriptor")
        self.input.validate()

    def sha256(self) -> str:
        """Request fingerprint for idempotency (EVT-001): inspection_id + input descriptor + pixel digest.

        Any object exposing the buffer protocol (bytes, numpy arrays, ...) is hashed with its
        format and shape. A frame that cannot be fingerprinted gets a RANDOM digest: a retry of
        it never matches, so it becomes an idempotency conflict (NO_DECISION + AEX-API-004) and
        a stored result is never replayed for content that could not be compared (review R1-M1)."""
        frame = self.frame
        if frame is None:
            frame_digest = None
        else:
            try:
                with memoryview(frame) as mv:
                    h = hashlib.sha256(f"{mv.format}|{mv.shape}|".encode("ascii"))
                    # Contiguous frames are hashed in place (no copy on the 8 GB host).
                    h.update(mv.cast("B") if mv.c_contiguous else mv.tobytes())
                frame_digest = h.hexdigest()
            except Exception:  # any failure -> never comparable -> conflict, never a replay
                frame_digest = "unfingerprintable:" + uuid.uuid4().hex
        canonical = json.dumps({"inspection_id": self.inspection_id, "input": self.input.to_dict(),
                                "frame_sha256": frame_digest}, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class InferenceResult:
    inspection_id: str
    runtime: RuntimeRef
    input: InputDescriptor
    status: InferenceStatus
    model: ModelRef | None = None
    predictions: tuple[Prediction, ...] = ()
    inference_ms: float = 0.0
    error_code: str | None = None
    completed_at_utc: str = field(default_factory=utc_now_iso)
    software_version: str = __version__
    schema_version: str = INFERENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require(self.schema_version == INFERENCE_SCHEMA_VERSION, f"unsupported schema_version {self.schema_version!r}")
        validate_utc(self.completed_at_utc, "completed_at_utc")
        validate_non_empty_str(self.software_version, "software_version")
        _require(isinstance(self.inspection_id, str) and bool(self.inspection_id), "inspection_id required")
        _require(isinstance(self.status, InferenceStatus), "status must be InferenceStatus")
        _require(isinstance(self.runtime, RuntimeRef), "runtime must be RuntimeRef")
        _require(isinstance(self.input, InputDescriptor), "input must be InputDescriptor")
        _require(self.model is None or isinstance(self.model, ModelRef), "model must be ModelRef or None")
        _require(isinstance(self.predictions, tuple) and all(isinstance(p, Prediction) for p in self.predictions),
                 "predictions must be a tuple of Prediction")
        _require(self.error_code is None or isinstance(self.error_code, str), "error_code must be str or None")
        _finite_non_negative(self.inference_ms, "inference_ms")
        if self.status is InferenceStatus.OK:
            _require(self.model is not None, "status OK requires an identified model (id/version/sha256)")
            _require(self.error_code is None, "status OK must not carry error_code")
        else:
            _require(bool(self.error_code), "non-OK status requires error_code")
            _require(len(self.predictions) == 0, "non-OK status must not carry predictions")

    @property
    def is_usable(self) -> bool:
        """True only when the output may be considered by a decision rule."""
        return self.status is InferenceStatus.OK and self.model is not None and self.input.quality is ImageQuality.VALID

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "inspection_id": self.inspection_id,
            "model": self.model.to_dict() if self.model else None,
            "runtime": self.runtime.to_dict(),
            "input": self.input.to_dict(),
            "predictions": [p.to_dict() for p in self.predictions],
            "inference_ms": self.inference_ms,
            "status": self.status.value,
            "error_code": self.error_code,
            "completed_at_utc": self.completed_at_utc,
            "software_version": self.software_version,
        }

    _KEYS = ("schema_version", "inspection_id", "model", "runtime", "input", "predictions", "inference_ms",
             "status", "error_code", "completed_at_utc", "software_version")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InferenceResult:
        require_keys(data, cls._KEYS, "inference")
        try:
            status = InferenceStatus(data["status"])
            return cls(
                schema_version=data["schema_version"],
                inspection_id=data["inspection_id"],
                model=ModelRef.from_dict(data["model"]) if data["model"] is not None else None,
                runtime=RuntimeRef.from_dict(data["runtime"]),
                input=InputDescriptor.from_dict(data["input"]),
                predictions=tuple(Prediction.from_dict(p) for p in data["predictions"]),
                inference_ms=data["inference_ms"],
                status=status,
                error_code=data["error_code"],
                completed_at_utc=data["completed_at_utc"],
                software_version=data["software_version"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ContractError):
                raise
            raise ContractError(f"invalid inference payload: {exc}") from exc


@runtime_checkable
class InferenceAdapter(Protocol):
    """Interface every inference adapter must implement (no domain imports)."""

    adapter_id: str
    adapter_version: str

    @property
    def model(self) -> ModelRef | None: ...

    @property
    def ready(self) -> bool: ...

    def infer(self, request: InferenceRequest) -> InferenceResult: ...
