"""Normalized Inference Adapter contract (Vol.02 §17, schema 1.0).

The adapter invokes a runtime/model and normalizes its output. It NEVER decides
PASS/DEFECT: that belongs to the DecisionEngine. Any output without a valid
model, with invalid input or an incompatible runtime must carry a non-OK
status so that the decision layer returns NO_DECISION.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from acurix_edge.contracts.errors import ContractError

INFERENCE_SCHEMA_VERSION = "1.0"
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
        _require(bool(isinstance(self.sha256, str) and _SHA256_RE.fullmatch(self.sha256)), "model sha256 must be 64 lowercase hex chars")

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "version": self.version, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelRef":
        if not isinstance(data, dict):
            raise ContractError("model must be an object")
        return cls(id=data.get("id"), version=data.get("version"), sha256=data.get("sha256"))  # no str() coercion


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
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeRef":
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
    def from_dict(cls, data: dict[str, Any]) -> "InputDescriptor":
        try:
            quality = ImageQuality(data["quality"])
        except ValueError as exc:
            raise ContractError(f"unknown image quality {data.get('quality')!r}") from exc
        return cls(width=data["width"], height=data["height"], quality=quality)


@dataclass(frozen=True, slots=True)
class Prediction:
    """Technical model output. `confidence` is NOT a calibrated probability."""

    class_id: str
    confidence: float
    bbox: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
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
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Prediction":
        bbox = data.get("bbox")
        return cls(
            class_id=str(data["class_id"]),
            confidence=data["confidence"],
            bbox=tuple(bbox) if bbox is not None else None,  # type: ignore[arg-type]
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
    schema_version: str = INFERENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require(self.schema_version == INFERENCE_SCHEMA_VERSION, f"unsupported schema_version {self.schema_version!r}")
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
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InferenceResult":
        try:
            status = InferenceStatus(data["status"])
            return cls(
                schema_version=str(data["schema_version"]),
                inspection_id=str(data["inspection_id"]),
                model=ModelRef.from_dict(data["model"]) if data.get("model") else None,
                runtime=RuntimeRef.from_dict(data["runtime"]),
                input=InputDescriptor.from_dict(data["input"]),
                predictions=tuple(Prediction.from_dict(p) for p in data.get("predictions", [])),
                inference_ms=data["inference_ms"],
                status=status,
                error_code=data.get("error_code"),
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
