"""Versioned, framework-independent contracts (Vol.02 §6.2, §17)."""

from acurix_edge.contracts.decision import Decision, DecisionReason, InspectionResult
from acurix_edge.contracts.errors import ContractError
from acurix_edge.contracts.events import EVENT_SCHEMA_VERSION, EventEnvelope, ReviewStatus
from acurix_edge.contracts.inference import (
    INFERENCE_SCHEMA_VERSION,
    ImageQuality,
    InferenceAdapter,
    InferenceRequest,
    InferenceResult,
    InferenceStatus,
    InputDescriptor,
    ModelRef,
    Prediction,
    RuntimeRef,
)

__all__ = [
    "ContractError",
    "INFERENCE_SCHEMA_VERSION",
    "ImageQuality",
    "InferenceAdapter",
    "InferenceRequest",
    "InferenceResult",
    "InferenceStatus",
    "InputDescriptor",
    "ModelRef",
    "Prediction",
    "RuntimeRef",
    "DecisionReason",
    "InspectionResult",
    "Decision",
    "EVENT_SCHEMA_VERSION",
    "EventEnvelope",
    "ReviewStatus",
]
