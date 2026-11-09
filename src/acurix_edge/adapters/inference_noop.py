"""NoOp inference adapter.

Used while no model is approved. It never produces predictions and always
returns status NO_MODEL, which the DecisionEngine maps to NO_DECISION.
It is NOT a demonstration of accuracy and must never be presented as one.
"""

from __future__ import annotations

import platform
import time

from acurix_edge.contracts.inference import (
    InferenceRequest,
    InferenceResult,
    InferenceStatus,
    ModelRef,
    RuntimeRef,
)


class NoOpInferenceAdapter:
    adapter_id = "noop"
    adapter_version = "0.1.0"

    def __init__(self) -> None:
        self._runtime = RuntimeRef(name="noop", version=platform.python_version(), device="cpu")

    @property
    def model(self) -> ModelRef | None:
        return None

    @property
    def ready(self) -> bool:
        # No model loaded: the adapter is never ready to support a decision.
        return False

    def infer(self, request: InferenceRequest) -> InferenceResult:
        start = time.perf_counter()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return InferenceResult(
            inspection_id=request.inspection_id,
            model=None,
            runtime=self._runtime,
            input=request.input,
            predictions=(),
            inference_ms=elapsed_ms,
            status=InferenceStatus.NO_MODEL,
            error_code="NO_MODEL_LOADED",
        )
