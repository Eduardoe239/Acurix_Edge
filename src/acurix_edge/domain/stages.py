"""Inspection stage interface (AEX-CV-007 part 1; used by the cascade orchestrator in 0.4.0).

A stage receives a request whose frame ALREADY passed the quality gate and returns an
InferenceResult (it never decides PASS/DEFECT). Every stage declares:
- `stage_id` / `stage_kind` (quality_gate | anomaly | router | specialist — same values as the
  `cascade_step.stage` column of migration 2);
- `model`: the verified artifact (id, version, sha256) it runs with;
- `approved`: taken from ModelPolicy (approved_models) — never self-declared, never from the inventory.

`guarded_run` is the only way the application calls a stage: exceptions, wrong return types,
mismatched ids or models become non-OK results (never PASS).

SHADOW MODE — owner exception to ADR-017 item 6 (13/11/2026; the ADR must be amended):
- (a) only an UNAPPROVED stage whose artifact is in the separate inventory as NOT_APPROVED
  (`may_run_unapproved=True`, decided by the application) is CALLED; any other unapproved stage is
  NOT called (ADR-017 item 6 as written) and the decision gets MODEL_NOT_APPROVED;
- (b) a shadow output is recorded in the event marked "shadow" and NEVER reaches the decision rule:
  the decision layer can only ever see INCOMPATIBLE/MODEL_NOT_APPROVED (no class, no confidence);
- (c) property-tested: a shadow result never changes the decision (tests/unit/test_cv_007.py).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from acurix_edge.contracts.inference import (
    InferenceRequest,
    InferenceResult,
    InferenceStatus,
    ModelRef,
    RuntimeRef,
)

log = logging.getLogger("acurix_edge.stages")

STAGE_KINDS = frozenset({"quality_gate", "anomaly", "router", "specialist"})
GUARD_RUNTIME = RuntimeRef(name="stage-guard", version="1", device="cpu")


@runtime_checkable
class InspectionStage(Protocol):
    @property
    def stage_id(self) -> str: ...

    @property
    def stage_kind(self) -> str: ...

    @property
    def model(self) -> ModelRef | None: ...

    @property
    def approved(self) -> bool: ...

    def run(self, request: InferenceRequest) -> InferenceResult: ...


@dataclass(frozen=True, slots=True)
class UnavailableStage:
    """Placeholder for a stage whose artifact was rejected (CAS-005): always NO_MODEL."""

    stage_id: str
    stage_kind: str
    reason: str

    @property
    def model(self) -> ModelRef | None:
        return None

    @property
    def approved(self) -> bool:
        return False

    def run(self, request: InferenceRequest) -> InferenceResult:
        return InferenceResult(inspection_id=request.inspection_id, runtime=GUARD_RUNTIME, input=request.input,
                               status=InferenceStatus.NO_MODEL, error_code=f"STAGE_UNAVAILABLE:{self.reason}")


@dataclass(frozen=True, slots=True)
class StageRun:
    stage_id: str
    stage_kind: str
    approved: bool
    shadow: bool                   # True: ran, recorded, but NOT usable for the decision
    raw: InferenceResult           # what the stage produced (after contract checks)
    for_decision: InferenceResult  # what the decision layer is allowed to see
    elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        if self.shadow:
            notice = "SHADOW: stage NOT APPROVED; this output is NOT a decision and must never be shown as one"
        elif self.approved:
            notice = "approved stage"
        else:
            notice = "NOT CALLED: stage not approved and not in the inventory as NOT_APPROVED"
        return {"stage_id": self.stage_id, "stage_kind": self.stage_kind, "approved": self.approved,
                "shadow": self.shadow, "notice": notice, "elapsed_ms": self.elapsed_ms,
                "result": self.raw.to_dict(),
                "decision_input_status": self.for_decision.status.value,
                "decision_input_error": self.for_decision.error_code}


def _failure(request: InferenceRequest, status: InferenceStatus, code: str, model: ModelRef | None = None
             ) -> InferenceResult:
    return InferenceResult(inspection_id=request.inspection_id, runtime=GUARD_RUNTIME, input=request.input,
                           status=status, error_code=code, model=model)


def guarded_run(stage: InspectionStage, request: InferenceRequest, *, may_run_unapproved: bool = False
                ) -> StageRun:
    """`may_run_unapproved`: the application confirmed the stage's artifact is inventoried as
    NOT_APPROVED (shadow exception). Without it an unapproved stage is never called."""
    started = time.perf_counter()
    try:
        approved = stage.approved is True
        expected_model = stage.model
        stage_id, stage_kind = str(stage.stage_id), str(stage.stage_kind)
    except Exception as exc:
        log.error("stage identity unreadable", extra={"code": "STAGE_IDENTITY_ERROR",
                                                      "ctx": {"error": type(exc).__name__}})
        bad = _failure(request, InferenceStatus.INCOMPATIBLE, "STAGE_IDENTITY_ERROR")
        return StageRun("unknown", "unknown", False, False, bad, bad, (time.perf_counter() - started) * 1000.0)
    if stage_kind not in STAGE_KINDS:
        bad = _failure(request, InferenceStatus.INCOMPATIBLE, "STAGE_KIND_UNKNOWN")
        return StageRun(stage_id, stage_kind, False, False, bad, bad, (time.perf_counter() - started) * 1000.0)
    if not approved and not may_run_unapproved:
        # ADR-017 item 6: an unapproved stage outside the shadow exception is NOT called.
        skipped = _failure(request, InferenceStatus.NO_MODEL, "STAGE_NOT_CALLED:NOT_APPROVED_NOT_INVENTORIED")
        blocked = _failure(request, InferenceStatus.INCOMPATIBLE, "MODEL_NOT_APPROVED", model=None)
        return StageRun(stage_id, stage_kind, False, False, skipped, blocked, (time.perf_counter() - started) * 1000.0)
    try:
        out: object = stage.run(request)
    except Exception as exc:
        log.error("stage raised", extra={"code": "STAGE_EXCEPTION", "ctx": {"stage": stage_id,
                                                                           "error": type(exc).__name__}})
        out = _failure(request, InferenceStatus.RUNTIME_ERROR, f"STAGE_EXCEPTION:{type(exc).__name__}")
    if not isinstance(out, InferenceResult):
        raw = _failure(request, InferenceStatus.INCOMPATIBLE, "STAGE_INVALID_RETURN_TYPE")
    elif out.inspection_id != request.inspection_id:
        raw = _failure(request, InferenceStatus.INCOMPATIBLE, "INSPECTION_ID_MISMATCH")
    elif out.model != expected_model and out.status is InferenceStatus.OK:
        raw = _failure(request, InferenceStatus.INCOMPATIBLE, "MODEL_IDENTITY_MISMATCH")
    elif out.input != request.input:
        raw = _failure(request, InferenceStatus.INCOMPATIBLE, "INPUT_DESCRIPTOR_MISMATCH")
    else:
        raw = out
    if approved:
        for_decision, shadow = raw, False
    else:
        # DEC-001 / ADR-016: an unapproved stage never feeds the decision, whatever it measured.
        for_decision = _failure(request, InferenceStatus.INCOMPATIBLE, "MODEL_NOT_APPROVED", model=None)
        shadow = True
    return StageRun(stage_id, stage_kind, approved, shadow, raw, for_decision,
                    (time.perf_counter() - started) * 1000.0)
