"""Test-only fixtures. Nothing here is a real model, rule, class or threshold.

`FixtureOkAdapter` and `FixtureApprovedRule` exist only to prove that the
decision guard lets a conclusive result through when *every* pre-condition is
satisfied, and blocks it otherwise. They are not product behavior.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from acurix_edge.contracts.decision import Decision, DecisionReason, InspectionResult
from acurix_edge.contracts.inference import (
    ImageQuality,
    InferenceRequest,
    InferenceResult,
    InferenceStatus,
    InputDescriptor,
    ModelRef,
    Prediction,
    RuntimeRef,
)

FIXTURE_SHA = "0" * 64
FIXTURE_MODEL = ModelRef(id="fixture-model", version="0.0.0-test", sha256=FIXTURE_SHA)
FIXTURE_RUNTIME = RuntimeRef(name="fixture", version="0", device="cpu")

def request(quality: ImageQuality = ImageQuality.VALID, inspection_id: str = "insp-1") -> InferenceRequest:
    return InferenceRequest(inspection_id=inspection_id, input=InputDescriptor(64, 48, quality))


def ok_result(quality: ImageQuality = ImageQuality.VALID) -> InferenceResult:
    return InferenceResult(
        inspection_id="insp-1",
        model=FIXTURE_MODEL,
        runtime=FIXTURE_RUNTIME,
        input=InputDescriptor(64, 48, quality),
        predictions=(Prediction("fixture_class", 0.9),),
        inference_ms=1.0,
        status=InferenceStatus.OK,
    )


class FixtureOkAdapter:
    adapter_id = "fixture-ok"
    adapter_version = "0.0.0-test"
    model = FIXTURE_MODEL
    ready = True

    def infer(self, req: InferenceRequest) -> InferenceResult:
        return InferenceResult(
            inspection_id=req.inspection_id, model=FIXTURE_MODEL, runtime=FIXTURE_RUNTIME, input=req.input,
            predictions=(Prediction("fixture_class", 0.9),), inference_ms=1.0, status=InferenceStatus.OK,
        )


# Test-only policy seam: approves ONLY the fixture model so the guard can be exercised.
# Adapters are authorized by exact class (R4-09); test subclasses must be registered explicitly.
# The product policy (from configuration) is always default-deny in 0.1.0.
from acurix_edge.adapters.inference_noop import NoOpInferenceAdapter  # noqa: E402
from acurix_edge.domain.model_policy import ModelPolicy  # noqa: E402

FIXTURE_POLICY = ModelPolicy(approved_models=frozenset({FIXTURE_MODEL}), real_model_loading=True,
                             allowed_adapters=frozenset({NoOpInferenceAdapter, FixtureOkAdapter}))


def fixture_policy_with(*adapter_classes: type, base: ModelPolicy = FIXTURE_POLICY) -> ModelPolicy:
    return ModelPolicy(approved_models=base.approved_models, real_model_loading=base.real_model_loading,
                       allowed_adapters=base.allowed_adapters | frozenset(adapter_classes))


class FixtureApprovedRule:
    rule_id = "fixture-rule"
    rule_version = "0.0.0-test"

    def __init__(self, approved: bool = True, result: InspectionResult = InspectionResult.PASS) -> None:
        self.approved = approved
        self._result = result
        self.calls = 0

    def evaluate(self, inference: InferenceResult) -> Decision:
        self.calls += 1
        return Decision(self._result, DecisionReason.RULE_APPLIED, self.rule_id, self.rule_version)


def _fixture_host(status_name: str):
    from acurix_edge.health.service import ComponentStatus

    status = ComponentStatus(status_name)
    return lambda: (status, "fixture host probe")


# Host probes for tests that need OBSERVING independent of the real machine (this dev host often
# has < 1.5 GB free RAM, which correctly yields DEGRADED under ADR-014). Real probes are tested
# separately in tests/unit/test_health_005.py.
HEALTHY_HOST = {"clock": _fixture_host("OK"), "memory": _fixture_host("OK"), "cpu": _fixture_host("OK"),
                "gpu": _fixture_host("DISABLED")}


def tamper_event_row(conn, sql: str, params: tuple[object, ...] = ()) -> None:
    """Simulates EXTERNAL corruption/tampering of inspection_event (R4-03 tests). Since migration 3
    the database aborts any UPDATE; someone with raw write access to the file can still drop the
    trigger, which is exactly what this does before re-creating it."""
    from acurix_edge.persistence.sqlite_store import _MIGRATIONS, IMMUTABILITY_TRIGGER

    conn.execute(f"DROP TRIGGER IF EXISTS {IMMUTABILITY_TRIGGER}")
    try:
        conn.execute(sql, params)
    finally:
        conn.execute(_MIGRATIONS[3][0])


class TempDirTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
