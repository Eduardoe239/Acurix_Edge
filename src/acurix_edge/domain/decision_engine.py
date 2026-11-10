"""Decision Engine with fail-safe guard (Vol.02 §4.3, §17).

PASS/DEFECT can only be produced by an identified, versioned and explicitly
approved DecisionRule, with a usable inference result, valid input quality,
system in OBSERVING and health OK. Every other path yields NO_DECISION.

Foundation 0.1.0 ships NO concrete rule: the first use case is NOT_FROZEN
(Execution Baseline v1.2.1 §7). Classes and thresholds must not be invented.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

from acurix_edge.contracts.decision import Decision, DecisionReason, InspectionResult
from acurix_edge.contracts.inference import ImageQuality, InferenceResult, InferenceStatus
from acurix_edge.domain.state_machine import RESULT_CAPABLE_STATES, SystemState

log = logging.getLogger("acurix_edge.decision")


@dataclass(frozen=True, slots=True)
class DecisionContext:
    system_state: SystemState
    health_ok: bool


@runtime_checkable
class DecisionRule(Protocol):
    rule_id: str
    rule_version: str
    approved: bool  # set only from a versioned, owner-approved configuration

    def evaluate(self, inference: InferenceResult) -> Decision: ...


def is_rule_approved(rule: object | None) -> bool:
    """Strict check: `approved` must be the boolean True (not merely truthy)."""
    return (
        rule is not None
        and getattr(rule, "approved", False) is True
        and isinstance(getattr(rule, "rule_id", None), str) and bool(getattr(rule, "rule_id"))
        and isinstance(getattr(rule, "rule_version", None), str) and bool(getattr(rule, "rule_version"))
    )


def _no_decision(reason: DecisionReason, detail: str) -> Decision:
    return Decision(result=InspectionResult.NO_DECISION, reason=reason, detail=detail)


class DecisionEngine:
    def __init__(self, rule: DecisionRule | None = None) -> None:
        self._rule = rule

    @property
    def rule(self) -> DecisionRule | None:
        return self._rule

    def decide(self, inference: InferenceResult, ctx: DecisionContext) -> Decision:
        """Every returned Decision carries the latency measured here (a rule cannot set its own)."""
        start = time.perf_counter()
        decision = self._decide(inference, ctx)
        return replace(decision, latency_ms=(time.perf_counter() - start) * 1000.0)

    def _decide(self, inference: InferenceResult, ctx: DecisionContext) -> Decision:
        if ctx.system_state not in RESULT_CAPABLE_STATES:
            return _no_decision(DecisionReason.SYSTEM_NOT_OBSERVING, f"system state {ctx.system_state}")
        if not ctx.health_ok:
            return _no_decision(DecisionReason.HEALTH_NOT_OK, "health not OK")
        if inference.status is not InferenceStatus.OK or inference.model is None:
            return _no_decision(DecisionReason.INFERENCE_NOT_OK, f"inference status {inference.status} ({inference.error_code})")
        if inference.input.quality is not ImageQuality.VALID:
            return _no_decision(DecisionReason.INPUT_QUALITY, f"input quality {inference.input.quality}")
        rule = self._rule
        if not is_rule_approved(rule):
            return _no_decision(DecisionReason.NO_APPROVED_RULE, "no approved decision rule (use case NOT_FROZEN)")
        try:
            decision = rule.evaluate(inference)
        except Exception as exc:
            log.exception("decision rule failed", extra={"code": "DECISION_RULE_ERROR"})
            return _no_decision(DecisionReason.RULE_ERROR, type(exc).__name__)
        if not isinstance(decision, Decision):
            return _no_decision(DecisionReason.RULE_ERROR, "rule returned invalid type")
        if decision.result in (InspectionResult.PASS, InspectionResult.DEFECT) and (
            decision.rule_id != rule.rule_id or decision.rule_version != rule.rule_version
        ):
            return _no_decision(DecisionReason.RULE_ERROR, "rule identity mismatch")
        return decision
