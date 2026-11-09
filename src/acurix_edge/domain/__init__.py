"""Domain layer: state machine and decision engine (framework-independent)."""

from acurix_edge.domain.decision_engine import DecisionContext, DecisionEngine, DecisionRule
from acurix_edge.domain.state_machine import (
    ALLOWED_TRANSITIONS,
    InvalidTransition,
    StateMachine,
    SystemState,
    TransitionRecord,
)

__all__ = [
    "ALLOWED_TRANSITIONS",
    "DecisionContext",
    "DecisionEngine",
    "DecisionRule",
    "InvalidTransition",
    "StateMachine",
    "SystemState",
    "TransitionRecord",
]
