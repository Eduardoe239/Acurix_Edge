"""Explicit system state machine (Vol.02 §5, Manual Mestre §8).

Single source of truth for system state. Invalid transitions are rejected,
recorded and logged. The machine always boots in IDLE: a restart never reopens
an inspection session implicitly.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from acurix_edge.contracts.events import utc_now_iso

log = logging.getLogger("acurix_edge.state")


class SystemState(StrEnum):
    IDLE = "IDLE"
    INITIALIZING = "INITIALIZING"
    OBSERVING = "OBSERVING"
    DEGRADED = "DEGRADED"
    FAULT = "FAULT"
    MAINTENANCE = "MAINTENANCE"


S = SystemState
ALLOWED_TRANSITIONS: dict[SystemState, frozenset[SystemState]] = {
    S.IDLE: frozenset({S.INITIALIZING, S.MAINTENANCE, S.FAULT}),
    S.INITIALIZING: frozenset({S.OBSERVING, S.DEGRADED, S.FAULT, S.IDLE}),
    S.OBSERVING: frozenset({S.DEGRADED, S.FAULT, S.IDLE}),
    S.DEGRADED: frozenset({S.OBSERVING, S.FAULT, S.IDLE}),
    # FAULT is latched (R4-01): no ordinary transition leaves it. The only exit is
    # release_fault(), called exclusively by the audited Application.reset_fault().
    S.FAULT: frozenset(),
    S.MAINTENANCE: frozenset({S.IDLE, S.FAULT}),
}
FAULT_RELEASE_TARGETS = frozenset({S.IDLE})

# States in which conclusive inspection results may be considered at all.
RESULT_CAPABLE_STATES = frozenset({S.OBSERVING})


class InvalidTransition(RuntimeError):
    def __init__(self, source: SystemState, target: SystemState, reason: str) -> None:
        super().__init__(f"transition {source} -> {target} not allowed ({reason})")
        self.source = source
        self.target = target


@dataclass(frozen=True, slots=True)
class TransitionRecord:
    seq: int
    at_utc: str
    source: SystemState
    target: SystemState
    reason: str
    accepted: bool


Listener = Callable[[TransitionRecord], None]


class StateMachine:
    def __init__(self, history_size: int = 256) -> None:
        self._state = SystemState.IDLE
        self._lock = threading.RLock()
        self._history: deque[TransitionRecord] = deque(maxlen=history_size)
        self._listeners: list[Listener] = []
        self._rejected = 0
        self._seq = 0

    @property
    def state(self) -> SystemState:
        with self._lock:
            return self._state

    @property
    def rejected_count(self) -> int:
        with self._lock:
            return self._rejected

    def history(self) -> list[TransitionRecord]:
        with self._lock:
            return list(self._history)

    def subscribe(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def can_transition(self, target: SystemState | str) -> bool:
        target = SystemState(target)
        with self._lock:
            return target in ALLOWED_TRANSITIONS[self._state]

    def _apply(self, target: SystemState, reason: str) -> TransitionRecord:
        # caller holds self._lock
        source = self._state
        accepted = target in ALLOWED_TRANSITIONS[source]
        self._seq += 1
        record = TransitionRecord(self._seq, utc_now_iso(), source, target, reason, accepted)
        self._history.append(record)
        if accepted:
            self._state = target
        else:
            self._rejected += 1
        return record

    @staticmethod
    def _coerce(target: SystemState | str, reason: str) -> SystemState:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("transition reason is mandatory")
        try:
            return SystemState(target)
        except ValueError as exc:
            raise ValueError(f"unknown state {target!r}") from exc

    def transition(self, target: SystemState | str, reason: str) -> TransitionRecord:
        target = self._coerce(target, reason)
        with self._lock:
            record = self._apply(target, reason)
        self._finish(record)
        return record

    def _finish(self, record: TransitionRecord) -> None:
        """Emit to listeners outside the lock, then log/raise."""
        self._emit(record)
        source, target, reason, accepted = record.source, record.target, record.reason, record.accepted
        if not accepted:
            log.warning(
                "state transition rejected",
                extra={"code": "STATE_TRANSITION_REJECTED", "ctx": {"source": source, "target": target, "reason": reason}},
            )
            raise InvalidTransition(source, target, reason)
        log.info(
            "state transition",
            extra={"code": "STATE_TRANSITION", "ctx": {"source": source, "target": target, "reason": reason}},
        )

    def enter_fault(self, reason: str) -> TransitionRecord | None:
        """Move to FAULT from any state. Idempotent when already in FAULT.

        Check-and-apply is atomic; listeners run outside the lock like transition().
        """
        target = self._coerce(SystemState.FAULT, reason)
        with self._lock:
            if self._state is SystemState.FAULT:
                record = None
            else:
                record = self._apply(target, reason)
        if record is None:
            log.error("fault reported while in FAULT", extra={"code": "FAULT_REPEATED", "ctx": {"reason": reason}})
            return None
        self._finish(record)
        return record

    def release_fault(self, reason: str, target: SystemState = SystemState.IDLE) -> TransitionRecord:
        """Leave the FAULT latch. Only Application.reset_fault() may call this, after its
        audited pre-conditions (healthy critical components, latch cleared) are met."""
        target = self._coerce(target, reason)
        with self._lock:
            source = self._state
            if source is not SystemState.FAULT or target not in FAULT_RELEASE_TARGETS:
                self._seq += 1
                record = TransitionRecord(self._seq, utc_now_iso(), source, target, reason, False)
                self._history.append(record)
                self._rejected += 1
            else:
                self._seq += 1
                record = TransitionRecord(self._seq, utc_now_iso(), source, target, reason, True)
                self._history.append(record)
                self._state = target
        self._finish(record)
        return record

    def _emit(self, record: TransitionRecord) -> None:
        for listener in list(self._listeners):
            try:
                listener(record)
            except Exception:  # listener failures must not corrupt state
                log.exception("state listener failed", extra={"code": "STATE_LISTENER_ERROR"})
