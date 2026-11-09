"""Event Service: builds traceable envelopes and persists them.

- Persistence failure is never reported as persisted (Vol.02 §5 rule 4); it
  increments `failed` and drives the system to FAULT.
- Idempotent: re-submitting the same event_id/payload is counted as duplicate.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass

from acurix_edge.contracts.decision import Decision, InspectionResult
from acurix_edge.contracts.events import EventEnvelope, ReviewStatus
from acurix_edge.contracts.inference import InferenceResult
from acurix_edge.domain.state_machine import StateMachine
from acurix_edge.persistence.sqlite_store import (
    AppendOutcome,
    IdempotencyConflict,
    PersistenceError,
    SqliteEventStore,
)

log = logging.getLogger("acurix_edge.events")


@dataclass
class EventCounters:
    """`created == persisted + duplicates + failed` (envelope level).

    `inspection_duplicates` counts retries of an already recorded (session, inspection_id) that
    were answered from storage without building a new envelope; `corrupt_records` counts stored
    payloads that failed decoding/integrity checks (R4-03)."""

    created: int = 0
    persisted: int = 0
    duplicates: int = 0
    failed: int = 0
    inspection_duplicates: int = 0
    corrupt_records: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RecordResult:
    event_id: str | None  # None only when no event could be identified (e.g. corrupt stored record)
    persisted: bool
    outcome: AppendOutcome | None
    error: str | None = None


class EventService:
    def __init__(
        self,
        store: SqliteEventStore,
        state: StateMachine,
        *,
        site_id: str,
        station_id: str,
        configuration: dict[str, str],
        software_version: str,
    ) -> None:
        self._store = store
        self._state = state
        self._site_id = site_id
        self._station_id = station_id
        self._configuration = dict(configuration)
        self._software_version = software_version
        self._counters = EventCounters()
        self._lock = threading.Lock()

    def counters(self) -> EventCounters:
        with self._lock:
            return EventCounters(**asdict(self._counters))

    def note_inspection_duplicate(self, *, session_id: str, inspection_id: str, event_id: str, result: str) -> None:
        with self._lock:
            self._counters.inspection_duplicates += 1
        log.warning("inspection retried; returning stored outcome (no new event)", extra={
            "code": "INSPECTION_DUPLICATE",
            "ctx": {"session_id": session_id, "inspection_id": inspection_id, "event_id": event_id, "result": result}})

    def note_corrupt_record(self, *, session_id: str, inspection_id: str, error: str) -> None:
        with self._lock:
            self._counters.corrupt_records += 1
        log.critical("stored event record is corrupt", extra={
            "code": "EVENT_RECORD_CORRUPT",
            "ctx": {"session_id": session_id, "inspection_id": inspection_id, "error": error}})
        self._state.enter_fault("EVENT_RECORD_CORRUPT")

    def build(
        self,
        *,
        decision: Decision,
        inference: InferenceResult,
        session_id: str,
        captured_at_utc: str,
        capture_sequence: int,
        event_id: str | None = None,
    ) -> EventEnvelope:
        return EventEnvelope(
            event_id=event_id or EventEnvelope.new_id(),
            site_id=self._site_id,
            station_id=self._station_id,
            session_id=session_id,
            inspection_id=inference.inspection_id,
            captured_at_utc=captured_at_utc,
            capture_sequence=capture_sequence,
            result=decision.result,
            reason=decision.reason.value,
            inference_status=inference.status.value,
            rule=(
                {"id": decision.rule_id, "version": decision.rule_version}
                if decision.rule_id and decision.rule_version
                else None
            ),
            class_id=decision.class_id,
            confidence=decision.confidence,
            model=inference.model.to_dict() if inference.model else None,
            configuration=self._configuration,
            review_status=ReviewStatus.PENDING if decision.result is InspectionResult.REVIEW else ReviewStatus.NOT_REQUIRED,
            software_version=self._software_version,
            extra={
                "inference_error_code": inference.error_code,
                "decision_detail": decision.detail,
                "runtime": inference.runtime.to_dict(),
            },
        )

    def record(self, envelope: EventEnvelope) -> RecordResult:
        with self._lock:
            self._counters.created += 1
        try:
            outcome = self._store.append(envelope)
        except IdempotencyConflict as exc:
            with self._lock:
                self._counters.failed += 1
            log.error("event idempotency conflict", extra={"code": "EVENT_ID_CONFLICT", "ctx": {"event_id": envelope.event_id}})
            return RecordResult(envelope.event_id, False, None, str(exc))
        except Exception as exc:  # PersistenceError or any unexpected failure: never report as persisted
            with self._lock:
                self._counters.failed += 1
            log.critical(
                "event persistence failed",
                extra={"code": "PERSISTENCE_FAILURE", "ctx": {"event_id": envelope.event_id, "error": type(exc).__name__}},
            )
            self._state.enter_fault("PERSISTENCE_FAILURE")
            detail = str(exc) if isinstance(exc, PersistenceError) else type(exc).__name__
            return RecordResult(envelope.event_id, False, None, detail)
        with self._lock:
            if outcome is AppendOutcome.INSERTED:
                self._counters.persisted += 1
            else:
                self._counters.duplicates += 1
        if outcome is AppendOutcome.DUPLICATE:
            log.warning("event re-submitted with identical payload", extra={
                "code": "EVENT_DUPLICATE", "ctx": {"event_id": envelope.event_id,
                                                    "inspection_id": envelope.inspection_id}})
        return RecordResult(envelope.event_id, outcome is AppendOutcome.INSERTED or outcome is AppendOutcome.DUPLICATE, outcome)
