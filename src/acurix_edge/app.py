"""Application composition root (modular monolith, single process).

Lifecycle: boot -> IDLE (never auto-resumes a session) -> start_session() ->
INITIALIZING -> OBSERVING | DEGRADED -> close_session() -> IDLE.
Observation mode only: no device/actuator output exists in 0.1.0.

FAULT latch (R4-01): FAULT is left ONLY through the audited reset_fault().
close_session()/shutdown() end the session but keep FAULT. The latch is persisted
(latch file in the data directory + audit trail) and restored at boot, so a
restart never clears a fault silently. Either source alone latches (fail-safe).

Guarantees enforced here:
- health is re-evaluated for every inspection (no cached report authorizes PASS);
- adapter exceptions/mismatched outputs become non-OK inference results and
  are still recorded (never silently dropped, never PASS);
- `process()` is serialized (single writer of session sequence).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from acurix_edge import SOFTWARE_NAME, __version__
from acurix_edge.adapters.inference_noop import NoOpInferenceAdapter
from acurix_edge.config.settings import AppConfig
from acurix_edge.contracts.decision import Decision, DecisionReason, InspectionResult
from acurix_edge.contracts.errors import ContractError
from acurix_edge.contracts.events import validate_utc, utc_now_iso
from acurix_edge.contracts.inference import (
    InferenceAdapter,
    InferenceRequest,
    InferenceResult,
    InferenceStatus,
    RuntimeRef,
)
from acurix_edge.domain.decision_engine import DecisionContext, DecisionEngine, DecisionRule, is_rule_approved
from acurix_edge.domain.model_policy import ModelPolicy
from acurix_edge.domain.state_machine import StateMachine, SystemState, TransitionRecord
from acurix_edge.events.service import EventService, RecordResult
from acurix_edge.health.service import ComponentStatus, HealthReport, HealthService, host_memory, storage_check
from acurix_edge.observability.logging import bind_context
from acurix_edge.persistence.sqlite_store import (
    AppendOutcome,
    CorruptRecordError,
    PersistenceError,
    SqliteEventStore,
)

log = logging.getLogger("acurix_edge.app")

_ADAPTERS = {"noop": NoOpInferenceAdapter}
FAULT_LATCH_FILE = "FAULT_LATCH.json"
CaptureProbe = Callable[[], tuple[ComponentStatus, str]]


class SessionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ProcessOutcome:
    record: RecordResult
    result: str
    reason: str


class Application:
    def __init__(
        self,
        config: AppConfig,
        base_dir: Path,
        *,
        inference: InferenceAdapter | None = None,
        rule: DecisionRule | None = None,
        store: SqliteEventStore | None = None,
        capture_probe: CaptureProbe | None = None,
        read_only: bool = False,
        model_policy: ModelPolicy | None = None,
    ) -> None:
        self.config = config
        self.base_dir = base_dir
        self.read_only = read_only
        self.state = StateMachine()
        # ADR-016 default-deny: product policy comes from (locked) configuration.
        self.model_policy = model_policy if model_policy is not None else ModelPolicy.from_config(config)
        self.inference: InferenceAdapter = inference or _ADAPTERS[config.inference.adapter]()
        # R4-09: authorization by the policy's trusted class registry, never by the adapter's own claim.
        if not self.model_policy.is_adapter_allowed(self.inference):
            raise ValueError(
                f"inference adapter class {type(self.inference).__name__} not authorized (ADR-016)")
        self.decision = DecisionEngine(rule)
        self.store = store or SqliteEventStore(config.database_path(base_dir), read_only=read_only)
        self.health = HealthService()
        self._capture_probe = capture_probe
        self.events = EventService(
            self.store,
            self.state,
            site_id=config.site_id,
            station_id=config.station_id,
            configuration=config.reference(),
            software_version=__version__,
        )
        self.session_id: str | None = None
        self._pending_session_id: str | None = None  # correlation for audits during start (R4-04)
        self._latch_path = config.database_path(base_dir).parent / FAULT_LATCH_FILE
        self._fault_latch: dict[str, Any] | None = None
        self._sequence = 0
        self._process_lock = threading.RLock()  # serializes process/start/close (session lifecycle)
        self._health_lock = threading.RLock()
        self._last_health: HealthReport | None = None
        self._store_open_error: str | None = None
        self.state.subscribe(self._audit_transition)
        self._register_checks()

    # -- boot ----------------------------------------------------------------
    def boot(self) -> None:
        try:
            self.store.open()
        except PersistenceError as exc:
            self._store_open_error = str(exc)
            log.critical("database unavailable at boot", extra={"code": "DB_UNAVAILABLE", "ctx": {"error": str(exc)}})
            self.state.enter_fault("DB_UNAVAILABLE_AT_BOOT")
        latch = self._read_persisted_latch()
        if latch is not None:
            self._fault_latch = latch
            if self.state.state is not SystemState.FAULT:
                self.state.enter_fault(f"FAULT_LATCH_RESTORED: {latch.get('reason', 'unknown')}")
        self._evaluate_health()

    def shutdown(self) -> None:
        """Ends any session and closes storage. Never leaves FAULT (the latch stays persisted)."""
        with self._process_lock:
            if self.session_id is not None:
                self.close_session()
            self.store.close()

    # -- FAULT latch (R4-01) ---------------------------------------------------
    def _read_persisted_latch(self) -> dict[str, Any] | None:
        """Latch from the latch file OR the audit trail. Unreadable latch data latches (fail-safe)."""
        try:
            raw = self._latch_path.read_bytes()
        except FileNotFoundError:
            raw = None
        except OSError as exc:
            return {"reason": "FAULT_LATCH_UNREADABLE", "source": "file", "error": type(exc).__name__}
        if raw is not None:
            try:
                data = json.loads(raw.decode("utf-8"))
                if not isinstance(data, dict) or not isinstance(data.get("reason"), str):
                    raise ValueError("latch structure")
                return {**data, "source": "file"}
            except (ValueError, UnicodeError):
                return {"reason": "FAULT_LATCH_CORRUPT", "source": "file"}
        if self.store.is_open:
            try:
                last = self.store.last_state_audit()
            except PersistenceError:
                return {"reason": "FAULT_LATCH_AUDIT_UNREADABLE", "source": "audit"}
            if last is not None and last["action"] == "STATE_TRANSITION" and last["target"].endswith("->FAULT"):
                detail = last["detail"] if isinstance(last["detail"], dict) else {}
                return {"reason": str(detail.get("reason", "unknown")), "at_utc": detail.get("at_utc"),
                        "source": "audit"}
        return None

    def _write_latch(self, reason: str, at_utc: str) -> None:
        if self.read_only or self._latch_path.exists():
            return  # read-only never writes; an existing latch keeps the original cause
        tmp = self._latch_path.with_name(self._latch_path.name + ".tmp")
        try:
            self._latch_path.parent.mkdir(parents=True, exist_ok=True)
            # I-01: a FAULT during start_session happens before session_id is assigned; the latch
            # must carry the attempted id (same correlation as the audit trail).
            sid = self.session_id or self._pending_session_id
            tmp.write_text(json.dumps({"reason": reason, "at_utc": at_utc, "session_id": sid},
                                      sort_keys=True), encoding="utf-8")
            os.replace(tmp, self._latch_path)
        except OSError as exc:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)  # best effort; the reader ignores a stray .tmp anyway
            # The in-memory FAULT still holds; the audit trail is the second latch source.
            log.critical("fault latch could not be persisted",
                         extra={"code": "FAULT_LATCH_WRITE_FAILED", "ctx": {"error": type(exc).__name__}})

    # -- health --------------------------------------------------------------
    def _register_checks(self) -> None:
        def database() -> tuple[ComponentStatus, str]:
            if not self.store.is_open:
                return ComponentStatus.FAULT, self._store_open_error or "database not open"
            ok, detail = self.store.check()
            return (ComponentStatus.OK if ok else ComponentStatus.FAULT), detail

        def model() -> tuple[ComponentStatus, str]:
            m = self.inference.model
            if not self.inference.ready or m is None:
                return ComponentStatus.DEGRADED, f"adapter '{self.inference.adapter_id}': no approved model loaded"
            if not self.model_policy.is_model_approved(m):
                return ComponentStatus.DEGRADED, f"model {m.id}@{m.version} not in approved_models (default-deny)"
            return ComponentStatus.OK, f"{m.id}@{m.version}"

        def decision_rule() -> tuple[ComponentStatus, str]:
            rule = self.decision.rule
            if is_rule_approved(rule):
                return ComponentStatus.OK, f"{rule.rule_id}@{rule.rule_version}"  # type: ignore[union-attr]
            return ComponentStatus.DEGRADED, "no approved decision rule (use case NOT_FROZEN)"

        def camera() -> tuple[ComponentStatus, str]:
            if self._capture_probe is None:
                return ComponentStatus.UNKNOWN, "capture adapter not implemented in 0.1.0"
            return self._capture_probe()

        self.health.register("database", database, critical=True)
        self.health.register(
            "storage",
            storage_check(self.config.database_path(self.base_dir).parent, self.config.storage.min_free_disk_mb),
            critical=True,
        )
        self.health.register("model", model, critical=False)
        self.health.register("decision_rule", decision_rule, critical=False)
        self.health.register("camera", camera, critical=False)

    def _evaluate_health(self) -> HealthReport:
        report = self.health.evaluate()
        self._last_health = report
        return report

    def refresh_health(self) -> HealthReport:
        # Lock order: process -> health -> store. Evaluate + state change are one critical section,
        # serialized with session start/close/processing (no lost or rejected transitions).
        with self._process_lock, self._health_lock:
            report = self._evaluate_health()
            current = self.state.state
            if report.overall is ComponentStatus.FAULT and current is not SystemState.FAULT:
                self.state.enter_fault("HEALTH_FAULT")
            elif current is SystemState.OBSERVING and report.overall is not ComponentStatus.OK:
                self.state.transition(SystemState.DEGRADED, "HEALTH_DEGRADED")
            elif current is SystemState.DEGRADED and report.ok:
                self.state.transition(SystemState.OBSERVING, "HEALTH_RECOVERED")
            return report

    # -- session -------------------------------------------------------------
    def start_session(self) -> str:
        with self._process_lock, self._health_lock:
            return self._start_session()

    def _start_session(self) -> str:
        if self.read_only:
            raise SessionError("application opened read-only")
        if self.state.state is not SystemState.IDLE:
            raise SessionError(f"cannot start session in state {self.state.state}")
        session_id = str(uuid.uuid4())
        self._pending_session_id = session_id  # audits of the start transitions carry this id (R4-04)
        try:
            self.state.transition(SystemState.INITIALIZING, "SESSION_START_REQUESTED")
            report = self._evaluate_health()
            if report.overall is ComponentStatus.FAULT:
                self.state.enter_fault("INIT_PRECONDITION_FAULT")
                raise SessionError("pre-conditions failed: " + ", ".join(
                    c.name for c in report.components if c.status is not ComponentStatus.OK))
            target = SystemState.OBSERVING if report.ok else SystemState.DEGRADED
            self.state.transition(target, "INIT_COMPLETE" if report.ok else "INIT_DEGRADED")
            self._sequence = 0
            self.session_id = session_id  # only after a successful transition
        finally:
            self._pending_session_id = None
        return session_id

    def reset_fault(self, reason: str, *, actor_ref: str = "operator") -> None:
        """The ONLY exit from FAULT (R4-01): explicit, audited recovery FAULT -> IDLE.

        Order (each step fail-safe): re-open storage if needed; critical health must be non-FAULT;
        write the FAULT_RESET audit (failure aborts); remove the latch file (failure aborts);
        release the state machine latch. Any previous session is discarded (never resumed)."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reset reason is mandatory")
        if not isinstance(actor_ref, str) or not actor_ref.strip():
            raise ValueError("actor_ref is mandatory")
        if self.read_only:
            raise SessionError("application opened read-only")
        with self._process_lock, self._health_lock:
            if self.state.state is not SystemState.FAULT:
                raise SessionError(f"reset only allowed from FAULT (current {self.state.state})")
            if not self.store.is_open:
                try:
                    self.store.open()
                    self._store_open_error = None
                except PersistenceError as exc:
                    self._store_open_error = str(exc)
            report = self._evaluate_health()
            if report.overall is ComponentStatus.FAULT:
                raise SessionError("cannot reset: critical components still FAULT: " + ", ".join(
                    c.name for c in report.components if c.critical and c.status is not ComponentStatus.OK))
            try:
                self.store.audit(actor_ref=actor_ref.strip(), action="FAULT_RESET", target="FAULT->IDLE",
                                 detail={"reason": reason.strip(), "latch": self._fault_latch},
                                 correlation_id=self.session_id)
            except PersistenceError as exc:
                raise SessionError(f"cannot reset: audit write failed ({exc})") from exc
            try:
                self._latch_path.unlink(missing_ok=True)
            except OSError as exc:
                raise SessionError(f"cannot reset: fault latch not removable ({type(exc).__name__})") from exc
            self._fault_latch = None
            self.state.release_fault(f"FAULT_RESET: {reason.strip()}")
            self.session_id = None

    def close_session(self) -> None:
        """Ends the session. In FAULT the session is discarded but FAULT is kept (R4-01)."""
        with self._process_lock:
            if self.session_id is None:
                raise SessionError("no active session")
            if self.state.state is SystemState.FAULT:
                closed = self.session_id
                self.session_id = None
                log.warning("session closed while in FAULT; fault remains latched",
                            extra={"code": "SESSION_CLOSED_IN_FAULT", "ctx": {"session_id": closed}})
                if not self.read_only and self.store.is_open:
                    try:
                        self.store.audit(actor_ref="system", action="SESSION_CLOSED_IN_FAULT", target="FAULT",
                                         detail={"reason": "close_session while FAULT"}, correlation_id=closed)
                    except PersistenceError:
                        log.error("audit write failed", extra={"code": "AUDIT_WRITE_FAILED"})
                return
            self.state.transition(SystemState.IDLE, "SESSION_CLOSED")
            self.session_id = None

    # -- processing ----------------------------------------------------------
    def _safe_infer(self, request: InferenceRequest) -> InferenceResult:
        def failure(status: InferenceStatus, code: str) -> InferenceResult:
            try:
                runtime = RuntimeRef(name=str(self.inference.adapter_id), version=str(self.inference.adapter_version))
            except Exception:
                runtime = RuntimeRef(name="unknown-adapter", version="unknown")
            return InferenceResult(
                inspection_id=request.inspection_id,
                runtime=runtime,
                input=request.input,
                status=status,
                error_code=code,
            )

        try:
            result = self.inference.infer(request)
            expected_model = self.inference.model
        except Exception as exc:
            log.error("inference adapter raised", extra={"code": "INFERENCE_EXCEPTION", "ctx": {"error": type(exc).__name__}})
            return failure(InferenceStatus.RUNTIME_ERROR, f"ADAPTER_EXCEPTION:{type(exc).__name__}")
        if not isinstance(result, InferenceResult):
            return failure(InferenceStatus.INCOMPATIBLE, "ADAPTER_INVALID_RETURN_TYPE")
        if result.inspection_id != request.inspection_id:
            return failure(InferenceStatus.INCOMPATIBLE, "INSPECTION_ID_MISMATCH")
        if result.model != expected_model:
            return failure(InferenceStatus.INCOMPATIBLE, "MODEL_IDENTITY_MISMATCH")
        if result.input != request.input:  # adapter must not upgrade the input quality flag
            return failure(InferenceStatus.INCOMPATIBLE, "INPUT_DESCRIPTOR_MISMATCH")
        if result.model is not None and not self.model_policy.is_model_approved(result.model):
            return failure(InferenceStatus.INCOMPATIBLE, "MODEL_NOT_APPROVED")  # ADR-016 default-deny
        return result

    def process(self, request: InferenceRequest, *, captured_at_utc: str | None = None) -> ProcessOutcome:
        # R4-02: caller errors are rejected before any processing (no inference, no event, no result).
        if not isinstance(request, InferenceRequest):
            raise ContractError(f"request must be InferenceRequest, got {type(request).__name__}")
        request.validate()
        if captured_at_utc is not None:
            validate_utc(captured_at_utc)
        with self._process_lock:
            session_id = self.session_id
            if session_id is None:
                raise SessionError("no active session")
            if self.state.state not in (SystemState.OBSERVING, SystemState.DEGRADED):
                raise SessionError(f"cannot process in state {self.state.state}")
            captured = utc_now_iso() if captured_at_utc is None else captured_at_utc
            with bind_context(session_id=session_id, correlation_id=request.inspection_id):
                # Idempotency per (session, inspection_id): a retry returns the stored outcome.
                try:
                    existing = self.store.get_by_inspection(session_id, request.inspection_id)
                except CorruptRecordError as exc:
                    # R4-03: explicit handling — counted, logged, FAULT; never re-inferred or trusted.
                    self.events.note_corrupt_record(session_id=session_id, inspection_id=request.inspection_id,
                                                    error=str(exc))
                    return ProcessOutcome(record=RecordResult(None, False, None, str(exc)),
                                          result=InspectionResult.NO_DECISION.value, reason="EVENT_RECORD_CORRUPT")
                except PersistenceError:
                    existing = None  # the append below will surface the storage failure
                if existing is not None:
                    self.events.note_inspection_duplicate(session_id=session_id, inspection_id=request.inspection_id,
                                                          event_id=existing.event_id, result=existing.result.value)
                    return ProcessOutcome(
                        record=RecordResult(existing.event_id, True, AppendOutcome.DUPLICATE),
                        result=existing.result.value, reason=existing.reason,
                    )
                inference = self._safe_infer(request)
                # Health is evaluated for EVERY inspection, after inference, under the health lock:
                # no cached report can authorize a conclusive result.
                health_ok = self.refresh_health().ok
                decision = self.decision.decide(inference, DecisionContext(self.state.state, health_ok))
                build = dict(inference=inference, session_id=session_id, captured_at_utc=captured,
                             capture_sequence=self._sequence + 1)
                try:
                    envelope = self.events.build(decision=decision, **build)
                except Exception as exc:
                    # Untraceable/invalid decision or payload: record NO_DECISION instead of losing it.
                    log.error("decision not representable; recording NO_DECISION",
                              extra={"code": "DECISION_CONTRACT_ERROR", "ctx": {"error": type(exc).__name__}})
                    decision = Decision(InspectionResult.NO_DECISION, DecisionReason.RULE_ERROR,
                                        detail="decision violated event contract")
                    envelope = self.events.build(decision=decision, **build)
                self._sequence += 1
                record = self.events.record(envelope)
                if not record.persisted:
                    # An untraceable result must never be presented as conclusive.
                    return ProcessOutcome(record=record, result=InspectionResult.NO_DECISION.value,
                                          reason="EVENT_NOT_PERSISTED")
                return ProcessOutcome(record=record, result=decision.result.value, reason=decision.reason.value)

    # -- status --------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        model = self.inference.model
        return {
            "software": {"name": SOFTWARE_NAME, "version": __version__},
            "configuration": self.config.reference(),
            "state": self.state.state.value,
            "fault_latch": dict(self._fault_latch) if self._fault_latch else None,
            "session_id": self.session_id,
            "inference": {
                "adapter": self.inference.adapter_id,
                "adapter_version": self.inference.adapter_version,
                "model": model.to_dict() if model else None,
            },
            "model_policy": self.model_policy.to_dict(),
            "health": self._last_health.to_dict() if self._last_health else None,
            "events": self.events.counters().to_dict(),
            "rejected_transitions": self.state.rejected_count,
            "host_memory": host_memory(),
            "mode": "OBSERVATION_ONLY",
        }

    # -- audit ---------------------------------------------------------------
    def _audit_transition(self, rec: TransitionRecord) -> None:
        if rec.accepted and rec.target is SystemState.FAULT:
            if self._fault_latch is None:
                self._fault_latch = {"reason": rec.reason, "at_utc": rec.at_utc, "source": "runtime"}
            self._write_latch(rec.reason, rec.at_utc)
        if self.read_only or not self.store.is_open:
            return
        try:
            self.store.audit(
                actor_ref="system",
                action="STATE_TRANSITION" if rec.accepted else "STATE_TRANSITION_REJECTED",
                target=f"{rec.source}->{rec.target}",
                detail={"reason": rec.reason, "at_utc": rec.at_utc, "seq": rec.seq},
                correlation_id=self.session_id or self._pending_session_id,
            )
        except PersistenceError:
            log.error("audit write failed", extra={"code": "AUDIT_WRITE_FAILED"})
