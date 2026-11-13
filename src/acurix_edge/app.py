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
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from acurix_edge import SOFTWARE_NAME, __version__

if TYPE_CHECKING:  # these import OpenCV; the app loads them only when capture/vision is used
    from acurix_edge.capture.service import CaptureService
    from acurix_edge.vision.quality_gate import QualityGate
from acurix_edge.adapters.inference_noop import NoOpInferenceAdapter
from acurix_edge.capture.errors import CaptureError
from acurix_edge.config.settings import AppConfig, envelope_fingerprint
from acurix_edge.contracts.capture import CapturedFrame, CaptureStatus
from acurix_edge.contracts.decision import Decision, DecisionReason, InspectionResult
from acurix_edge.contracts.errors import ContractError, ErrorCode
from acurix_edge.contracts.events import EventEnvelope, utc_now_iso, validate_utc
from acurix_edge.contracts.inference import (
    ImageQuality,
    InferenceAdapter,
    InferenceRequest,
    InferenceResult,
    InferenceStatus,
    InputDescriptor,
    RuntimeRef,
)
from acurix_edge.contracts.quality import Comparison, QualityReason, QualityResult, QualityStatus
from acurix_edge.domain.artifacts import ArtifactRejected, ArtifactVerifier
from acurix_edge.domain.decision_engine import DecisionContext, DecisionEngine, DecisionRule, is_rule_approved
from acurix_edge.domain.model_policy import ModelPolicy
from acurix_edge.domain.stages import InspectionStage, UnavailableStage, guarded_run
from acurix_edge.domain.state_machine import (
    StateMachine,
    SystemState,
    TransitionRecord,
    rejection_code,
)
from acurix_edge.events.service import EventService, RecordResult
from acurix_edge.health.checks import ClockCheck, CpuCheck, GpuCheck, QueueGauge, memory_check, storage_check
from acurix_edge.health.service import (
    RESOURCES,
    Check,
    CheckResult,
    ComponentStatus,
    HealthReport,
    HealthService,
    host_memory,
)
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
QUALITY_RUNTIME = RuntimeRef(name="quality-gate", version="1", device="cpu")
ENVELOPE_ACTIONS = ("QUALITY_ENVELOPE_ACTIVATED", "QUALITY_ENVELOPE_CHANGED")
GATE_FAILURE_LIMIT = 3  # consecutive quality-gate exceptions before health reports DEGRADED
CaptureProbe = Callable[[], tuple[ComponentStatus, str]]


class SessionError(RuntimeError):
    """Operation refused by session/state rules. `code` is the AEX catalog code (Anexo 02A §3)."""

    def __init__(self, message: str, code: ErrorCode = ErrorCode.STATE_001) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True, slots=True)
class ProcessOutcome:
    record: RecordResult
    result: str
    reason: str
    code: ErrorCode | None = None  # AEX code whenever the outcome was forced to NO_DECISION by a failure


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
        host_checks: Mapping[str, Check] | None = None,
        capture: CaptureService | None = None,
    ) -> None:
        if capture is not None and capture_probe is not None:
            raise ValueError("use either capture or capture_probe, not both")
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
        self.queue = QueueGauge()
        self._quality_gate: QualityGate | None = None
        self._envelope_conflict: str | None = None
        self._gate_failures = 0  # consecutive quality-gate exceptions (review L-d)
        # CAS-005: inventory-backed verification; approval only from the policy (never the inventory).
        self.artifacts = ArtifactVerifier(config.models.artifacts, self.model_policy,
                                          on_incident=self._on_artifact_incident)
        self.capture = capture
        if capture is not None:
            capture.on_lost = self._on_capture_lost
            capture.on_drop_alarm = self._on_capture_drop_alarm
        self.state.subscribe(self._audit_transition)
        self._register_checks(host_checks)

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
        self._check_quality_envelope()
        self._evaluate_health()

    def shutdown(self) -> None:
        """Ends any session and closes storage. Never leaves FAULT (the latch stays persisted)."""
        with self._process_lock:
            if self.session_id is not None:
                self.close_session()
            if self.capture is not None:
                self.capture.shutdown()
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
    def _register_checks(self, host_checks: Mapping[str, Check] | None) -> None:
        def database() -> CheckResult:
            if not self.store.is_open:
                return ComponentStatus.FAULT, self._store_open_error or "database not open", ErrorCode.DB_001
            ok, detail = self.store.check()
            return (ComponentStatus.OK, detail) if ok else (ComponentStatus.FAULT, detail, ErrorCode.DB_001)

        def models() -> CheckResult:
            m = self.inference.model
            if not self.inference.ready or m is None:
                approved = len(self.model_policy.approved_models)
                return (ComponentStatus.DEGRADED,
                        f"adapter '{self.inference.adapter_id}': no approved model loaded "
                        f"(approved_models: {approved}, ADR-016 default-deny)")
            if not self.model_policy.is_model_approved(m):
                return (ComponentStatus.DEGRADED, f"model {m.id}@{m.version} not in approved_models (default-deny)",
                        ErrorCode.CFG_002)
            return ComponentStatus.OK, f"{m.id}@{m.version}"

        def decision_rule() -> CheckResult:
            rule = self.decision.rule
            if is_rule_approved(rule):
                return ComponentStatus.OK, f"{rule.rule_id}@{rule.rule_version}"
            return ComponentStatus.DEGRADED, "no approved decision rule (use case NOT_FROZEN)"

        def camera() -> CheckResult:
            if self.capture is not None:
                return self.capture.health()
            if self._capture_probe is None:
                return (ComponentStatus.UNKNOWN, "no capture source in 0.1.0 (absent; real camera not opened)",
                        ErrorCode.CAP_001)
            return self._capture_probe()

        host: dict[str, Check] = {"clock": ClockCheck(backward_tolerance_s=self.config.clock.backward_tolerance_s,
                                                      last_event_utc=self.store.last_event_persisted_utc),
                                  "memory": memory_check(), "cpu": CpuCheck(), "gpu": GpuCheck()}
        unknown = set(host_checks or {}) - set(host)
        if unknown:
            raise ValueError(f"unknown host checks {sorted(unknown)}")
        host.update(host_checks or {})
        # components: OpenAPI ComponentHealth names only
        self.health.register("camera", camera, critical=False)
        self.health.register("models", models, critical=False)
        self.health.register("database", database, critical=True)
        self.health.register(
            "storage",
            storage_check(self.config.database_path(self.base_dir).parent, self.config.storage.min_free_disk_mb),
            critical=True,
        )
        # Critical (owner decision 10/11/2026): clock earlier than the last persisted event -> FAULT
        # latch; every other clock problem is reported DEGRADED by the check and blocks OBSERVING.
        self.health.register("clock", host["clock"], critical=True)
        # resources: outside the OpenAPI enum (owner decision 10/11/2026), still part of `overall`
        self.health.register("memory", host["memory"], critical=False, group=RESOURCES)
        self.health.register("cpu", host["cpu"], critical=False, group=RESOURCES)
        self.health.register("gpu", host["gpu"], critical=False, group=RESOURCES)
        self.health.register("queue", self._queue_check, critical=False, group=RESOURCES)
        self.health.register("decision_rule", decision_rule, critical=False, group=RESOURCES)
        self.health.register("quality_envelope", self._envelope_check, critical=False, group=RESOURCES)

    def _queue_check(self) -> CheckResult:
        """`resources.queue` (Ordem 2 block): inspection gauge + capture buffer counters (CAP-003)."""
        if self.capture is None:
            return self.queue.check()
        c = self.capture.counters(active_only=True)
        causes =", ".join(f"{k}={v}" for k, v in sorted(c.dropped_by_cause.items())) or "none"
        detail = (f"capture: captured {c.captured}, delivered {c.delivered}, dropped {c.dropped} ({causes}), "
                  f"depth {c.depth}/{c.capacity}, max {c.max_depth}, alarm above {c.drop_alarm_threshold}; "
                  f"inspections in flight {self.queue.depth}")
        if c.over_limit:
            return ComponentStatus.DEGRADED, detail, ErrorCode.CAP_002
        return ComponentStatus.OK, detail

    # -- capture events (called from capture threads) ------------------------------------
    def _on_capture_lost(self, detail: str, session_id: str | None) -> None:
        """HLT-004: camera lost during a session -> FAULT (latched) + AEX-CAP-001 in the audit trail.
        Runs on a capture thread; the state machine and the store are thread-safe."""
        if session_id is None:
            return  # not bound to a session: health shows FAULT for the camera, no session to fail
        self.state.enter_fault(f"CAMERA_LOST: {ErrorCode.CAP_001.value}")
        if not self.read_only and self.store.is_open:
            try:
                self.store.audit(actor_ref="system", action="CAMERA_LOST", target="camera",
                                 detail={"code": ErrorCode.CAP_001.value, "detail": detail,
                                         "source_id": self.capture.source.source_id if self.capture else None},
                                 correlation_id=session_id)
            except PersistenceError:
                log.error("audit write failed", extra={"code": "AUDIT_WRITE_FAILED"})

    def _on_capture_drop_alarm(self, counters: Any, session_id: str | None) -> None:
        """CAP-003: drops above the configured limit -> AEX-CAP-002 in the audit trail (once per session)."""
        if self.read_only or not self.store.is_open:
            return
        try:
            self.store.audit(actor_ref="system", action="CAPTURE_DROP_LIMIT_EXCEEDED", target="capture_queue",
                             detail={"code": ErrorCode.CAP_002.value, "counters": counters.to_dict()},
                             correlation_id=session_id)
        except PersistenceError:
            log.error("audit write failed", extra={"code": "AUDIT_WRITE_FAILED"})

    def next_frame(self, timeout: float | None = None) -> CapturedFrame | None:
        """Next captured frame of the ACTIVE session (None on timeout). Raises SessionError when
        there is no capture, no session, or FAULT; capture errors propagate (EndOfStream, CaptureLost)."""
        if self.capture is None:
            raise SessionError("no capture source configured", ErrorCode.CAP_001)
        with self._process_lock:
            if self.state.state is SystemState.FAULT:
                raise SessionError("capture blocked: FAULT latched", ErrorCode.STATE_002)
            session_id = self.session_id
            if session_id is None:
                raise SessionError("no active session", ErrorCode.SES_003)
        return self.capture.next_frame(session_id, timeout)

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
        if self.session_id is not None:
            raise SessionError("a session is already active", ErrorCode.SES_001)
        if self.state.state is not SystemState.IDLE:
            raise SessionError(f"cannot start session in state {self.state.state}", rejection_code(self.state.state))
        session_id = str(uuid.uuid4())
        self._pending_session_id = session_id  # audits of the start transitions carry this id (R4-04)
        try:
            self.state.transition(SystemState.INITIALIZING, "SESSION_START_REQUESTED")
            try:
                self._verify_camera(session_id)
                report = self._evaluate_health()
                if report.overall is ComponentStatus.FAULT:
                    self.state.enter_fault("INIT_PRECONDITION_FAULT")
                    raise SessionError("pre-conditions failed: " + ", ".join(
                        c.name for c in report.all_entries if c.status is not ComponentStatus.OK),
                        ErrorCode.SES_002)
                target = SystemState.OBSERVING if report.ok else SystemState.DEGRADED
                self.state.transition(target, "INIT_COMPLETE" if report.ok else "INIT_DEGRADED")
                if self.capture is not None:
                    self.capture.begin(session_id)  # frames are bound to THIS session only (CAP-004)
            except BaseException as exc:
                try:
                    if self.capture is not None:
                        self.capture.end()
                finally:
                    self._abort_start(session_id, exc)  # runs even if end() raises (review L5)
                raise
            self._sequence = 0
            self.session_id = session_id  # only after a successful transition
        finally:
            self._pending_session_id = None
        return session_id

    def _abort_start(self, session_id: str, exc: BaseException) -> None:
        """A failed start must not leave the system in INITIALIZING or in OBSERVING/DEGRADED without
        a session (review M1/M4/L5): back to IDLE, audited. Decided by the CURRENT state, not by the
        exception type: handled paths already left INITIALIZING (FAULT keeps its latch, IDLE is final)."""
        current = self.state.state
        if current in (SystemState.INITIALIZING, SystemState.OBSERVING, SystemState.DEGRADED):
            if self.store.is_open:
                try:
                    self.store.audit(actor_ref="system", action="SESSION_START_ABORTED", target=str(current),
                                     detail={"error": type(exc).__name__}, correlation_id=session_id)
                except PersistenceError:
                    log.error("audit write failed", extra={"code": "AUDIT_WRITE_FAILED"})
            self.state.transition(SystemState.IDLE, f"SESSION_START_ABORTED: {type(exc).__name__}")

    def _verify_camera(self, session_id: str) -> None:
        """HLT-002: with a capture source, the camera must open and deliver a valid frame before
        OBSERVING. Failure: back to IDLE (not FAULT: nothing was observed), audited with
        AEX-CAP-001, SessionError(AEX-CAP-001). Called in INITIALIZING."""
        if self.capture is None:
            return
        try:
            self.capture.verify()
        except CaptureError as exc:
            if self.store.is_open:
                try:
                    self.store.audit(actor_ref="system", action="CAMERA_UNAVAILABLE", target="camera",
                                     detail={"code": ErrorCode.CAP_001.value, "detail": str(exc)},
                                     correlation_id=session_id)
                except PersistenceError:
                    log.error("audit write failed", extra={"code": "AUDIT_WRITE_FAILED"})
            self.state.transition(SystemState.IDLE, f"CAMERA_UNAVAILABLE: {ErrorCode.CAP_001.value}")
            raise SessionError(f"camera not verified: {exc}", ErrorCode.CAP_001) from exc

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
                raise SessionError(f"reset only allowed from FAULT (current {self.state.state})", ErrorCode.STATE_001)
            if not self.store.is_open:
                try:
                    self.store.open()
                    self._store_open_error = None
                except PersistenceError as exc:
                    self._store_open_error = str(exc)
            self._check_quality_envelope()  # review M1-b: the store may have just been reopened
            report = self._evaluate_health()
            if report.overall is ComponentStatus.FAULT:
                raise SessionError("cannot reset: critical components still FAULT: " + ", ".join(
                    c.name for c in report.all_entries if c.critical and c.status is not ComponentStatus.OK),
                    ErrorCode.STATE_002)
            try:
                self.store.audit(actor_ref=actor_ref.strip(), action="FAULT_RESET", target="FAULT->IDLE",
                                 detail={"reason": reason.strip(), "latch": self._fault_latch},
                                 correlation_id=self.session_id)
            except PersistenceError as exc:
                raise SessionError(f"cannot reset: audit write failed ({exc})", exc.code) from exc
            try:
                self._latch_path.unlink(missing_ok=True)
            except OSError as exc:
                raise SessionError(f"cannot reset: fault latch not removable ({type(exc).__name__})",
                                   ErrorCode.SYS_001) from exc
            self._fault_latch = None
            self.state.release_fault(f"FAULT_RESET: {reason.strip()}")
            if self.capture is not None and self.session_id is not None:
                self.capture.end(self.session_id)  # the old session is discarded, never resumed (CAP-004)
            self.session_id = None

    def close_session(self) -> None:
        """Ends the session. In FAULT the session is discarded but FAULT is kept (R4-01)."""
        with self._process_lock:
            if self.session_id is None:
                raise SessionError("no active session", ErrorCode.SES_003)
            if self.capture is not None:
                self.capture.end(self.session_id)
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

    # -- maintenance (Manual Mestre §8: blocks normal operation; changes are recorded) ---------
    def enter_maintenance(self, reason: str, *, actor_ref: str) -> None:
        self._maintenance(SystemState.MAINTENANCE, "MAINTENANCE_ENTER", reason, actor_ref)

    def exit_maintenance(self, reason: str, *, actor_ref: str) -> None:
        self._maintenance(SystemState.IDLE, "MAINTENANCE_EXIT", reason, actor_ref)

    def _maintenance(self, target: SystemState, action: str, reason: str, actor_ref: str) -> None:
        for name, value in (("reason", reason), ("actor_ref", actor_ref)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} is mandatory")
        if self.read_only:
            raise SessionError("application opened read-only")
        with self._process_lock, self._health_lock:
            source = self.state.state
            expected = SystemState.IDLE if target is SystemState.MAINTENANCE else SystemState.MAINTENANCE
            if source is not expected:
                # Refused WITHOUT attempting the transition (e.g. OBSERVING->IDLE is a valid table
                # entry but not a maintenance exit). The refusal is still audited with its code.
                code = rejection_code(source)
                if self.store.is_open:
                    with contextlib.suppress(PersistenceError):
                        self.store.audit(actor_ref=actor_ref.strip(), action="STATE_TRANSITION_REJECTED",
                                         target=f"{source}->{target}",
                                         detail={"reason": f"{action}: {reason.strip()}", "code": code.value})
                raise SessionError(f"{action} not allowed from {source}", code)
            # The actor audit is written BEFORE the transition: no unattributed maintenance.
            try:
                self.store.audit(actor_ref=actor_ref.strip(), action=action, target=f"{source}->{target}",
                                 detail={"reason": reason.strip()})
            except PersistenceError as exc:
                raise SessionError(f"{action} refused: audit write failed ({exc})", ErrorCode.DB_001) from exc
            self.state.transition(target, f"{action}: {reason.strip()}")

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
            # typed as object: an adapter may violate its signature; that is checked below at runtime
            result: object = self.inference.infer(request)
            expected_model = self.inference.model
        except Exception as exc:
            log.error("inference adapter raised",
                      extra={"code": "INFERENCE_EXCEPTION", "ctx": {"error": type(exc).__name__}})
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
        """LEGACY entry point (0.1.0): the caller declares the input quality and NO quality gate runs.
        Captured frames MUST go through `process_frame()` (QG-001: every frame is gated). Not called
        by product code since 0.3.0; kept for the existing contract tests (review L6).
        Queue gauge wraps the serialized processing: depth = inspections waiting or in flight."""
        self.queue.enter()
        try:
            return self._process(request, captured_at_utc=captured_at_utc)
        finally:
            self.queue.leave()

    # -- quality gate + stages (QG-001..003, CAS-005, AEX-CV-007) -------------------------
    @property
    def quality_gate(self) -> QualityGate:
        """Created on first use (imports OpenCV only when frames are actually inspected)."""
        if self._quality_gate is None:
            from acurix_edge.vision.quality_gate import QualityGate as _Gate

            self._quality_gate = _Gate(self.config.quality)
        return self._quality_gate

    def process_frame(self, frame: CapturedFrame, *, stage: InspectionStage | None = None) -> ProcessOutcome:
        """Captured frame -> quality gate (EVERY frame) -> stage (only if VALID) -> decision -> event.
        INPUT_INVALID -> NO_DECISION with the reasons recorded in the event; no stage is called."""
        if not isinstance(frame, CapturedFrame) or frame.status is not CaptureStatus.OK or frame.pixels is None:
            raise ContractError("process_frame requires an OK CapturedFrame with pixels")
        if self._envelope_conflict is not None:
            raise SessionError(f"quality envelope refused: {self._envelope_conflict}", ErrorCode.CFG_001)
        try:
            quality = self.quality_gate.evaluate(frame.pixels)
            self._gate_failures = 0
        except Exception as exc:  # review L5: a gate failure is a recorded INPUT_INVALID, never silent
            self._gate_failures += 1  # consecutive failures are visible in health (quality_envelope)
            log.error("quality gate failed", extra={"code": "QUALITY_GATE_ERROR",
                                                    "ctx": {"error": type(exc).__name__,
                                                            "consecutive": self._gate_failures}})
            q = self.config.quality
            quality = QualityResult(
                status=QualityStatus.INPUT_INVALID,
                reasons=(QualityReason("QUALITY_GATE_ERROR", "gate_error", 1.0, 0.0, Comparison.ABOVE_MAX),),
                metrics={"gate_error": 1.0}, elapsed_ms=0.0, envelope_version=q.version, envelope_status=q.status,
                envelope_validated=False, roi_px=(0, 0, 0, 0))
        request = InferenceRequest(
            inspection_id=frame.frame_id,
            input=InputDescriptor(frame.input.width, frame.input.height,
                                  ImageQuality.VALID if quality.is_valid else ImageQuality.INVALID),
            frame=frame.pixels)
        self.queue.enter()
        try:
            return self._process(request, captured_at_utc=frame.captured_at_utc, quality=quality, stage=stage)
        finally:
            self.queue.leave()

    def _shadow_eligible(self, stage: InspectionStage) -> bool:
        """Owner exception to ADR-017 item 6, condition (a): an UNAPPROVED stage may be called (shadow)
        only if its exact artifact (id, version, sha256) is in the separate inventory as NOT_APPROVED."""
        try:
            model = stage.model
        except Exception:
            return False
        if model is None:
            return False
        entry = self.artifacts.entry(model.id, model.version)
        return entry is not None and entry.ref == model and entry.status == "NOT_APPROVED"

    def load_label_skew_stage(self) -> InspectionStage:
        """CAS-005: the specialist's parameters are used ONLY after id/version/sha256 verification
        against the inventory. Any refusal -> incident + an unavailable stage (NO_DECISION)."""
        from acurix_edge.vision.label_skew import ARTIFACT_ID, STAGE_KIND, LabelSkewSpecialist, ParamsError

        version = next((str(a["version"]) for a in self.config.models.artifacts if a.get("id") == ARTIFACT_ID), "")
        stage_id = f"{STAGE_KIND}:{ARTIFACT_ID}"
        try:
            artifact = self.artifacts.load(ARTIFACT_ID, version)
            return LabelSkewSpecialist(artifact)
        except ArtifactRejected as exc:
            return UnavailableStage(stage_id, STAGE_KIND, f"{exc.reason.value}:{exc.code.value}")
        except ParamsError as exc:
            self._audit_incident("ARTIFACT_REJECTED", f"{ARTIFACT_ID}@{version}",
                                 {"code": ErrorCode.CAS_002.value, "reason": "PARAMS_INVALID", "detail": str(exc)})
            return UnavailableStage(stage_id, STAGE_KIND, f"PARAMS_INVALID:{ErrorCode.CAS_002.value}")

    def _audit_incident(self, action: str, target: str, detail: dict[str, Any]) -> bool:
        """Write an audit entry; returns False when it could not be written."""
        if self.read_only or not self.store.is_open:
            return False
        try:
            self.store.audit(actor_ref="system", action=action, target=target, detail=detail,
                             correlation_id=self.session_id)
            return True
        except PersistenceError:
            log.error("audit write failed", extra={"code": "AUDIT_WRITE_FAILED"})
            return False

    def _on_artifact_incident(self, exc: ArtifactRejected) -> None:
        self._audit_incident("ARTIFACT_REJECTED", f"{exc.artifact_id}@{exc.version}",
                             {"code": exc.code.value, "reason": exc.reason.value, "detail": str(exc)})

    def _envelope_fingerprint(self) -> tuple[str, str]:
        return envelope_fingerprint(self.config.quality)

    def _check_quality_envelope(self) -> None:
        """QG-003: the envelope in use is recorded in the audit trail. A new version is audited
        as a change; the SAME version with different limits is refused (a limit change must
        bump the version) — the gate stays unavailable and health reports it."""
        self._envelope_conflict = None
        if self.read_only:
            return
        if not self.store.is_open:  # review M1-b: never use an envelope that was not audited
            self._envelope_conflict = "envelope not audited (store unavailable); re-checked on reset_fault"
            return
        version, sha = self._envelope_fingerprint()
        target = f"quality@{version}"
        detail = {"version": version, "sha256": sha, "status": self.config.quality.status, "validated": False}
        try:
            # review M1: the SAME version is checked against its own history, not only the latest record
            same_version = self.store.last_audit_with_action(ENVELOPE_ACTIONS, target=target)
            latest = self.store.last_audit_with_action(ENVELOPE_ACTIONS)
        except PersistenceError as exc:  # review L1: never use an envelope that could not be audited
            self._envelope_conflict = f"envelope history unreadable ({exc.code.value})"
            return

        def summary(rec: dict[str, Any] | None) -> dict[str, Any] | None:  # review L2: bounded detail
            d = rec.get("detail") if rec else None
            return {"version": d.get("version"), "sha256": d.get("sha256")} if isinstance(d, dict) else None

        def corrupted(rec: dict[str, Any] | None) -> bool:
            d = rec.get("detail") if rec else None
            return rec is not None and not (isinstance(d, dict) and isinstance(d.get("sha256"), str)
                                            and len(d["sha256"]) == 64)

        if corrupted(same_version):  # review L-c: a damaged record is not "different limits"
            self._envelope_conflict = (f"envelope history for version {version} is corrupted (record unreadable "
                                       "or without sha256); the version cannot be verified")
            self._audit_incident("QUALITY_ENVELOPE_REJECTED", target,
                                 {**detail, "code": ErrorCode.CFG_001.value, "reason": "HISTORY_CORRUPTED"})
            return
        mine, prev = summary(same_version), summary(latest)
        if mine is not None and mine.get("sha256") != sha:
            self._envelope_conflict = (f"version {version} already recorded with different limits "
                                       f"(sha {str(mine.get('sha256'))[:12]}...); a limit change needs a new version")
            self._audit_incident("QUALITY_ENVELOPE_REJECTED", target,
                                 {**detail, "code": ErrorCode.CFG_001.value, "recorded": mine})
            return
        if prev is not None and prev.get("sha256") == sha:
            return  # already the active envelope
        action = "QUALITY_ENVELOPE_ACTIVATED" if prev is None else "QUALITY_ENVELOPE_CHANGED"
        if not self._audit_incident(action, target, {**detail, "previous": prev}):
            self._envelope_conflict = "envelope activation could not be audited"  # review L1

    def _envelope_check(self) -> CheckResult:
        q = self.config.quality
        if self._envelope_conflict is not None:
            return ComponentStatus.DEGRADED, f"quality envelope refused: {self._envelope_conflict}", ErrorCode.CFG_001
        if self._gate_failures >= GATE_FAILURE_LIMIT:  # review L-d: a broken gate is not silent in health
            return (ComponentStatus.DEGRADED, f"quality gate failing: {self._gate_failures} consecutive errors "
                    "(every frame recorded as INPUT_INVALID/QUALITY_GATE_ERROR)", ErrorCode.SYS_001)
        return ComponentStatus.OK, (f"quality envelope {q.version} status={q.status.upper()} "
                                    "(NOT validated: calibrate in the optical test, Anexo 04A 3.2)")

    def _process(self, request: InferenceRequest, *, captured_at_utc: str | None = None,
                 quality: QualityResult | None = None, stage: InspectionStage | None = None) -> ProcessOutcome:
        # R4-02: caller errors are rejected before any processing (no inference, no event, no result).
        if not isinstance(request, InferenceRequest):
            raise ContractError(f"request must be InferenceRequest, got {type(request).__name__}")
        request.validate()
        if captured_at_utc is not None:
            validate_utc(captured_at_utc)
        with self._process_lock:
            session_id = self.session_id
            current = self.state.state
            if current is SystemState.FAULT:
                raise SessionError("processing blocked: FAULT latched", ErrorCode.STATE_002)
            if session_id is None:
                raise SessionError("no active session", ErrorCode.SES_003)
            if current not in (SystemState.OBSERVING, SystemState.DEGRADED):
                raise SessionError(f"cannot process in state {current}", ErrorCode.STATE_001)
            captured = utc_now_iso() if captured_at_utc is None else captured_at_utc
            with bind_context(session_id=session_id, correlation_id=request.inspection_id):
                # Idempotency per (session, inspection_id): a retry returns the stored outcome.
                try:
                    existing = self.store.get_by_inspection(session_id, request.inspection_id)
                except CorruptRecordError as exc:
                    # R4-03: explicit handling — counted, logged, FAULT; never re-inferred or trusted.
                    self.events.note_corrupt_record(session_id=session_id, inspection_id=request.inspection_id,
                                                    error=str(exc))
                    return ProcessOutcome(record=RecordResult(None, False, None, str(exc), ErrorCode.DB_001),
                                          result=InspectionResult.NO_DECISION.value, reason="EVENT_RECORD_CORRUPT",
                                          code=ErrorCode.DB_001)
                except PersistenceError:
                    existing = None  # the append below will surface the storage failure
                request_sha256 = request.sha256()
                if existing is not None:
                    if existing.request_sha256 is not None and existing.request_sha256 != request_sha256:
                        # EVT-001: same inspection_id, different content -> recorded conflict, nothing
                        # re-inferred or overwritten, never a conclusive answer.
                        self.events.note_inspection_conflict(session_id=session_id,
                                                             inspection_id=request.inspection_id,
                                                             event_id=existing.event_id)
                        return ProcessOutcome(
                            record=RecordResult(existing.event_id, False, None,
                                                "inspection_id already recorded with different content",
                                                ErrorCode.API_004),
                            result=InspectionResult.NO_DECISION.value, reason="IDEMPOTENCY_CONFLICT",
                            code=ErrorCode.API_004)
                    self.events.note_inspection_duplicate(session_id=session_id, inspection_id=request.inspection_id,
                                                          event_id=existing.event_id, result=existing.result.value)
                    return ProcessOutcome(
                        record=RecordResult(existing.event_id, True, AppendOutcome.DUPLICATE),
                        result=existing.result.value, reason=existing.reason,
                    )
                started = time.perf_counter()
                extra: dict[str, Any] = {}
                forced: Decision | None = None
                if quality is not None:
                    extra["quality"] = quality.to_dict()
                if quality is not None and not quality.is_valid:
                    # QG-002: invalid image -> INPUT_INVALID + NO_DECISION; NO stage/model is called.
                    codes = ",".join(r.code for r in quality.reasons)
                    inference = InferenceResult(inspection_id=request.inspection_id, runtime=QUALITY_RUNTIME,
                                                input=request.input, status=InferenceStatus.INPUT_INVALID,
                                                error_code=f"QUALITY_GATE:{codes}")
                    forced = Decision(InspectionResult.NO_DECISION, DecisionReason.INPUT_QUALITY,
                                      detail=f"INPUT_INVALID: {codes}"[:200])
                elif stage is not None:
                    run = guarded_run(stage, request, may_run_unapproved=self._shadow_eligible(stage))
                    extra["stages"] = [run.to_dict()]
                    inference = run.for_decision
                    if inference.model is not None and not self.model_policy.is_model_approved(inference.model):
                        # defense in depth: the application policy has the last word (ADR-016)
                        inference = InferenceResult(inspection_id=request.inspection_id, runtime=inference.runtime,
                                                    input=request.input, status=InferenceStatus.INCOMPATIBLE,
                                                    error_code="MODEL_NOT_APPROVED")
                else:
                    inference = self._safe_infer(request)
                inference_elapsed_ms = (time.perf_counter() - started) * 1000.0
                # Health is evaluated for EVERY inspection, after inference, under the health lock:
                # no cached report can authorize a conclusive result. Its duration is NOT part of
                # latency_ms (contract: inference + decision; review R1-M2).
                health_ok = self.refresh_health().ok
                decision = forced or self.decision.decide(inference, DecisionContext(self.state.state, health_ok))
                if (quality is not None and not quality.envelope_validated
                        and decision.result in (InspectionResult.PASS, InspectionResult.DEFECT)):
                    # A provisional envelope never certifies an image for a conclusive result (QG-003).
                    decision = Decision(InspectionResult.NO_DECISION, DecisionReason.INPUT_QUALITY,
                                        detail="quality envelope provisional (not validated)",
                                        latency_ms=decision.latency_ms)
                latency_ms = inference_elapsed_ms + (decision.latency_ms or 0.0)
                sequence = self._sequence + 1

                def build(d: Decision) -> EventEnvelope:
                    return self.events.build(decision=d, inference=inference, session_id=session_id,
                                             captured_at_utc=captured, capture_sequence=sequence,
                                             request_sha256=request_sha256, latency_ms=latency_ms,
                                             extra_detail=extra)

                try:
                    envelope = build(decision)
                except Exception as exc:
                    # Untraceable/invalid decision or payload: record NO_DECISION instead of losing it.
                    log.error("decision not representable; recording NO_DECISION",
                              extra={"code": "DECISION_CONTRACT_ERROR", "ctx": {"error": type(exc).__name__}})
                    decision = Decision(InspectionResult.NO_DECISION, DecisionReason.RULE_ERROR,
                                        detail="decision violated event contract")
                    envelope = build(decision)
                self._sequence += 1
                record = self.events.record(envelope)
                if not record.persisted:
                    # An untraceable result must never be presented as conclusive.
                    return ProcessOutcome(record=record, result=InspectionResult.NO_DECISION.value,
                                          reason="EVENT_NOT_PERSISTED", code=record.code or ErrorCode.DB_001)
                return ProcessOutcome(record=record, result=decision.result.value, reason=decision.reason.value)

    # -- status --------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        """OpenAPI `Status` document (+ extra properties). Health is evaluated NOW: a stale
        report can never be shown as current normality."""
        report = self.refresh_health()
        model = self.inference.model
        counters = self.events.counters()
        state = self.state.state
        if self.capture is not None:  # capture buffer is the source of frame counts (CAP-003)
            cap = self.capture.counters()
            frames_received, frames_dropped = cap.captured, cap.dropped
            max_queue = max(cap.max_depth, self.queue.max_depth)
        else:
            frames_received, frames_dropped, max_queue = self.queue.received, self.queue.dropped, self.queue.max_depth
        return {
            # -- OpenAPI Status -------------------------------------------------
            "state": state.value,
            "fault_latched": state is SystemState.FAULT or self._fault_latch is not None,
            "active_session_id": self.session_id,
            "components": [c.to_dict() for c in report.components],
            "software_version": __version__,
            "config_version": self.config.config_version,
            "config_sha256": self.config.sha256,
            # 0.1.0 contracts have no cascade `stage`; it is null until the cascade (0.4.0) defines it.
            "models": [{**model.to_dict(), "stage": None, "specialist_for": None,
                        "loaded": bool(self.inference.ready)}] if model else [],
            "counters": {
                "frames_received": frames_received,
                "frames_processed": counters.created,
                "frames_dropped": frames_dropped,
                "events_created": counters.created,
                "events_persisted": counters.persisted,
                "events_failed": counters.failed,
                "max_queue_depth": max_queue,
            },
            "generated_at_utc": utc_now_iso(),
            # -- extra properties (allowed by the schema; owner decision 10/11/2026) ----------
            "software_name": SOFTWARE_NAME,
            "overall": report.overall.value,
            "health_checked_at_utc": report.checked_at_utc,
            "resources": [c.to_dict() for c in report.resources],
            "fault_latch": dict(self._fault_latch) if self._fault_latch else None,
            "inference": {
                "adapter": self.inference.adapter_id,
                "adapter_version": self.inference.adapter_version,
                "model": model.to_dict() if model else None,
            },
            "model_policy": self.model_policy.to_dict(),
            "events": counters.to_dict(),
            "rejected_transitions": self.state.rejected_count,
            "host_memory": host_memory(),
            "mode": "OBSERVATION_ONLY",
            "capture": ({"source_id": self.capture.source.source_id, "state": self.capture.state.value,
                         "detail": self.capture.detail, **self.capture.counters().to_dict()}
                        if self.capture is not None else None),
        }

    def health_document(self) -> tuple[int, dict[str, Any]]:
        """OpenAPI `Health` for GET /health: (HTTP status, body). 503 when FAULT, else 200.
        No sensitive data (the route is unauthenticated)."""
        report = self.refresh_health()
        state = self.state.state
        fault = state is SystemState.FAULT or report.overall is ComponentStatus.FAULT
        return (503 if fault else 200), {"state": state.value, "overall": report.overall.value,
                                         "checked_at_utc": report.checked_at_utc}

    # -- audit ---------------------------------------------------------------
    def _audit_transition(self, rec: TransitionRecord) -> None:
        if rec.accepted and rec.target is SystemState.FAULT:
            if self._fault_latch is None:
                self._fault_latch = {"reason": rec.reason, "at_utc": rec.at_utc, "source": "runtime"}
            self._write_latch(rec.reason, rec.at_utc)
        if self.read_only or not self.store.is_open:
            return
        detail: dict[str, Any] = {"reason": rec.reason, "at_utc": rec.at_utc, "seq": rec.seq}
        if not rec.accepted:
            detail["code"] = rejection_code(rec.source).value
        try:
            self.store.audit(
                actor_ref="system",
                action="STATE_TRANSITION" if rec.accepted else "STATE_TRANSITION_REJECTED",
                target=f"{rec.source}->{rec.target}",
                detail=detail,
                correlation_id=self.session_id or self._pending_session_id,
            )
        except PersistenceError:
            log.error("audit write failed", extra={"code": "AUDIT_WRITE_FAILED"})
