"""Regression tests for the fourth independent review (R4-01 .. R4-10)."""

import contextlib
import io
import json
import logging
import math
import sqlite3
import unittest
from unittest import mock

from acurix_edge.__main__ import main
from acurix_edge.adapters.inference_noop import NoOpInferenceAdapter
from acurix_edge.app import FAULT_LATCH_FILE, Application, SessionError
from acurix_edge.config.settings import AppConfig
from acurix_edge.contracts.decision import Decision, DecisionReason, InspectionResult
from acurix_edge.contracts.errors import ContractError
from acurix_edge.contracts.inference import ImageQuality, InferenceRequest, InputDescriptor
from acurix_edge.domain.model_policy import ModelPolicy
from acurix_edge.domain.state_machine import InvalidTransition, StateMachine, SystemState
from acurix_edge.observability.logging import MAX_ITEMS, MAX_REDACT_CHARS, configure_logging, redact
from acurix_edge.persistence import sqlite_store
from acurix_edge.persistence.sqlite_store import CorruptRecordError, PersistenceError, SqliteEventStore
from tests.support import (
    FIXTURE_MODEL,
    FIXTURE_POLICY,
    HEALTHY_HOST,
    FixtureApprovedRule,
    FixtureOkAdapter,
    TempDirTestCase,
    fixture_policy_with,
    request,
    tamper_event_row,
)


class Probe:
    def __call__(self):
        from acurix_edge.health.service import ComponentStatus
        return ComponentStatus.OK, "fixture probe"


class AppCase(TempDirTestCase):
    def app(self, **kw) -> Application:
        cfg = kw.pop("config", AppConfig())
        kw.setdefault("model_policy", fixture_policy_with(type(kw["inference"])) if "inference" in kw else FIXTURE_POLICY)
        kw.setdefault("host_checks", HEALTHY_HOST)
        a = Application(cfg, self.tmp, **kw)
        a.boot()
        self.addCleanup(a.shutdown)
        return a

    def observing(self, **kw) -> Application:
        a = self.app(inference=kw.pop("inference", FixtureOkAdapter()), rule=FixtureApprovedRule(),
                     capture_probe=Probe(), **kw)
        a.start_session()
        self.assertIs(a.state.state, SystemState.OBSERVING)
        return a

    def fault_it(self, a: Application) -> None:
        a.store.close()
        a.refresh_health()
        self.assertIs(a.state.state, SystemState.FAULT)

    @property
    def latch(self):
        return self.tmp / ".data" / FAULT_LATCH_FILE


# -- R4-01 ------------------------------------------------------------------------
class FaultLatchStateMachineTests(unittest.TestCase):
    def test_no_ordinary_transition_leaves_fault(self):
        for target in SystemState:
            with self.subTest(target=target):
                sm = StateMachine()
                sm.enter_fault("x")
                with self.assertRaises(InvalidTransition):
                    sm.transition(target, "try")
                self.assertIs(sm.state, SystemState.FAULT)

    def test_release_fault_only_from_fault_and_only_to_idle(self):
        sm = StateMachine()
        with self.assertRaises(InvalidTransition):
            sm.release_fault("not in fault")
        self.assertIs(sm.state, SystemState.IDLE)
        sm.enter_fault("x")
        with self.assertRaises(InvalidTransition):
            sm.release_fault("to maintenance", target=SystemState.MAINTENANCE)
        self.assertIs(sm.state, SystemState.FAULT)
        with self.assertRaises(ValueError):
            sm.release_fault(" ")
        rec = sm.release_fault("reset")
        self.assertTrue(rec.accepted)
        self.assertIs(sm.state, SystemState.IDLE)


class FaultLatchApplicationTests(AppCase):
    def test_close_session_in_fault_keeps_fault(self):
        a = self.observing()
        sid = a.session_id
        self.fault_it(a)
        a.store.open()  # storage back, but the fault must stay latched
        a.close_session()
        self.assertIs(a.state.state, SystemState.FAULT)
        self.assertIsNone(a.session_id)
        audit = a.store.list_audit()
        self.assertIn(("SESSION_CLOSED_IN_FAULT", sid), [(r["action"], r["correlation_id"]) for r in audit])
        self.assertNotIn("FAULT->IDLE", [r["target"] for r in audit if r["action"] == "STATE_TRANSITION"])

    def test_shutdown_in_fault_keeps_fault_and_latch(self):
        a = self.observing()
        self.fault_it(a)
        a.shutdown()
        self.assertIs(a.state.state, SystemState.FAULT)
        self.assertTrue(self.latch.is_file())

    def test_restart_after_fault_boots_in_fault(self):
        a = self.observing()
        self.fault_it(a)
        a.shutdown()
        b = self.app()
        self.assertIs(b.state.state, SystemState.FAULT)
        self.assertEqual(b.status()["fault_latch"]["source"], "file")
        with self.assertRaises(SessionError):
            b.start_session()

    def test_audit_trail_alone_latches_when_latch_file_is_lost(self):
        a = self.observing()
        a.events._state.enter_fault("TEST_FAULT")  # store still open: audited
        a.shutdown()
        self.latch.unlink()
        b = self.app()
        self.assertIs(b.state.state, SystemState.FAULT)
        self.assertEqual(b.status()["fault_latch"]["source"], "audit")
        self.assertEqual(b.status()["fault_latch"]["reason"], "TEST_FAULT")

    def test_corrupt_latch_file_latches(self):
        self.app().shutdown()
        self.latch.write_bytes(b"\xff not json")
        b = self.app()
        self.assertIs(b.state.state, SystemState.FAULT)
        self.assertEqual(b.status()["fault_latch"]["reason"], "FAULT_LATCH_CORRUPT")

    def test_reset_fault_clears_latch_and_restart_is_idle(self):
        a = self.observing()
        self.fault_it(a)
        a.reset_fault("operator verified storage", actor_ref="op-1")
        self.assertIs(a.state.state, SystemState.IDLE)
        self.assertFalse(self.latch.exists())
        reset = [r for r in a.store.list_audit() if r["action"] == "FAULT_RESET"]
        self.assertEqual(len(reset), 1)
        self.assertEqual(reset[0]["actor_ref"], "op-1")
        a.shutdown()
        b = self.app()
        self.assertIs(b.state.state, SystemState.IDLE)
        self.assertIsNone(b.status()["fault_latch"])

    def test_reset_aborts_when_audit_cannot_be_written(self):
        a = self.observing()
        self.fault_it(a)
        a.store.open()
        real_audit = a.store.audit

        def audit(**kw):
            if kw.get("action") == "FAULT_RESET":
                raise PersistenceError("audit table locked")
            return real_audit(**kw)

        a.store.audit = audit
        with self.assertRaises(SessionError):
            a.reset_fault("try")
        self.assertIs(a.state.state, SystemState.FAULT)
        self.assertTrue(self.latch.is_file())

    def test_reset_refused_read_only(self):
        a = self.observing()
        self.fault_it(a)
        a.shutdown()
        ro = Application(AppConfig(), self.tmp, read_only=True)
        ro.boot()
        self.addCleanup(ro.shutdown)
        self.assertIs(ro.state.state, SystemState.FAULT)
        with self.assertRaises(SessionError):
            ro.reset_fault("x")
        self.assertTrue(self.latch.is_file())

    def run_cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(list(args))
        return code, out.getvalue()

    def test_cli_status_reports_latch_and_reset_fault_releases_it(self):
        root = logging.getLogger("acurix_edge")
        saved = (list(root.handlers), root.propagate, root.level)
        self.addCleanup(lambda: (root.handlers.clear(), root.handlers.extend(saved[0]),
                                 setattr(root, "propagate", saved[1]), root.setLevel(saved[2])))
        base = str(self.tmp)
        self.assertEqual(self.run_cli("init", "--base-dir", base)[0], 0)
        self.latch.write_text(json.dumps({"reason": "PRIOR_FAULT", "at_utc": "x"}), encoding="utf-8")
        code, out = self.run_cli("status", "--base-dir", base)
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(out)["state"], "FAULT")
        self.assertTrue(self.latch.is_file())  # status is read-only
        code, out = self.run_cli("reset-fault", "--reason", "storage verified", "--base-dir", base)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["state"], "IDLE")
        self.assertEqual(self.run_cli("status", "--base-dir", base)[0], 1)  # DEGRADED, no longer FAULT
        self.assertEqual(self.run_cli("reset-fault", "--reason", "again", "--base-dir", base)[0], 2)


# -- R4-02 ------------------------------------------------------------------------
class InferenceRequestValidationTests(AppCase):
    def test_invalid_identifiers_rejected(self):
        for bad in ("", " ", "a b", "x" * 129, "id\n", "١٢", None, 12, b"id", "-lead"):
            with self.subTest(bad=bad), self.assertRaises(ContractError):
                InferenceRequest(inspection_id=bad, input=InputDescriptor(1, 1))

    def test_invalid_input_descriptor_rejected(self):
        for kw in (dict(input=None), dict(input={"width": 1}), dict(input=(64, 48))):
            with self.subTest(kw=kw), self.assertRaises(ContractError):
                InferenceRequest(inspection_id="ok", **kw)
        for w, h in ((0, 48), (64, 0), (70000, 1), (True, 1), (1.0, 1)):
            with self.subTest(w=w, h=h), self.assertRaises(ContractError):
                InputDescriptor(w, h, ImageQuality.VALID)
        InputDescriptor(0, 0, ImageQuality.INVALID)  # an explicitly invalid frame may be empty

    def test_process_rejects_invalid_request_before_any_processing(self):
        adapter = FixtureOkAdapter()
        adapter.infer = mock.Mock(side_effect=AssertionError("adapter must not be called"))
        a = self.observing(inference=adapter)
        mutated = request(inspection_id="good")
        object.__setattr__(mutated, "inspection_id", "")
        fake = mock.Mock(inspection_id="dup", input=InputDescriptor(64, 48, ImageQuality.VALID))
        for bad in (mutated, fake, None, {"inspection_id": "x"}):
            with self.subTest(bad=bad), self.assertRaises(ContractError):
                a.process(bad)
        with self.assertRaises(ContractError):
            a.process(request(), captured_at_utc="2026-13-01T00:00:00.000Z")
        self.assertEqual(a.store.count_events(), 0)
        self.assertEqual(a.events.counters().created, 0)
        adapter.infer.assert_not_called()


# -- R4-03 ------------------------------------------------------------------------
class DuplicateAndCorruptTests(AppCase):
    def test_inspection_duplicate_is_counted_and_logged(self):
        a = self.observing()
        first = a.process(request(inspection_id="same"))
        with self.assertLogs("acurix_edge.events", level="WARNING") as cm:
            again = a.process(request(inspection_id="same"))
        self.assertEqual(again.record.event_id, first.record.event_id)
        self.assertEqual(again.record.outcome.value, "DUPLICATE")
        c = a.events.counters()
        self.assertEqual((c.inspection_duplicates, c.created, c.persisted), (1, 1, 1))
        self.assertTrue(any("INSPECTION_DUPLICATE" == getattr(r, "code", None) for r in cm.records))

    def test_event_level_duplicate_is_logged(self):
        a = self.observing()
        o = a.process(request(inspection_id="x1"))
        envelope = a.store.get(o.record.event_id)
        with self.assertLogs("acurix_edge.events", level="WARNING") as cm:
            rec = a.events.record(envelope)
        self.assertEqual(rec.outcome.value, "DUPLICATE")
        self.assertEqual(a.events.counters().duplicates, 1)
        self.assertTrue(any("EVENT_DUPLICATE" == getattr(r, "code", None) for r in cm.records))

    def corrupt(self, a: Application, event_id: str, payload: str) -> None:
        tamper_event_row(a.store._c(), "UPDATE inspection_event SET payload_json = ? WHERE event_id = ?",
                         (payload, event_id))

    def test_corrupt_payload_is_explicit_fault_not_crash(self):
        adapter = FixtureOkAdapter()
        a = self.observing(inference=adapter)
        o = a.process(request(inspection_id="victim"))
        for payload in ("{not json", json.dumps({"event_id": o.record.event_id})):
            with self.subTest(payload=payload):
                self.corrupt(a, o.record.event_id, payload)
                with self.assertRaises(CorruptRecordError):
                    a.store.get(o.record.event_id)
                with self.assertRaises(CorruptRecordError):
                    a.store.list_events()
        adapter.infer = mock.Mock(side_effect=AssertionError("must not re-infer a corrupt record"))
        out = a.process(request(inspection_id="victim"))
        self.assertEqual((out.result, out.reason), ("NO_DECISION", "EVENT_RECORD_CORRUPT"))
        self.assertFalse(out.record.persisted)
        self.assertIs(a.state.state, SystemState.FAULT)
        self.assertEqual(a.events.counters().corrupt_records, 1)
        adapter.infer.assert_not_called()

    def test_tampered_payload_with_valid_json_fails_digest(self):
        a = self.observing()
        o = a.process(request(inspection_id="t1"))
        stored = a.store.get(o.record.event_id).to_dict()
        stored["reason"] = "TAMPERED"
        self.corrupt(a, o.record.event_id, json.dumps(stored, sort_keys=True, separators=(",", ":")))
        with self.assertRaises(CorruptRecordError) as cm:
            a.store.get(o.record.event_id)
        self.assertIn("payload_sha256 mismatch", str(cm.exception))
        # Digest recomputed to match: the contract/JSON decoding path must still reject it explicitly.
        import hashlib
        for payload in ("{not json", json.dumps({"event_id": o.record.event_id})):
            with self.subTest(payload=payload):
                tamper_event_row(
                    a.store._c(), "UPDATE inspection_event SET payload_json = ?, payload_sha256 = ? WHERE event_id = ?",
                    (payload, hashlib.sha256(payload.encode()).hexdigest(), o.record.event_id))
                with self.assertRaises(CorruptRecordError) as cm:
                    a.store.get(o.record.event_id)
                self.assertNotIn("payload_sha256 mismatch", str(cm.exception))


# -- R4-04 ------------------------------------------------------------------------
class CorrelationTests(AppCase):
    def test_session_start_audits_carry_pending_session_id(self):
        a = self.app()
        sid = a.start_session()
        rows = {r["target"]: r["correlation_id"] for r in a.store.list_audit() if r["action"] == "STATE_TRANSITION"}
        self.assertEqual(rows["IDLE->INITIALIZING"], sid)
        self.assertEqual(rows["INITIALIZING->DEGRADED"], sid)
        self.assertIsNone(a._pending_session_id)

    def test_failed_start_audit_carries_attempted_session_id(self):
        a = self.app()
        a.health.register("forced", lambda: (__import__("acurix_edge.health.service", fromlist=["x"]).ComponentStatus.FAULT,
                                             "forced"), critical=True)
        with self.assertRaises(SessionError):
            a.start_session()
        rows = [r for r in a.store.list_audit() if r["action"] == "STATE_TRANSITION"]
        init = next(r for r in rows if r["target"] == "IDLE->INITIALIZING")
        fault = next(r for r in rows if r["target"] == "INITIALIZING->FAULT")
        self.assertIsNotNone(init["correlation_id"])
        self.assertEqual(init["correlation_id"], fault["correlation_id"])
        self.assertIsNone(a.session_id)


# -- R4-05 ------------------------------------------------------------------------
class DecisionConfidenceTests(unittest.TestCase):
    def test_invalid_confidence_rejected(self):
        for bad in (math.nan, math.inf, -math.inf, -0.01, 1.01, True, False, "0.5", [0.5]):
            with self.subTest(bad=bad), self.assertRaises(ContractError):
                Decision(InspectionResult.REVIEW, DecisionReason.RULE_APPLIED, confidence=bad)

    def test_valid_confidence_accepted(self):
        for ok in (None, 0, 1, 0.0, 1.0, 0.5):
            Decision(InspectionResult.REVIEW, DecisionReason.RULE_APPLIED, confidence=ok)

    def test_schema_version_checked(self):
        with self.assertRaises(ContractError):
            Decision(InspectionResult.REVIEW, DecisionReason.RULE_APPLIED, schema_version="9.9")


# -- R4-06 ------------------------------------------------------------------------
class _BadCheckConnection(sqlite3.Connection):
    def execute(self, sql, *args):  # type: ignore[override]
        if "quick_check" in sql:
            return super().execute("SELECT 'row 1 missing from index'")
        return super().execute(sql, *args)


class InvalidBackupTests(TempDirTestCase):
    def test_backup_failing_quick_check_is_marked_invalid(self):
        s = SqliteEventStore(self.tmp / "data" / "db.sqlite")
        s.open()
        self.addCleanup(s.close)
        dest = self.tmp / "bk" / "copy.sqlite"
        real_connect = sqlite3.connect
        with mock.patch.object(sqlite_store.sqlite3, "connect",
                               side_effect=lambda p, *a, **k: real_connect(p, factory=_BadCheckConnection)):
            with self.assertRaises(PersistenceError) as cm:
                s.backup(dest)
        self.assertIn("quick_check", str(cm.exception))
        self.assertFalse(dest.exists())
        self.assertEqual(len(list((self.tmp / "bk").glob("copy.sqlite.*.invalid"))), 1)  # unique name (B1)

    def test_valid_backup_is_not_marked(self):
        s = SqliteEventStore(self.tmp / "data" / "db.sqlite")
        s.open()
        self.addCleanup(s.close)
        dest = s.backup(self.tmp / "bk" / "ok.sqlite")
        self.assertTrue(dest.is_file())
        self.assertFalse(dest.with_name(dest.name + ".invalid").exists())


# -- R4-09 ------------------------------------------------------------------------
class AllowlistTests(TempDirTestCase):
    def test_policy_structure_validated(self):
        bad = (
            dict(approved_models=[FIXTURE_MODEL]),
            dict(approved_models="x"),
            dict(approved_models=frozenset({"id@1"})),
            dict(real_model_loading=1),
            dict(allowed_adapters="noop"),
            dict(allowed_adapters=frozenset({"noop"})),
            dict(allowed_adapters=frozenset()),
            dict(allowed_adapters=frozenset({NoOpInferenceAdapter()})),
            dict(allowed_adapters={NoOpInferenceAdapter}),
        )
        for kw in bad:
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                ModelPolicy(**kw)

    def test_from_config_rejects_non_list(self):
        cfg = mock.Mock()
        cfg.models.approved_models, cfg.models.real_model_loading = "", False
        with self.assertRaises(ValueError):
            ModelPolicy.from_config(cfg)

    def test_adapter_cannot_authorize_itself(self):
        class Impostor(FixtureOkAdapter):
            adapter_id = "noop"  # claims the authorized id

        class NoOpSubclass(NoOpInferenceAdapter):
            pass

        renamed = NoOpInferenceAdapter()
        renamed.adapter_id = "something-else"
        for adapter in (Impostor(), NoOpSubclass(), renamed):
            with self.subTest(adapter=type(adapter).__name__), self.assertRaises(ValueError):
                Application(AppConfig(), self.tmp, inference=adapter)  # product policy
        with self.assertRaises(ValueError):
            Application(AppConfig(), self.tmp, inference=Impostor(), model_policy=FIXTURE_POLICY)

    def test_product_default_remains_noop_only(self):
        a = Application(AppConfig(), self.tmp)
        self.assertEqual(a.model_policy.to_dict()["allowed_adapters"], ["noop"])
        self.assertEqual(a.model_policy.approved_models, frozenset())


# -- R4-10 ------------------------------------------------------------------------
class LogSizeLimitTests(unittest.TestCase):
    def setUp(self):
        root = logging.getLogger("acurix_edge")
        saved = (list(root.handlers), root.propagate, root.level)

        def restore():
            for h in list(root.handlers):
                root.removeHandler(h)
            for h in saved[0]:
                root.addHandler(h)
            root.propagate, root.level = saved[1], saved[2]

        self.addCleanup(restore)
        self.buf = io.StringIO()
        configure_logging("DEBUG", stream=self.buf)
        self.log = logging.getLogger("acurix_edge.test")

    def entry(self):
        return json.loads(self.buf.getvalue().splitlines()[-1])

    def test_oversized_message_is_omitted_not_truncated(self):
        jwt_head = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJzZWNyZXQtdXNlciJ9"  # incomplete JWT: no rule matches it
        msg = "x" * MAX_REDACT_CHARS + " https://user:hunter2@host " + jwt_head
        self.log.error(msg, extra={"code": "BIG"})
        e = self.entry()
        raw = self.buf.getvalue()
        for secret in ("hunter2", jwt_head, "xxxxxxxx"):
            self.assertNotIn(secret, raw)
        self.assertTrue(e["msg"].startswith("<omitted:"))
        self.assertEqual(e["redaction"]["omitted_strings"], 1)
        self.assertEqual(e["redaction"]["omitted_chars"], len(msg))
        self.assertEqual(e["code"], "BIG")

    def test_oversized_ctx_and_exception_are_omitted(self):
        big = "password=" + "p" * MAX_REDACT_CHARS
        try:
            raise RuntimeError(big)
        except RuntimeError:
            self.log.exception("failed", extra={"ctx": {"detail": big, "items": list(range(MAX_ITEMS + 5))}})
        e = self.entry()
        self.assertNotIn("p" * 64, self.buf.getvalue())
        self.assertGreaterEqual(e["redaction"]["omitted_strings"], 3)  # ctx detail, exc message, trace
        self.assertEqual(e["redaction"]["dropped_items"], 5)
        self.assertEqual(len(e["ctx"]["items"]), MAX_ITEMS + 1)

    def test_normal_message_unchanged_and_no_redaction_field(self):
        self.log.info("token=abc123 ok")
        e = self.entry()
        self.assertEqual(e["msg"], "token=***REDACTED*** ok")
        self.assertNotIn("redaction", e)

    def test_redact_boundary(self):
        self.assertNotIn("omitted", redact("a" * MAX_REDACT_CHARS))
        self.assertIn("omitted", redact("a" * (MAX_REDACT_CHARS + 1)))


if __name__ == "__main__":
    unittest.main()
