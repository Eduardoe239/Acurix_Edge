import contextlib
import io
import json
from typing import Any, cast

from acurix_edge.__main__ import main
from acurix_edge.app import Application, SessionError
from acurix_edge.config.settings import AppConfig, StorageConfig
from acurix_edge.contracts.inference import InferenceResult, InferenceStatus, ModelRef
from acurix_edge.domain.state_machine import SystemState
from acurix_edge.health.service import ComponentStatus
from tests.support import (
    FIXTURE_POLICY,
    FIXTURE_RUNTIME,
    HEALTHY_HOST,
    FixtureApprovedRule,
    FixtureOkAdapter,
    TempDirTestCase,
    fixture_policy_with,
    request,
)


class Probe:
    """Test-only capture probe whose status can be changed during the test."""

    def __init__(self, status=ComponentStatus.OK):
        self.status = status

    def __call__(self):
        return self.status, "fixture probe"


class ApplicationTests(TempDirTestCase):
    def app(self, **kw) -> Application:
        cfg = kw.pop("config", AppConfig())
        # Test adapter subclasses are registered explicitly in the (test-only) policy seam.
        kw.setdefault("model_policy",
                      fixture_policy_with(type(kw["inference"])) if "inference" in kw else FIXTURE_POLICY)
        kw.setdefault("host_checks", HEALTHY_HOST)
        a = Application(cfg, self.tmp, **kw)
        a.boot()
        self.addCleanup(a.shutdown)
        return a

    def observing_app(self, **kw):
        probe = Probe()
        a = self.app(inference=kw.pop("inference", FixtureOkAdapter()), rule=kw.pop("rule", FixtureApprovedRule()),
                     capture_probe=probe, **kw)
        a.start_session()
        return a, probe

    def assert_counters_consistent(self, a):
        c = a.events.counters()
        self.assertEqual(c.created, c.persisted + c.duplicates + c.failed)

    # -- baseline behaviour ---------------------------------------------------
    def test_boot_idle_and_noop_session_is_degraded_not_observing(self):
        a = self.app()
        self.assertIs(a.state.state, SystemState.IDLE)
        a.start_session()
        self.assertIs(a.state.state, SystemState.DEGRADED)
        self.assertEqual(a.status()["overall"], "DEGRADED")

    def test_noop_pipeline_persists_no_decision(self):
        a = self.app()
        sid = a.start_session()
        outcomes = [a.process(request(inspection_id=f"i{i}")) for i in range(3)]
        for o in outcomes:
            self.assertEqual(o.result, "NO_DECISION")
            self.assertTrue(o.record.persisted)
        events = a.store.list_events(session_id=sid)
        self.assertEqual([e.capture_sequence for e in events], [1, 2, 3])
        e = events[0]
        self.assertIsNone(e.model)
        self.assertIsNone(e.rule)
        self.assertEqual(e.inference_status, "NO_MODEL")
        self.assertEqual(e.configuration, a.config.reference())
        self.assertEqual(a.events.counters().to_dict(), {"created": 3, "persisted": 3, "duplicates": 0, "failed": 0,
                                                         "inspection_duplicates": 0, "corrupt_records": 0,
                                                         "inspection_conflicts": 0})

    def test_no_pass_reachable_without_capture_even_with_fixture_model_and_rule(self):
        a = self.app(inference=FixtureOkAdapter(), rule=FixtureApprovedRule())
        a.start_session()
        self.assertIs(a.state.state, SystemState.DEGRADED)
        o = a.process(request())
        self.assertEqual(o.result, "NO_DECISION")
        self.assertEqual(o.reason, "SYSTEM_NOT_OBSERVING")

    # -- guard positive path + stale-health protection -----------------------
    def test_all_preconditions_ok_persists_traceable_conclusive_result(self):
        a, _ = self.observing_app()
        self.assertIs(a.state.state, SystemState.OBSERVING)
        o = a.process(request())
        self.assertEqual(o.result, "PASS")
        self.assertTrue(o.record.persisted)
        e = a.store.get(o.record.event_id)
        self.assertEqual(e.rule, {"id": "fixture-rule", "version": "0.0.0-test"})
        self.assertEqual(e.inference_status, "OK")
        self.assertEqual(e.model["sha256"], "0" * 64)

    def test_health_drop_blocks_next_decision(self):
        a, probe = self.observing_app()
        self.assertEqual(a.process(request()).result, "PASS")
        probe.status = ComponentStatus.FAULT  # non-critical component fails after start
        o = a.process(request(inspection_id="i2"))
        self.assertEqual(o.result, "NO_DECISION")
        self.assertIs(a.state.state, SystemState.DEGRADED)

    def test_stale_health_is_never_reused_regardless_of_timer_resolution(self):
        a, probe = self.observing_app()
        for i in range(50):  # tight loop: elapsed time often reads as 0 on Windows
            self.assertEqual(a.process(request(inspection_id=f"ok{i}")).result, "PASS")
        probe.status = ComponentStatus.DEGRADED
        self.assertEqual(a.process(request(inspection_id="after")).result, "NO_DECISION")

    def test_refresh_health_round_trip(self):
        a, probe = self.observing_app()
        probe.status = ComponentStatus.DEGRADED
        a.refresh_health()
        self.assertIs(a.state.state, SystemState.DEGRADED)
        probe.status = ComponentStatus.OK
        a.refresh_health()
        self.assertIs(a.state.state, SystemState.OBSERVING)

    def test_truthy_non_bool_approval_is_not_approval(self):
        rule = FixtureApprovedRule()
        rule.approved = cast(Any, "yes")  # truthy non-bool on purpose
        a, _ = self.observing_app(rule=rule)
        self.assertIs(a.state.state, SystemState.DEGRADED)
        self.assertEqual(a.process(request()).reason, "SYSTEM_NOT_OBSERVING")

    # -- adapter misbehaviour ------------------------------------------------
    def test_adapter_exception_is_recorded_as_no_decision(self):
        class Boom(FixtureOkAdapter):
            def infer(self, req):
                raise RuntimeError("driver crash")

        a, _ = self.observing_app(inference=Boom())
        o = a.process(request())
        self.assertEqual(o.result, "NO_DECISION")
        self.assertTrue(o.record.persisted)
        e = a.store.get(o.record.event_id)
        self.assertEqual(e.inference_status, "RUNTIME_ERROR")
        self.assertIn("RuntimeError", e.extra["inference_error_code"])

    def test_adapter_mismatched_outputs_rejected(self):
        other_model = ModelRef("other", "1", "1" * 64)

        class WrongId(FixtureOkAdapter):
            def infer(self, req):
                return InferenceResult(inspection_id="someone-else", model=self.model, runtime=FIXTURE_RUNTIME,
                                       input=req.input, status=InferenceStatus.OK)

        class WrongModel(FixtureOkAdapter):
            def infer(self, req):
                return InferenceResult(inspection_id=req.inspection_id, model=other_model, runtime=FIXTURE_RUNTIME,
                                       input=req.input, status=InferenceStatus.OK)

        class WrongType(FixtureOkAdapter):
            def infer(self, req):
                return {"status": "OK"}

        for adapter, code in ((WrongId(), "INSPECTION_ID_MISMATCH"), (WrongModel(), "MODEL_IDENTITY_MISMATCH"),
                              (WrongType(), "ADAPTER_INVALID_RETURN_TYPE")):
            with self.subTest(code=code):
                a, _ = self.observing_app(inference=adapter, store=None)
                o = a.process(request())
                self.assertEqual(o.result, "NO_DECISION")
                e = a.store.get(o.record.event_id)
                self.assertEqual(e.inference_status, "INCOMPATIBLE")
                self.assertEqual(e.extra["inference_error_code"], code)
                a.shutdown()

    def test_adapter_cannot_upgrade_input_quality(self):
        from acurix_edge.contracts.inference import ImageQuality, InputDescriptor

        class Upgrader(FixtureOkAdapter):
            def infer(self, req):
                return InferenceResult(inspection_id=req.inspection_id, model=self.model, runtime=FIXTURE_RUNTIME,
                                       input=InputDescriptor(req.input.width, req.input.height, ImageQuality.VALID),
                                       status=InferenceStatus.OK)

        a, _ = self.observing_app(inference=Upgrader())
        o = a.process(request(ImageQuality.UNKNOWN))
        self.assertEqual(o.result, "NO_DECISION")
        self.assertEqual(a.store.get(o.record.event_id).extra["inference_error_code"], "INPUT_DESCRIPTOR_MISMATCH")

    def test_rule_with_invalid_payload_is_recorded_as_no_decision(self):
        from acurix_edge.contracts.decision import Decision, DecisionReason, InspectionResult

        class OverConfident(FixtureApprovedRule):
            def evaluate(self, inference):
                return Decision(InspectionResult.PASS, DecisionReason.RULE_APPLIED, self.rule_id,
                                self.rule_version, confidence=1.5)

        a, _ = self.observing_app(rule=OverConfident())
        o = a.process(request())
        self.assertEqual(o.result, "NO_DECISION")
        self.assertEqual(o.reason, "RULE_ERROR")
        self.assertTrue(o.record.persisted)

    def test_invalid_capture_timestamp_rejected_before_processing(self):
        from acurix_edge.contracts.errors import ContractError

        a, _ = self.observing_app()
        with self.assertRaises(ContractError):
            a.process(request(), captured_at_utc="2026-11-03 12:00")
        self.assertEqual(a.events.counters().created, 0)
        self.assertEqual(a.process(request()).record.persisted, True)
        self.assertEqual(a.store.list_events()[0].capture_sequence, 1)

    def test_concurrent_refresh_health_is_serialized(self):
        import threading

        a, probe = self.observing_app()
        gate, entered = threading.Event(), threading.Event()
        calls = {"n": 0}

        def blocking_probe():
            calls["n"] += 1
            if calls["n"] == 1:
                entered.set()
                gate.wait(5)  # first refresh holds the health lock while evaluating
            return ComponentStatus.DEGRADED, "blocking probe"

        a._capture_probe = blocking_probe
        t1 = threading.Thread(target=a.refresh_health)
        t1.start()
        self.assertTrue(entered.wait(5))
        t2 = threading.Thread(target=a.refresh_health)
        t2.start()
        t2.join(0.2)
        self.assertTrue(t2.is_alive())  # second refresh is blocked behind the first (lock held)
        gate.set()
        t1.join(5)
        t2.join(5)
        self.assertIs(a.state.state, SystemState.DEGRADED)
        self.assertEqual(a.state.rejected_count, 0)

    def test_close_session_waits_for_inflight_inspection(self):
        import threading

        started, release = threading.Event(), threading.Event()

        class Blocking(FixtureOkAdapter):
            def infer(self, req):
                started.set()
                release.wait(5)
                return super().infer(req)

        a, _ = self.observing_app(inference=Blocking())
        results: dict[str, Any] = {}
        worker = threading.Thread(target=lambda: results.setdefault("o", a.process(request())))
        worker.start()
        self.assertTrue(started.wait(5))  # inspection is in flight
        closer = threading.Thread(target=a.close_session)
        closer.start()
        closer.join(0.2)
        self.assertTrue(closer.is_alive())  # close waits for the in-flight inspection
        release.set()
        worker.join(5)
        closer.join(5)
        self.assertTrue(results["o"].record.persisted)
        self.assertIs(a.state.state, SystemState.IDLE)
        with self.assertRaises(SessionError):
            a.process(request(inspection_id="after-close"))
        self.assertEqual(a.store.count_events(), 1)
        self.assert_counters_consistent(a)

    def test_start_session_serialized_with_refresh(self):
        import threading

        a = self.app(inference=FixtureOkAdapter(), rule=FixtureApprovedRule(), capture_probe=Probe())
        stop = threading.Event()

        def hammer():
            while not stop.is_set():
                a.refresh_health()

        t = threading.Thread(target=hammer)
        t.start()
        try:
            for _ in range(20):
                a.start_session()
                self.assertIs(a.state.state, SystemState.OBSERVING)
                a.close_session()
        finally:
            stop.set()
            t.join(5)
        self.assertEqual(a.state.rejected_count, 0)

    def test_retry_same_inspection_is_idempotent(self):
        a, _ = self.observing_app()
        first = a.process(request(inspection_id="same"))
        again = a.process(request(inspection_id="same"))
        self.assertEqual(again.record.event_id, first.record.event_id)
        self.assertEqual(again.record.outcome.value, "DUPLICATE")
        self.assertEqual(again.result, first.result)
        self.assertEqual(a.store.count_events(), 1)
        nxt = a.process(request(inspection_id="next"))
        self.assertEqual(a.store.get(nxt.record.event_id).capture_sequence, 2)

    def test_conclusive_result_not_returned_when_not_persisted(self):
        a, _ = self.observing_app()

        def explode(env):
            from acurix_edge.persistence.sqlite_store import PersistenceError
            raise PersistenceError("disk full")

        a.store.append = explode
        o = a.process(request())
        self.assertFalse(o.record.persisted)
        self.assertEqual(o.result, "NO_DECISION")
        self.assertEqual(o.reason, "EVENT_NOT_PERSISTED")
        self.assertIs(a.state.state, SystemState.FAULT)

    def test_adapter_constructing_invalid_result_is_runtime_error(self):
        class BadRuntime(FixtureOkAdapter):
            def infer(self, req):
                return InferenceResult(inspection_id=req.inspection_id, model=self.model, runtime="cpu",
                                       input=req.input, status=InferenceStatus.OK)

        a, _ = self.observing_app(inference=BadRuntime())
        o = a.process(request())
        self.assertEqual(o.result, "NO_DECISION")
        self.assertEqual(a.store.get(o.record.event_id).inference_status, "RUNTIME_ERROR")

    def test_empty_timestamp_is_rejected_not_replaced(self):
        from acurix_edge.contracts.errors import ContractError

        a, _ = self.observing_app()
        with self.assertRaises(ContractError):
            a.process(request(), captured_at_utc="")

    def test_reset_fault_requires_healthy_critical_components(self):
        a = self.app()
        a.start_session()
        a.store.close()
        a.refresh_health()
        self.assertIs(a.state.state, SystemState.FAULT)
        with self.assertRaises(ValueError):
            a.reset_fault(" ")
        a.reset_fault("operator verified storage")  # re-opens the store
        self.assertIs(a.state.state, SystemState.IDLE)
        self.assertIsNone(a.session_id)
        self.assertIn(("STATE_TRANSITION", "FAULT->IDLE"),
                      [(r["action"], r["target"]) for r in a.store.list_audit()])
        with self.assertRaises(SessionError):
            a.reset_fault("not in fault")

    def test_reset_fault_refused_while_db_unavailable(self):
        (self.tmp / "blocked").write_text("not a dir")
        a = self.app(config=AppConfig(storage=StorageConfig(data_dir="blocked")))
        with self.assertRaises(SessionError):
            a.reset_fault("try")
        self.assertIs(a.state.state, SystemState.FAULT)

    # -- ADR-016 default-deny model policy -------------------------------------
    def test_product_policy_is_default_deny(self):
        a = Application(AppConfig(), self.tmp)
        self.assertEqual(a.status()["model_policy"], {"policy": "DEFAULT_DENY", "approved_models": [],
                                                       "real_model_loading": False, "allowed_adapters": ["noop"]})

    def test_product_policy_rejects_non_noop_adapter(self):
        with self.assertRaises(ValueError):
            Application(AppConfig(), self.tmp, inference=FixtureOkAdapter())

    def test_unapproved_model_never_reaches_decision(self):
        from acurix_edge.domain.model_policy import ModelPolicy

        # adapter allowed, but its model is NOT in approved_models (or loading disabled)
        for policy in (ModelPolicy(allowed_adapters=frozenset({FixtureOkAdapter})),
                       ModelPolicy(approved_models=FIXTURE_POLICY.approved_models, real_model_loading=False,
                                   allowed_adapters=frozenset({FixtureOkAdapter}))):
            with self.subTest(policy=policy):
                a, _ = self.observing_app(model_policy=policy)
                self.assertIs(a.state.state, SystemState.DEGRADED)
                model_health = next(c for c in a.status()["components"] if c["component"] == "models")
                self.assertEqual(model_health["status"], "DEGRADED")
                self.assertIn("default-deny", model_health["detail"])
                o = a.process(request())
                self.assertEqual(o.result, "NO_DECISION")
                e = a.store.get(o.record.event_id)
                self.assertEqual(e.extra["inference_error_code"], "MODEL_NOT_APPROVED")
                a.shutdown()

    # -- persistence failures -------------------------------------------------
    def test_persistence_failure_mid_session_goes_fault_and_not_persisted(self):
        a = self.app()
        a.start_session()
        a.store.close()  # simulate database becoming unavailable
        o = a.process(request())
        self.assertFalse(o.record.persisted)
        self.assertIsNotNone(o.record.error)
        self.assertIs(a.state.state, SystemState.FAULT)
        self.assertEqual(a.events.counters().failed, 1)
        self.assert_counters_consistent(a)
        with self.assertRaises(SessionError):
            a.process(request())

    def test_unexpected_store_exception_counted_and_faults(self):
        a = self.app()
        a.start_session()

        def explode(env):
            raise TypeError("unexpected")

        a.store.append = explode
        o = a.process(request())
        self.assertFalse(o.record.persisted)
        self.assertIs(a.state.state, SystemState.FAULT)
        self.assert_counters_consistent(a)

    def test_boot_with_unavailable_db_is_fault_and_session_refused(self):
        (self.tmp / "blocked").write_text("not a dir")
        cfg = AppConfig(storage=StorageConfig(data_dir="blocked"))
        a = self.app(config=cfg)
        self.assertIs(a.state.state, SystemState.FAULT)
        self.assertEqual(a.status()["overall"], "FAULT")
        with self.assertRaises(SessionError):
            a.start_session()

    def test_restart_does_not_resume_session_and_keeps_events(self):
        a = Application(AppConfig(), self.tmp)
        a.boot()
        a.start_session()
        a.process(request())
        a.store.close()  # abrupt stop without closing session
        b = self.app()
        self.assertIs(b.state.state, SystemState.IDLE)
        self.assertIsNone(b.session_id)
        self.assertEqual(b.store.count_events(), 1)

    def test_transitions_are_audited_with_sequence(self):
        a = self.app()
        a.start_session()
        a.close_session()
        audit = a.store.list_audit()
        actions = [(r["action"], r["target"]) for r in audit]
        self.assertIn(("STATE_TRANSITION", "IDLE->INITIALIZING"), actions)
        self.assertIn(("STATE_TRANSITION", "DEGRADED->IDLE"), actions)
        seqs = [r["detail"]["seq"] for r in reversed(audit)]
        self.assertEqual(seqs, sorted(seqs))
        with self.assertRaises(SessionError):
            a.close_session()

    def test_refresh_health_moves_to_fault_on_db_loss(self):
        a = self.app()
        a.start_session()
        a.store.close()
        report = a.refresh_health()
        self.assertIs(report.component("database").status, ComponentStatus.FAULT)
        self.assertIs(a.state.state, SystemState.FAULT)

    def test_status_snapshot_contents(self):
        st = self.app().status()
        self.assertEqual(st["software_version"], "0.1.0")
        self.assertEqual(st["mode"], "OBSERVATION_ONLY")
        self.assertIsNone(st["inference"]["model"])
        names = {c["component"] for c in st["components"]}
        self.assertEqual(names, {"database", "storage", "models", "camera", "clock"})
        self.assertEqual({c["component"] for c in st["resources"]}, {"memory", "cpu", "gpu", "queue", "decision_rule"})
        json.dumps(st)

    # -- CLI ------------------------------------------------------------------
    def run_cli(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = main(list(args))
        return code, out.getvalue()

    def test_cli_status_is_read_only_and_init_creates_db(self):
        db = self.tmp / ".data" / "acurix.db"
        code, out = self.run_cli("status", "--base-dir", str(self.tmp))
        self.assertEqual(code, 2)  # database not initialized -> FAULT
        self.assertFalse(db.exists())  # status never creates the database
        self.assertEqual(json.loads(out)["state"], "FAULT")
        code, _ = self.run_cli("init", "--base-dir", str(self.tmp))
        self.assertEqual(code, 0)
        self.assertTrue(db.exists())
        code, out = self.run_cli("status", "--base-dir", str(self.tmp))
        self.assertEqual(code, 1)  # DEGRADED: no model/camera yet
        self.assertEqual(json.loads(out)["state"], "IDLE")

    def test_cli_invalid_config_exit_2(self):
        bad = self.tmp / "bad.toml"
        for text in ('[api]\nbind_host = "0.0.0.0"\n', '[api]\nbind_host = ["127.0.0.1"]\n',
                     '[logging]\nlevel = {a = 1}\n'):
            with self.subTest(text=text):
                bad.write_text(text)
                self.assertEqual(self.run_cli("status", "--config", str(bad))[0], 2)
