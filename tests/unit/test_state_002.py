"""AEX-STATE-002 acceptance: HLT-001, HLT-003, codes AEX-STATE-001/002."""

import itertools
import unittest

from acurix_edge.app import Application, SessionError
from acurix_edge.config.settings import AppConfig
from acurix_edge.contracts.decision import InspectionResult
from acurix_edge.contracts.errors import ErrorCode
from acurix_edge.domain.decision_engine import DecisionContext, DecisionEngine
from acurix_edge.domain.state_machine import (
    ALLOWED_TRANSITIONS,
    RESULT_CAPABLE_STATES,
    InvalidTransition,
    StateMachine,
    SystemState as S,
)
from tests.support import FIXTURE_POLICY, FixtureApprovedRule, TempDirTestCase, ok_result
from tests.unit.test_state_machine import machine_in

EXPECTED_TABLE = {
    S.IDLE: {S.INITIALIZING, S.MAINTENANCE, S.FAULT},
    S.INITIALIZING: {S.OBSERVING, S.DEGRADED, S.FAULT, S.IDLE},
    S.OBSERVING: {S.DEGRADED, S.FAULT, S.IDLE},
    S.DEGRADED: {S.OBSERVING, S.FAULT, S.IDLE},
    S.FAULT: set(),
    S.MAINTENANCE: {S.IDLE, S.FAULT},
}


class TransitionTableTests(unittest.TestCase):
    def test_table_is_the_reviewed_one(self):
        # Any change to the single transition table must be deliberate (this test fails otherwise).
        self.assertEqual({k: set(v) for k, v in ALLOWED_TRANSITIONS.items()}, EXPECTED_TABLE)

    def test_all_36_combinations_with_codes(self):
        allowed = rejected = 0
        for source, target in itertools.product(S, S):
            with self.subTest(source=source, target=target):
                sm = machine_in(source)
                if target in EXPECTED_TABLE[source]:
                    sm.transition(target, "test")
                    self.assertIs(sm.state, target)
                    allowed += 1
                else:
                    with self.assertRaises(InvalidTransition) as ctx:
                        sm.transition(target, "test")
                    want = ErrorCode.STATE_002 if source is S.FAULT else ErrorCode.STATE_001
                    self.assertIs(ctx.exception.code, want)
                    self.assertIn(want.value, str(ctx.exception))
                    self.assertIs(sm.state, source)
                    rejected += 1
        self.assertEqual((allowed, rejected), (15, 21))


class NoPassOutsideObservingTests(unittest.TestCase):
    def test_only_observing_can_produce_pass(self):
        engine = DecisionEngine(FixtureApprovedRule())
        for state in S:
            with self.subTest(state=state):
                d = engine.decide(ok_result(), DecisionContext(state, health_ok=True))
                if state is S.OBSERVING:
                    self.assertIs(d.result, InspectionResult.PASS)
                else:
                    self.assertIs(d.result, InspectionResult.NO_DECISION)
        self.assertEqual(RESULT_CAPABLE_STATES, frozenset({S.OBSERVING}))

    def test_observing_with_health_not_ok_is_no_decision(self):
        d = DecisionEngine(FixtureApprovedRule()).decide(ok_result(), DecisionContext(S.OBSERVING, health_ok=False))
        self.assertIs(d.result, InspectionResult.NO_DECISION)


class ApplicationStateTests(TempDirTestCase):
    def app(self) -> Application:
        a = Application(AppConfig(), self.tmp, model_policy=FIXTURE_POLICY)
        a.boot()
        self.addCleanup(a.shutdown)
        return a

    def test_rejected_transition_is_audited_with_code(self):
        a = self.app()
        with self.assertRaises(InvalidTransition):
            a.state.transition(S.OBSERVING, "skip init")
        row = a.store.list_audit(limit=1)[0]
        self.assertEqual((row["action"], row["target"]), ("STATE_TRANSITION_REJECTED", "IDLE->OBSERVING"))
        self.assertEqual(row["detail"]["code"], "AEX-STATE-001")

    def test_fault_survives_restart_and_blocks_with_state_002(self):
        a = self.app()
        a.state.enter_fault("TEST_FAULT")
        a.shutdown()
        b = self.app()  # new process image over the same data directory
        self.assertIs(b.state.state, S.FAULT)
        with self.assertRaises(SessionError) as ctx:
            b.start_session()
        self.assertIs(ctx.exception.code, ErrorCode.STATE_002)
        with self.assertRaises(InvalidTransition) as ctx2:
            b.state.transition(S.IDLE, "try to leave")
        self.assertIs(ctx2.exception.code, ErrorCode.STATE_002)
        row = b.store.list_audit(limit=1)[0]
        self.assertEqual(row["detail"]["code"], "AEX-STATE-002")

    def test_reset_records_actor_and_reason_then_restart_is_idle(self):
        a = self.app()
        a.state.enter_fault("TEST_FAULT")
        with self.assertRaises(ValueError):
            a.reset_fault("ok", actor_ref=" ")
        with self.assertRaises(ValueError):
            a.reset_fault("  ", actor_ref="tecnico-1")
        a.reset_fault("cabo trocado e banco verificado", actor_ref="tecnico-1")
        reset = next(r for r in a.store.list_audit() if r["action"] == "FAULT_RESET")
        self.assertEqual(reset["actor_ref"], "tecnico-1")
        self.assertEqual(reset["detail"]["reason"], "cabo trocado e banco verificado")
        self.assertEqual(reset["detail"]["latch"]["reason"], "TEST_FAULT")
        a.shutdown()
        self.assertIs(self.app().state.state, S.IDLE)

    def test_maintenance_is_audited_and_blocks_sessions(self):
        a = self.app()
        a.enter_maintenance("ajuste de iluminação", actor_ref="tecnico-1")
        self.assertIs(a.state.state, S.MAINTENANCE)
        with self.assertRaises(SessionError) as ctx:
            a.start_session()
        self.assertIs(ctx.exception.code, ErrorCode.STATE_001)
        a.exit_maintenance("ajuste concluído", actor_ref="tecnico-1")
        self.assertIs(a.state.state, S.IDLE)
        actions = [(r["action"], r["actor_ref"]) for r in a.store.list_audit()]
        self.assertIn(("MAINTENANCE_ENTER", "tecnico-1"), actions)
        self.assertIn(("MAINTENANCE_EXIT", "tecnico-1"), actions)

    def test_maintenance_exit_outside_maintenance_never_transitions(self):
        a = self.app()
        a.start_session()  # NoOp -> DEGRADED
        before = a.state.state
        with self.assertRaises(SessionError) as ctx:
            a.exit_maintenance("indevido", actor_ref="tecnico-1")
        self.assertIs(ctx.exception.code, ErrorCode.STATE_001)
        self.assertIs(a.state.state, before)  # DEGRADED->IDLE is a valid table entry, but not taken here
        row = a.store.list_audit(limit=1)[0]
        self.assertEqual(row["action"], "STATE_TRANSITION_REJECTED")
        self.assertEqual(row["detail"]["code"], "AEX-STATE-001")

    def test_maintenance_refused_in_fault_with_state_002(self):
        a = self.app()
        a.state.enter_fault("TEST_FAULT")
        with self.assertRaises(SessionError) as ctx:
            a.enter_maintenance("tentar", actor_ref="tecnico-1")
        self.assertIs(ctx.exception.code, ErrorCode.STATE_002)
        self.assertIs(a.state.state, S.FAULT)

    def test_second_session_is_ses_001(self):
        a = self.app()
        a.start_session()
        with self.assertRaises(SessionError) as ctx:
            a.start_session()
        self.assertIs(ctx.exception.code, ErrorCode.SES_001)


if __name__ == "__main__":
    unittest.main()
