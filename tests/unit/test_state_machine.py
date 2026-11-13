import itertools
import unittest
from typing import Any

from acurix_edge.domain.state_machine import (
    ALLOWED_TRANSITIONS,
    InvalidTransition,
    StateMachine,
)
from acurix_edge.domain.state_machine import (
    SystemState as S,
)


def machine_in(state: S) -> StateMachine:
    paths = {
        S.IDLE: [],
        S.INITIALIZING: [S.INITIALIZING],
        S.OBSERVING: [S.INITIALIZING, S.OBSERVING],
        S.DEGRADED: [S.INITIALIZING, S.DEGRADED],
        S.FAULT: [S.FAULT],
        S.MAINTENANCE: [S.MAINTENANCE],
    }
    sm = StateMachine()
    for target in paths[state]:
        sm.transition(target, "setup")
    assert sm.state is state
    return sm


class StateMachineTests(unittest.TestCase):
    def test_boots_in_idle(self):
        self.assertIs(StateMachine().state, S.IDLE)

    def test_all_six_states_defined(self):
        self.assertEqual({s.value for s in S},
                         {"IDLE", "INITIALIZING", "OBSERVING", "DEGRADED", "FAULT", "MAINTENANCE"})
        self.assertEqual(set(ALLOWED_TRANSITIONS), set(S))

    def test_full_transition_matrix(self):
        for source, target in itertools.product(S, S):
            with self.subTest(source=source, target=target):
                sm = machine_in(source)
                if target in ALLOWED_TRANSITIONS[source]:
                    sm.transition(target, "test")
                    self.assertIs(sm.state, target)
                else:
                    with self.assertRaises(InvalidTransition):
                        sm.transition(target, "test")
                    self.assertIs(sm.state, source)
                    self.assertEqual(sm.rejected_count, 1)
                    self.assertFalse(sm.history()[-1].accepted)

    def test_idle_cannot_jump_to_observing(self):
        with self.assertRaises(InvalidTransition):
            StateMachine().transition(S.OBSERVING, "skip init")

    def test_reason_is_mandatory(self):
        with self.assertRaises(ValueError):
            StateMachine().transition(S.INITIALIZING, "  ")

    def test_enter_fault_from_every_state_and_idempotent(self):
        for source in S:
            with self.subTest(source=source):
                sm = machine_in(source)
                sm.enter_fault("boom")
                self.assertIs(sm.state, S.FAULT)
                self.assertIsNone(sm.enter_fault("again"))
                self.assertIs(sm.state, S.FAULT)

    def test_listener_receives_rejections_and_failures_do_not_corrupt(self):
        seen: list[Any] = []
        sm = StateMachine()
        sm.subscribe(lambda rec: (_ for _ in ()).throw(RuntimeError("listener bug")))
        sm.subscribe(seen.append)
        sm.transition(S.INITIALIZING, "ok")
        with self.assertRaises(InvalidTransition):
            sm.transition(S.MAINTENANCE, "bad")
        self.assertEqual([r.accepted for r in seen], [True, False])
        self.assertIs(sm.state, S.INITIALIZING)

    def test_string_targets_coerced_or_rejected(self):
        sm = StateMachine()
        rec = sm.transition("INITIALIZING", "str target")
        self.assertIs(sm.state, S.INITIALIZING)
        self.assertIs(type(rec.target), S)
        with self.assertRaises(ValueError):
            sm.transition("RUNNING", "unknown")
        self.assertIs(sm.state, S.INITIALIZING)

    def test_sequence_numbers_monotonic(self):
        sm = StateMachine()
        sm.transition(S.INITIALIZING, "a")
        with self.assertRaises(InvalidTransition):
            sm.transition(S.MAINTENANCE, "b")
        sm.enter_fault("c")
        self.assertEqual([r.seq for r in sm.history()], [1, 2, 3])

    def test_history_bounded(self):
        sm = StateMachine(history_size=4)
        for _ in range(5):
            sm.transition(S.MAINTENANCE, "m")
            sm.transition(S.IDLE, "i")
        self.assertEqual(len(sm.history()), 4)


if __name__ == "__main__":
    unittest.main()
