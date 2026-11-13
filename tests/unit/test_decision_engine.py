import unittest

from acurix_edge.adapters.inference_noop import NoOpInferenceAdapter
from acurix_edge.contracts.decision import Decision, DecisionReason, InspectionResult
from acurix_edge.contracts.errors import ContractError
from acurix_edge.contracts.inference import ImageQuality
from acurix_edge.domain.decision_engine import DecisionContext, DecisionEngine, is_rule_approved
from acurix_edge.domain.state_machine import SystemState
from tests.support import FixtureApprovedRule, ok_result, request

OBS_OK = DecisionContext(SystemState.OBSERVING, health_ok=True)


class DecisionGuardTests(unittest.TestCase):
    def test_noop_never_passes_in_any_state_or_health(self):
        noop = NoOpInferenceAdapter()
        for rule in (None, FixtureApprovedRule(approved=True), FixtureApprovedRule(approved=False)):
            engine = DecisionEngine(rule)
            for state in SystemState:
                for health in (True, False):
                    for quality in ImageQuality:
                        with self.subTest(rule=rule, state=state, health=health, quality=quality):
                            d = engine.decide(noop.infer(request(quality)), DecisionContext(state, health))
                            self.assertNotIn(d.result, (InspectionResult.PASS, InspectionResult.DEFECT))
                            self.assertIs(d.result, InspectionResult.NO_DECISION)

    def test_no_rule_yields_no_decision(self):
        d = DecisionEngine().decide(ok_result(), OBS_OK)
        self.assertIs(d.result, InspectionResult.NO_DECISION)
        self.assertIs(d.reason, DecisionReason.NO_APPROVED_RULE)

    def test_unapproved_rule_is_not_evaluated(self):
        rule = FixtureApprovedRule(approved=False)
        d = DecisionEngine(rule).decide(ok_result(), OBS_OK)
        self.assertIs(d.reason, DecisionReason.NO_APPROVED_RULE)
        self.assertEqual(rule.calls, 0)

    def test_each_precondition_blocks_conclusive_result(self):
        rule = FixtureApprovedRule()
        engine = DecisionEngine(rule)
        cases = [
            (ok_result(), DecisionContext(SystemState.DEGRADED, True), DecisionReason.SYSTEM_NOT_OBSERVING),
            (ok_result(), DecisionContext(SystemState.FAULT, True), DecisionReason.SYSTEM_NOT_OBSERVING),
            (ok_result(), DecisionContext(SystemState.INITIALIZING, True), DecisionReason.SYSTEM_NOT_OBSERVING),
            (ok_result(), DecisionContext(SystemState.OBSERVING, False), DecisionReason.HEALTH_NOT_OK),
            (ok_result(ImageQuality.INVALID), OBS_OK, DecisionReason.INPUT_QUALITY),
            (ok_result(ImageQuality.UNKNOWN), OBS_OK, DecisionReason.INPUT_QUALITY),
        ]
        for inf, ctx, reason in cases:
            with self.subTest(reason=reason, ctx=ctx):
                d = engine.decide(inf, ctx)
                self.assertIs(d.result, InspectionResult.NO_DECISION)
                self.assertIs(d.reason, reason)
        self.assertEqual(rule.calls, 0)

    def test_all_preconditions_met_allows_rule_result(self):
        d = DecisionEngine(FixtureApprovedRule()).decide(ok_result(), OBS_OK)
        self.assertIs(d.result, InspectionResult.PASS)
        self.assertEqual(d.rule_id, "fixture-rule")

    def test_rule_exception_yields_no_decision(self):
        class Broken(FixtureApprovedRule):
            def evaluate(self, inference):
                raise RuntimeError("bug")

        d = DecisionEngine(Broken()).decide(ok_result(), OBS_OK)
        self.assertIs(d.reason, DecisionReason.RULE_ERROR)

    def test_rule_identity_mismatch_rejected(self):
        class Impostor(FixtureApprovedRule):
            def evaluate(self, inference):
                return Decision(InspectionResult.PASS, DecisionReason.RULE_APPLIED, "other", "9")

        d = DecisionEngine(Impostor()).decide(ok_result(), OBS_OK)
        self.assertIs(d.result, InspectionResult.NO_DECISION)

    def test_rule_returning_wrong_type(self):
        class Bad(FixtureApprovedRule):
            def evaluate(self, inference):
                return "PASS"

        self.assertIs(DecisionEngine(Bad()).decide(ok_result(), OBS_OK).reason, DecisionReason.RULE_ERROR)

    def test_conclusive_decision_requires_rule_identity(self):
        for result in (InspectionResult.PASS, InspectionResult.DEFECT):
            with self.subTest(result=result):
                with self.assertRaises(ContractError):
                    Decision(result, DecisionReason.NO_APPROVED_RULE)
                with self.assertRaises(ContractError):
                    Decision(result, DecisionReason.RULE_APPLIED, rule_id="x")


class RuleApprovalShortCircuitTests(unittest.TestCase):
    def test_unapproved_rule_identity_is_never_read(self):
        """NFR-014 review LOW-1: approval is checked first; a broken identity of an UNAPPROVED rule
        must not raise (result stays NO_APPROVED_RULE)."""

        class Broken:
            approved = False

            @property
            def rule_id(self):
                raise RuntimeError("must not be read")

            rule_version = "1"

        self.assertFalse(is_rule_approved(Broken()))
        self.assertFalse(is_rule_approved(None))
        d = DecisionEngine(Broken()).decide(ok_result(), OBS_OK)
        self.assertIs(d.reason, DecisionReason.NO_APPROVED_RULE)
        self.assertTrue(is_rule_approved(FixtureApprovedRule()))


if __name__ == "__main__":
    unittest.main()
