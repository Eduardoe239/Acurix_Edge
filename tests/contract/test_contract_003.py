"""AEX-CONTRACT-003 acceptance: DEC-001, HLT-006, CFG-001.

For every contract (captured frame, inference result, decision, event): valid round trip,
invalid value, missing field and wrong version. Plus: timestamp/latency/software version on
every contract, explicit UNKNOWN/NO_DECISION, NoOp only yields NO_DECISION.
"""

import json
import unittest
import uuid

from acurix_edge import __version__
from acurix_edge.adapters.inference_noop import NoOpInferenceAdapter
from acurix_edge.contracts.capture import CAPTURE_SCHEMA_VERSION, CapturedFrame, CaptureStatus
from acurix_edge.contracts.decision import DECISION_SCHEMA_VERSION, Decision, DecisionReason, InspectionResult
from acurix_edge.contracts.errors import CATALOG, ContractError, ErrorCode
from acurix_edge.contracts.events import EVENT_SCHEMA_VERSION, EventEnvelope
from acurix_edge.contracts.inference import (
    INFERENCE_SCHEMA_VERSION,
    ImageQuality,
    InferenceResult,
    InputDescriptor,
)
from acurix_edge.domain.decision_engine import DecisionContext, DecisionEngine
from acurix_edge.domain.state_machine import SystemState
from tests.contract.test_event_contract import make as make_event
from tests.support import FixtureApprovedRule, ok_result, request

TS = "2026-11-10T12:00:00.000Z"


def frame(**kw) -> CapturedFrame:
    base = dict(frame_id=str(uuid.uuid4()), source_id="fixture-source", sequence=0, captured_at_utc=TS,
                capture_latency_ms=2.0, input=InputDescriptor(64, 48), status=CaptureStatus.OK)
    base.update(kw)
    return CapturedFrame(**base)


def decision(**kw) -> Decision:
    base = dict(result=InspectionResult.NO_DECISION, reason=DecisionReason.NO_APPROVED_RULE, latency_ms=0.1)
    base.update(kw)
    return Decision(**base)


# (factory of a valid object, deserializer, current schema version)
CONTRACTS = {
    "frame": (frame, CapturedFrame.from_dict, CAPTURE_SCHEMA_VERSION),
    "inference": (ok_result, InferenceResult.from_dict, INFERENCE_SCHEMA_VERSION),
    "decision": (decision, Decision.from_dict, DECISION_SCHEMA_VERSION),
    "event": (make_event, EventEnvelope.from_dict, EVENT_SCHEMA_VERSION),
}
TIMESTAMP_KEY = {"frame": "captured_at_utc", "inference": "completed_at_utc", "decision": "decided_at_utc",
                 "event": "captured_at_utc"}
LATENCY_KEY = {"frame": "capture_latency_ms", "inference": "inference_ms", "decision": "latency_ms",
               "event": "latency_ms"}


def wire(obj) -> dict:
    return json.loads(json.dumps(obj.to_dict()))


class CommonContractTests(unittest.TestCase):
    def test_valid_round_trip(self):
        for name, (factory, load, version) in CONTRACTS.items():
            with self.subTest(contract=name):
                obj = factory()
                d = wire(obj)
                self.assertEqual(d["schema_version"], version)
                self.assertEqual(load(d), obj)

    def test_every_contract_carries_timestamp_latency_and_software_version(self):
        for name, (factory, _load, _v) in CONTRACTS.items():
            with self.subTest(contract=name):
                d = wire(factory())
                self.assertRegex(d[TIMESTAMP_KEY[name]], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$")
                self.assertIsInstance(d[LATENCY_KEY[name]], (int, float))
                self.assertTrue(d["software_version"])

    def test_missing_field_is_rejected(self):
        for name, (factory, load, _v) in CONTRACTS.items():
            for key in wire(factory()):
                with self.subTest(contract=name, missing=key):
                    d = wire(factory())
                    d.pop(key)
                    if name == "event" and key in ("rule", "class_id", "confidence", "media_ref", "extra",
                                                   "review_status", "model"):
                        continue  # optional-with-null in the event schema (documented, OpenAPI Event)
                    with self.assertRaises(ContractError):
                        load(d)

    def test_wrong_version_is_rejected(self):
        for name, (factory, load, _v) in CONTRACTS.items():
            for bad in ("9.9", "", None, 1.0):
                with self.subTest(contract=name, version=bad):
                    d = wire(factory())
                    d["schema_version"] = bad
                    with self.assertRaises(ContractError):
                        load(d)

    def test_invalid_values_are_rejected(self):
        cases = {
            "frame": [{"captured_at_utc": "2026-11-10 12:00"}, {"capture_latency_ms": -1}, {"sequence": -1},
                      {"frame_id": "x"}, {"status": "FINE"}, {"software_version": ""}],
            "inference": [{"completed_at_utc": "now"}, {"inference_ms": -0.5}, {"status": "MAYBE"},
                          {"software_version": ""}],
            "decision": [{"decided_at_utc": "x"}, {"latency_ms": float("inf")}, {"result": "OK"},
                         {"software_version": ""}],
            "event": [{"captured_at_utc": "x"}, {"latency_ms": -1}, {"request_sha256": "abc"},
                      {"result": "OK"}],
        }
        for name, mutations in cases.items():
            factory, load, _v = CONTRACTS[name]
            for m in mutations:
                with self.subTest(contract=name, mutation=m):
                    d = wire(factory())
                    d.update(m)
                    with self.assertRaises((ContractError, ValueError)):
                        load(d)


class FrameContractTests(unittest.TestCase):
    def test_unknown_is_explicit_and_default_quality_is_unknown(self):
        f = frame()
        self.assertIs(f.input.quality, ImageQuality.UNKNOWN)
        self.assertEqual(f.software_version, __version__)

    def test_non_ok_requires_capture_code_and_no_pixels(self):
        for status in (CaptureStatus.SOURCE_UNAVAILABLE, CaptureStatus.UNKNOWN):
            with self.subTest(status=status):
                with self.assertRaises(ContractError):
                    frame(status=status)  # no code
                with self.assertRaises(ContractError):
                    frame(status=status, error_code=ErrorCode.DB_001)  # not a capture code
                with self.assertRaises(ContractError):
                    frame(status=status, error_code=ErrorCode.CAP_001, pixels=b"x")
                self.assertEqual(frame(status=status, error_code=ErrorCode.CAP_001).to_dict()["error_code"],
                                 "AEX-CAP-001")
        with self.assertRaises(ContractError):
            frame(error_code=ErrorCode.CAP_001)  # OK with error code

    def test_pixels_are_never_serialized(self):
        self.assertNotIn("pixels", frame(pixels=b"\x00" * 16).to_dict())


class DecisionContractTests(unittest.TestCase):
    def test_conclusive_requires_identified_rule(self):  # DEC-001
        for result in (InspectionResult.PASS, InspectionResult.DEFECT):
            with self.subTest(result=result):
                with self.assertRaises(ContractError):
                    Decision(result, DecisionReason.NO_APPROVED_RULE)
                with self.assertRaises(ContractError):
                    Decision(result, DecisionReason.RULE_APPLIED, rule_id="r")  # no version
                Decision(result, DecisionReason.RULE_APPLIED, rule_id="r", rule_version="1")

    def test_unapproved_rule_yields_no_decision(self):
        d = DecisionEngine(FixtureApprovedRule(approved=False)).decide(
            ok_result(), DecisionContext(SystemState.OBSERVING, True))
        self.assertIs(d.result, InspectionResult.NO_DECISION)
        self.assertIs(d.reason, DecisionReason.NO_APPROVED_RULE)

    def test_engine_measures_latency_even_if_rule_claims_one(self):
        class Liar(FixtureApprovedRule):
            def evaluate(self, inference):
                return Decision(InspectionResult.PASS, DecisionReason.RULE_APPLIED, self.rule_id,
                                self.rule_version, latency_ms=12345.0)

        d = DecisionEngine(Liar()).decide(ok_result(), DecisionContext(SystemState.OBSERVING, True))
        self.assertIsNotNone(d.latency_ms)
        self.assertLess(d.latency_ms, 12345.0)
        no = DecisionEngine(None).decide(ok_result(), DecisionContext(SystemState.IDLE, True))
        self.assertIsNotNone(no.latency_ms)


class NoOpOnlyNoDecisionTests(unittest.TestCase):
    def test_noop_never_produces_conclusive_result(self):
        engine = DecisionEngine(FixtureApprovedRule())
        adapter = NoOpInferenceAdapter()
        for quality in ImageQuality:
            for state in SystemState:
                for health_ok in (True, False):
                    with self.subTest(quality=quality, state=state, health_ok=health_ok):
                        r = adapter.infer(request(quality))
                        d = engine.decide(r, DecisionContext(state, health_ok))
                        self.assertIs(d.result, InspectionResult.NO_DECISION)


class EventVersioningTests(unittest.TestCase):
    def legacy_dict(self) -> dict:
        d = wire(make_event())
        d["schema_version"] = "1.0"
        d.pop("latency_ms")
        d.pop("request_sha256")
        return d

    def test_legacy_1_0_events_still_decode_unchanged(self):
        d = self.legacy_dict()
        e = EventEnvelope.from_dict(d)
        self.assertEqual(e.schema_version, "1.0")
        self.assertIsNone(e.latency_ms)
        self.assertEqual(e.to_dict(), d)  # byte-identical re-serialization keeps the stored hash valid

    def test_1_0_cannot_carry_1_1_fields_and_1_1_requires_them(self):
        d = self.legacy_dict()
        d["latency_ms"] = 1.0
        with self.assertRaises(ContractError):
            EventEnvelope.from_dict(d)
        for key in ("latency_ms", "request_sha256"):
            with self.subTest(key=key):
                with self.assertRaises(ContractError):
                    make_event(**{key: None})


class ErrorCatalogTests(unittest.TestCase):
    def test_catalog_complete_and_well_formed(self):
        self.assertEqual(set(CATALOG), set(ErrorCode))
        import re

        openapi_pattern = re.compile(r"^AEX-[A-Z]{2,4}-[0-9]{3}$")  # OpenAPI Problem.code
        outside = {c for c in ErrorCode if not openapi_pattern.fullmatch(c.value)}
        # KNOWN DOC DIVERGENCE — owner decision 10/11/2026: AEX-STATE-001/002 stay; the defect is
        # in the OpenAPI Problem.code pattern {2,4} and will be fixed in the document. Until then
        # this registers the divergence; any other code outside the pattern fails here.
        self.assertEqual(outside, {ErrorCode.STATE_001, ErrorCode.STATE_002})
        self.assertIsNone(CATALOG[ErrorCode.CLK_001].http_status)  # internal code (owner, 10/11/2026)
        for code in ErrorCode:
            self.assertRegex(code.value, r"^AEX-[A-Z]{2,5}-[0-9]{3}$")
        self.assertEqual(CATALOG[ErrorCode.DB_001].http_status, 503)
        self.assertEqual(CATALOG[ErrorCode.DB_002].http_status, 507)
        self.assertIsNone(CATALOG[ErrorCode.CAS_004].http_status)


if __name__ == "__main__":
    unittest.main()
