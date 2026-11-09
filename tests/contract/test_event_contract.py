import json
import math
import unittest
import uuid

from acurix_edge.contracts.decision import InspectionResult
from acurix_edge.contracts.errors import ContractError
from acurix_edge.contracts.events import EVENT_SCHEMA_VERSION, EventEnvelope, ReviewStatus

CFG = {"version": "0.1.0", "sha256": "b" * 64}
MODEL = {"id": "fixture-model", "version": "0.0.0-test", "sha256": "0" * 64}
RULE = {"id": "fixture-rule", "version": "0.0.0-test"}


def make(**kw) -> EventEnvelope:
    base = dict(
        event_id=EventEnvelope.new_id(), site_id="site", station_id="st", session_id="s", inspection_id="insp-1",
        captured_at_utc="2026-11-03T12:00:00.000Z", capture_sequence=1, result=InspectionResult.REVIEW,
        reason="X", inference_status="NO_MODEL", model=None, configuration=CFG, software_version="0.1.0",
        review_status=ReviewStatus.PENDING,
    )
    base.update(kw)
    return EventEnvelope(**base)


CONCLUSIVE_OK = dict(result=InspectionResult.PASS, model=MODEL, rule=RULE, inference_status="OK")


class EventContractTests(unittest.TestCase):
    def test_roundtrip_and_stable_hash(self):
        e = make(media_ref="evidence/2026/a.jpg")
        d = json.loads(e.canonical_json())
        self.assertEqual(d["schema_version"], EVENT_SCHEMA_VERSION)
        again = EventEnvelope.from_dict(d)
        self.assertEqual(again, e)
        self.assertEqual(again.payload_sha256(), e.payload_sha256())
        self.assertNotEqual(make(event_id=e.event_id, capture_sequence=2).payload_sha256(), e.payload_sha256())

    def test_conclusive_with_full_traceability_is_accepted(self):
        e = make(**CONCLUSIVE_OK, confidence=0.5, class_id="fixture_class")
        self.assertEqual(EventEnvelope.from_dict(json.loads(e.canonical_json())), e)

    def test_rejections(self):
        bad = [
            dict(event_id="not-a-uuid"),
            dict(event_id=str(uuid.uuid4()).upper()),
            dict(event_id="urn:uuid:" + str(uuid.uuid4())),
            dict(event_id="{" + str(uuid.uuid4()) + "}"),
            dict(captured_at_utc="2026-11-03T12:00:00-03:00"),
            dict(captured_at_utc="2026-11-03T12:00:00Z"),
            dict(captured_at_utc="yesterdayZ"),
            dict(captured_at_utc="٢026-11-03T12:00:00.000Z"),  # non-ASCII digit
            dict(inspection_id=""),
            dict(capture_sequence=-1),
            dict(capture_sequence=True),
            dict(result="PASS"),
            dict(configuration={"version": "0.1.0"}),
            dict(confidence=math.nan),
            dict(confidence=1.5),
            dict(confidence=True),
            dict(class_id=""),
            dict(inference_status="MAYBE"),
            dict(media_ref="../secret.jpg"),
            dict(media_ref="C:/abs.jpg"),
            dict(media_ref="/abs.jpg"),
            dict(software_version=""),
            dict(schema_version="2.0"),
            dict(model={"id": "m"}),
            dict(rule={"id": "r"}),
        ]
        for result in (InspectionResult.PASS, InspectionResult.DEFECT):
            bad += [
                {**CONCLUSIVE_OK, "result": result, "model": None},
                {**CONCLUSIVE_OK, "result": result, "rule": None},
                {**CONCLUSIVE_OK, "result": result, "inference_status": "NO_MODEL"},
                {**CONCLUSIVE_OK, "result": result, "model": {**MODEL, "sha256": "bad"}},
                {**CONCLUSIVE_OK, "result": result, "model": {"id": "m", "version": "1"}},
            ]
        for kw in bad:
            with self.subTest(kw=kw):
                with self.assertRaises(ContractError):
                    make(**kw)

    def test_from_dict_malformed(self):
        for mutate in (
            lambda d: d.update(result="MAYBE"),
            lambda d: d.pop("session_id"),
            lambda d: d.update(session_id=None),
            lambda d: d.pop("inference_status"),
        ):
            d = json.loads(make().canonical_json())
            mutate(d)
            with self.subTest(d=d):
                with self.assertRaises(ContractError):
                    EventEnvelope.from_dict(d)

    def test_caller_mutation_does_not_change_envelope(self):
        cfg = dict(CFG)
        extra = {"k": "v"}
        e = make(configuration=cfg, extra=extra)
        digest = e.payload_sha256()
        cfg["version"] = "9.9.9"
        extra["k"] = "changed"
        e.to_dict()["configuration"]["version"] = "x"
        self.assertEqual(e.payload_sha256(), digest)


if __name__ == "__main__":
    unittest.main()
