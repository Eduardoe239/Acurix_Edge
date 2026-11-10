import json
import math
import unittest

from acurix_edge.adapters.inference_noop import NoOpInferenceAdapter
from acurix_edge.contracts.errors import ContractError
from acurix_edge.contracts.inference import (
    INFERENCE_SCHEMA_VERSION,
    ImageQuality,
    InferenceAdapter,
    InferenceResult,
    InferenceStatus,
    InputDescriptor,
    ModelRef,
    Prediction,
)
from tests.support import FIXTURE_MODEL, FIXTURE_RUNTIME, FixtureOkAdapter, ok_result, request

REQUIRED_KEYS = {"schema_version", "inspection_id", "model", "runtime", "input", "predictions",
                 "inference_ms", "status", "error_code", "completed_at_utc", "software_version"}


class InferenceContractTests(unittest.TestCase):
    def test_adapters_satisfy_protocol(self):
        for adapter in (NoOpInferenceAdapter(), FixtureOkAdapter()):
            with self.subTest(adapter=adapter.adapter_id):
                self.assertIsInstance(adapter, InferenceAdapter)

    def test_noop_output_contract(self):
        r = NoOpInferenceAdapter().infer(request())
        self.assertIs(r.status, InferenceStatus.NO_MODEL)
        self.assertIsNone(r.model)
        self.assertEqual(r.predictions, ())
        self.assertTrue(r.error_code)
        self.assertFalse(r.is_usable)
        self.assertFalse(NoOpInferenceAdapter().ready)
        d = json.loads(json.dumps(r.to_dict()))
        self.assertEqual(set(d), REQUIRED_KEYS)
        self.assertEqual(d["schema_version"], INFERENCE_SCHEMA_VERSION)

    def test_roundtrip(self):
        for r in (ok_result(), NoOpInferenceAdapter().infer(request(ImageQuality.INVALID))):
            with self.subTest(status=r.status):
                self.assertEqual(InferenceResult.from_dict(json.loads(json.dumps(r.to_dict()))), r)

    def test_invariants(self):
        base = dict(inspection_id="i", runtime=FIXTURE_RUNTIME, input=InputDescriptor(1, 1, ImageQuality.VALID))
        bad = [
            dict(status=InferenceStatus.OK, model=None),
            dict(status=InferenceStatus.OK, model=FIXTURE_MODEL, error_code="E"),
            dict(status=InferenceStatus.NO_MODEL, error_code=None),
            dict(status=InferenceStatus.RUNTIME_ERROR, error_code="E", predictions=(Prediction("c", 0.5),)),
            dict(status=InferenceStatus.OK, model=FIXTURE_MODEL, inference_ms=-1.0),
            dict(status=InferenceStatus.OK, model=FIXTURE_MODEL, inference_ms=math.inf),
            dict(status=InferenceStatus.OK, model=FIXTURE_MODEL, schema_version="9.9"),
        ]
        for kw in bad:
            with self.subTest(kw=kw):
                with self.assertRaises(ContractError):
                    InferenceResult(**{**base, **kw})

    def test_prediction_and_model_validation(self):
        for conf in (-0.01, 1.01, math.nan, True):
            with self.subTest(conf=conf):
                with self.assertRaises(ContractError):
                    Prediction("c", conf)
        with self.assertRaises(ContractError):
            Prediction("", 0.5)
        with self.assertRaises(ContractError):
            Prediction("c", 0.5, bbox=(1, 2, 3))
        with self.assertRaises(ContractError):
            ModelRef("m", "1", "XYZ")
        with self.assertRaises(ContractError):
            ModelRef("m", "1", "A" * 64)  # uppercase not canonical
        with self.assertRaises(ContractError):
            InputDescriptor(-1, 1)
        for bad in ({"id": None, "version": "1", "sha256": "0" * 64},
                    {"id": "m", "version": None, "sha256": "0" * 64},
                    {"id": "m", "version": "1", "sha256": "0" * 64 + "\n"}):
            with self.subTest(bad=bad):
                with self.assertRaises(ContractError):
                    ModelRef.from_dict(bad)

    def test_from_dict_rejects_malformed(self):
        good = ok_result().to_dict()
        for mutate in (
            lambda d: d.pop("status"),
            lambda d: d.update(status="MAYBE"),
            lambda d: d["input"].update(quality="GREAT"),
            lambda d: d.update(model={"id": "m"}),
        ):
            d = json.loads(json.dumps(good))
            mutate(d)
            with self.subTest(d=d):
                with self.assertRaises(ContractError):
                    InferenceResult.from_dict(d)

    def test_usable_requires_ok_model_and_valid_input(self):
        self.assertTrue(ok_result().is_usable)
        self.assertFalse(ok_result(ImageQuality.UNKNOWN).is_usable)


if __name__ == "__main__":
    unittest.main()
