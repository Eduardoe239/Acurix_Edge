"""AEX-CV-007 part 1 — QG-001..003, CAS-005, DEC-001 and the geometric label-skew specialist.

All images are synthetic (tests/vision_support.py). Nothing here is a trained model, a real
image or an approved rule: approval stays default-deny (ADR-016)."""

import dataclasses
import json
import random
import shutil
import unittest
from typing import Any

from acurix_edge.app import Application, SessionError
from acurix_edge.capture.frames import FrameStamper
from acurix_edge.config.settings import (
    DEFAULT_ARTIFACTS,
    AppConfig,
    ConfigError,
    ModelsConfig,
    QualityConfig,
    load_config,
)
from acurix_edge.contracts.decision import InspectionResult
from acurix_edge.contracts.errors import ErrorCode
from acurix_edge.contracts.inference import (
    INFERENCE_SCHEMA_VERSION,
    ImageQuality,
    InferenceRequest,
    InferenceResult,
    InferenceStatus,
    InputDescriptor,
    ModelRef,
    Prediction,
)
from acurix_edge.contracts.quality import Comparison, QualityStatus
from acurix_edge.domain.artifacts import PACKAGE_ARTIFACTS_DIR, ArtifactRejected, ArtifactVerifier, RejectReason
from acurix_edge.domain.model_policy import ModelPolicy
from acurix_edge.domain.stages import UnavailableStage, guarded_run
from acurix_edge.health.service import ComponentStatus as C
from acurix_edge.vision.label_skew import CLASS_OK, CLASS_SKEWED, LabelSkewSpecialist, SkewParams, measure
from acurix_edge.vision.quality_gate import QualityGate
from tests.support import (
    FIXTURE_MODEL,
    FIXTURE_POLICY,
    FIXTURE_RUNTIME,
    HEALTHY_HOST,
    FixtureApprovedRule,
    FixtureOkAdapter,
    TempDirTestCase,
)
from tests.vision_support import GOOD, Scene, render

SKEW_ID, SKEW_VERSION = "label-skew-geometric", "0.1.0"
ANGLES = (0.0, 2.0, 3.0, 4.9, 5.1, 8.0, 15.0)
MAX_ERROR_DEG = 0.5  # Ordem 4 target on the fixtures


def verifier(root=PACKAGE_ARTIFACTS_DIR, inventory=DEFAULT_ARTIFACTS, incidents=None) -> ArtifactVerifier:
    return ArtifactVerifier(inventory, ModelPolicy(), root=root,
                            on_incident=(incidents.append if incidents is not None else None))


def specialist() -> LabelSkewSpecialist:
    return LabelSkewSpecialist(verifier().load(SKEW_ID, SKEW_VERSION))


def frame(scene: Scene, stamper: FrameStamper | None = None):
    return (stamper or FrameStamper("synthetic")).stamp(render(scene), 0.0)


def request_for(scene: Scene, inspection_id: str = "insp-1") -> InferenceRequest:
    img = render(scene)
    return InferenceRequest(inspection_id, InputDescriptor(img.shape[1], img.shape[0], ImageQuality.VALID), frame=img)


# -- QG-001 / QG-002: the gate itself --------------------------------------------------------------
class QualityGateTests(unittest.TestCase):
    gate = QualityGate(QualityConfig())
    CASES: dict[str, tuple[dict[str, Any], str, str]] = {
        "blurred": ({"blur_sigma": 4.0}, "FOCUS_LOW", "sharpness_laplacian_var"),
        "dark": ({"gain": 0.3}, "UNDEREXPOSED", "brightness_mean"),
        "overexposed": ({"gain": 2.4}, "OVEREXPOSED", "brightness_mean"),
        "reflection": ({"reflection_radius": 25.0}, "REFLECTION", "specular_fraction_roi"),
        "outside_template": ({"center": (60.0, 120.0)}, "OBJECT_OUTSIDE_ROI", "object_outside_roi_px"),
        "no_product": ({"product_bgr": (90, 90, 90), "label": False}, "OBJECT_NOT_FOUND",
                       "object_area_fraction_roi"),
        # review M2: wrong framing must not pass as VALID
        "camera_too_close": ({"product_size": (420.0, 330.0)}, "BACKGROUND_OUT_OF_RANGE", "background_offset"),
        "clutter_on_border": ({"clutter": True}, "BACKGROUND_NOT_UNIFORM", "border_outlier_fraction"),
        "all_black": ({"gain": 0.0}, "UNDEREXPOSED", "brightness_mean"),
        # review M3: product rotated beyond the position envelope (Anexo 04A §3.3, ±10°)
        "rotated_product": ({"product_angle": 20.0}, "OBJECT_ROTATED", "object_rotation_abs_deg"),
        "rotated_product_neg": ({"product_angle": -12.0}, "OBJECT_ROTATED", "object_rotation_abs_deg"),
    }

    def test_good_fixtures_pass(self):
        for pa in (0.0, 7.0, -9.0):
            for sk in (0.0, 8.0, -15.0):
                with self.subTest(product_angle=pa, skew=sk):
                    r = self.gate.evaluate(render(GOOD.with_(product_angle=pa, skew=sk, seed=int(pa + sk) + 50)))
                    self.assertIs(r.status, QualityStatus.VALID, [x.to_dict() for x in r.reasons])
                    self.assertEqual(r.reasons, ())

    def test_bad_fixtures_rejected_with_the_right_reason(self):
        for name, (kw, code, metric) in self.CASES.items():
            with self.subTest(case=name):
                r = self.gate.evaluate(render(GOOD.with_(**kw)))
                self.assertIs(r.status, QualityStatus.INPUT_INVALID)
                reason = next((x for x in r.reasons if x.code == code), None)
                self.assertIsNotNone(reason, f"{name}: got {[x.code for x in r.reasons]}")
                assert reason is not None
                self.assertEqual(reason.metric, metric)
                self.assertEqual(reason.value, r.metrics[metric])
                if reason.comparison is Comparison.BELOW_MIN or reason.comparison is Comparison.NOT_FOUND:
                    self.assertLess(reason.value, reason.limit)
                else:
                    self.assertGreater(reason.value, reason.limit)

    def test_metrics_recorded_for_every_frame_and_envelope_never_validated(self):
        for scene in (GOOD, GOOD.with_(blur_sigma=4.0)):
            r = self.gate.evaluate(render(scene))
            for key in ("sharpness_laplacian_var", "brightness_mean", "dark_fraction", "bright_fraction",
                        "specular_fraction_roi", "object_area_fraction_roi", "object_outside_roi_px",
                        "background_level", "background_offset", "border_outlier_fraction",
                        "object_rotation_deg"):
                self.assertIn(key, r.metrics)
            self.assertEqual((r.envelope_version, r.envelope_status, r.envelope_validated),
                             ("0.1.0", "provisional", False))
            self.assertGreaterEqual(r.elapsed_ms, 0.0)
            json.dumps(r.to_dict(), allow_nan=False)

    def test_grayscale_input_accepted(self):
        import cv2

        gray = cv2.cvtColor(render(GOOD), cv2.COLOR_BGR2GRAY)
        self.assertIs(self.gate.evaluate(gray).status, QualityStatus.VALID)


# -- QG-003: versioned, schema-validated envelope --------------------------------------------------
class EnvelopeConfigTests(TempDirTestCase):
    def load(self, body: str):
        p = self.tmp / "c.toml"
        p.write_text(body, encoding="utf-8")
        return load_config(p)

    def test_example_matches_defaults_and_is_provisional(self):
        cfg = load_config(None)
        self.assertEqual(cfg.quality.status, "provisional")
        self.assertEqual(cfg.models.approved_models, ())  # ADR-016 lock intact

    def test_invalid_envelopes_refused(self):
        for body in ('[quality]\nstatus = "validated"\n', '[quality]\nversion = "v1"\n',
                     "[quality]\nroi_x = 0.5\nroi_w = 0.6\n", "[quality]\nmin_brightness = 220.0\n",
                     "[quality]\nmax_specular_fraction = 1.5\n", "[quality]\nunknown_limit = 1\n"):
            with self.subTest(body=body), self.assertRaises(ConfigError):
                self.load(body)


class EnvelopeAppCase(TempDirTestCase):
    """Helpers only (no tests)."""

    def app(self, cfg: AppConfig) -> Application:
        a = Application(cfg, self.tmp, capture_probe=lambda: (C.OK, "fixture"), host_checks=HEALTHY_HOST)
        a.boot()
        self.addCleanup(a.shutdown)
        return a

    def actions(self, a):
        return [r for r in a.store.list_audit(200) if r["action"].startswith("QUALITY_ENVELOPE")]


class EnvelopeAuditTests(EnvelopeAppCase):
    def test_activation_change_and_same_version_conflict(self):
        base = AppConfig()
        a = self.app(base)
        first = self.actions(a)
        self.assertEqual([r["action"] for r in first], ["QUALITY_ENVELOPE_ACTIVATED"])
        self.assertEqual(first[0]["detail"]["version"], "0.1.0")
        self.assertFalse(first[0]["detail"]["validated"])
        a.shutdown()
        a = self.app(base)  # same envelope: nothing new
        self.assertEqual(len(self.actions(a)), 1)
        a.shutdown()
        changed = dataclasses.replace(base, quality=dataclasses.replace(base.quality, version="0.1.1",
                                                                         min_sharpness=55.0))
        a = self.app(changed)
        self.assertEqual(self.actions(a)[0]["action"], "QUALITY_ENVELOPE_CHANGED")
        self.assertEqual(self.actions(a)[0]["detail"]["previous"]["version"], "0.1.0")
        a.shutdown()
        # limit changed WITHOUT a new version -> refused, visible in health, frames refused
        sneaky = dataclasses.replace(changed, quality=dataclasses.replace(changed.quality, min_sharpness=10.0))
        a = self.app(sneaky)
        self.assertEqual(self.actions(a)[0]["action"], "QUALITY_ENVELOPE_REJECTED")
        env = next(r for r in a.status()["resources"] if r["component"] == "quality_envelope")
        self.assertEqual((env["status"], env["code"]), ("DEGRADED", "AEX-CFG-001"))
        a.start_session()
        with self.assertRaises(SessionError) as ctx:
            a.process_frame(frame(GOOD))
        self.assertIs(ctx.exception.code, ErrorCode.CFG_001)

    def test_health_says_provisional(self):
        env = next(r for r in self.app(AppConfig()).status()["resources"] if r["component"] == "quality_envelope")
        self.assertEqual(env["status"], "OK")
        self.assertIn("PROVISIONAL", env["detail"])
        self.assertIn("NOT validated", env["detail"])

    def test_version_cannot_be_reused_through_an_intermediate_version(self):  # review M1: A -> B -> A'
        base = AppConfig()

        def with_q(**kw):
            return dataclasses.replace(base, quality=dataclasses.replace(base.quality, **kw))

        for cfg in (base, with_q(version="0.1.1", min_sharpness=55.0)):
            self.app(cfg).shutdown()
        a = self.app(with_q(min_sharpness=12.0))  # back to 0.1.0 but with different limits
        self.assertEqual(self.actions(a)[0]["action"], "QUALITY_ENVELOPE_REJECTED")
        self.assertIsNotNone(a._envelope_conflict)
        a.shutdown()
        a = self.app(base)  # the ORIGINAL 0.1.0 limits are still accepted (recorded as a change back)
        self.assertIsNone(a._envelope_conflict)
        self.assertEqual(self.actions(a)[0]["action"], "QUALITY_ENVELOPE_CHANGED")
        self.assertEqual(set(self.actions(a)[0]["detail"]["previous"]), {"version", "sha256"})  # review L2

    def test_unreadable_envelope_history_refuses_the_envelope(self):  # review L1
        from unittest import mock

        from acurix_edge.persistence.sqlite_store import PersistenceError, SqliteEventStore

        with mock.patch.object(SqliteEventStore, "last_audit_with_action",
                               side_effect=PersistenceError("read failed")):
            a = self.app(AppConfig())
        env = next(r for r in a.status()["resources"] if r["component"] == "quality_envelope")
        self.assertEqual((env["status"], env["code"]), ("DEGRADED", "AEX-CFG-001"))


# -- CAS-005: verified artifacts; the inventory never approves -------------------------------------
class ArtifactVerificationTests(TempDirTestCase):
    def copy_root(self):
        root = self.tmp / "artifacts"
        root.mkdir()
        shutil.copy(PACKAGE_ARTIFACTS_DIR / "label-skew-geometric-0.1.0.json", root)
        return root

    def test_correct_hash_loads_but_is_not_approved(self):
        art = verifier().load(SKEW_ID, SKEW_VERSION)
        self.assertEqual(art.ref.sha256, DEFAULT_ARTIFACTS[0]["sha256"])
        self.assertFalse(art.approved)

    def test_divergent_missing_and_unlisted_are_refused_with_incident(self):
        root = self.copy_root()
        (root / "label-skew-geometric-0.1.0.json").write_bytes(
            (root / "label-skew-geometric-0.1.0.json").read_bytes().replace(b"5.0", b"9.0"))
        cases = [(root, SKEW_ID, SKEW_VERSION, RejectReason.HASH_MISMATCH, ErrorCode.CAS_003),
                 (self.tmp / "empty", SKEW_ID, SKEW_VERSION, RejectReason.MISSING, ErrorCode.CAS_003),
                 (PACKAGE_ARTIFACTS_DIR, "other-model", "1.0.0", RejectReason.NOT_LISTED, ErrorCode.CFG_002),
                 (PACKAGE_ARTIFACTS_DIR, SKEW_ID, "9.9.9", RejectReason.NOT_LISTED, ErrorCode.CFG_002)]
        for r, aid, ver, reason, code in cases:
            with self.subTest(reason=reason, artifact=f"{aid}@{ver}"):
                incidents: list[ArtifactRejected] = []
                with self.assertRaises(ArtifactRejected) as ctx:
                    verifier(root=r, incidents=incidents).load(aid, ver)
                self.assertEqual((ctx.exception.reason, ctx.exception.code), (reason, code))
                self.assertEqual(len(incidents), 1)

    def test_rejected_artifact_makes_the_stage_unavailable_and_is_audited(self):
        root = self.copy_root()
        (root / "label-skew-geometric-0.1.0.json").write_bytes(b"{}")
        a = Application(AppConfig(), self.tmp, capture_probe=lambda: (C.OK, "fixture"), host_checks=HEALTHY_HOST)
        a.boot()
        self.addCleanup(a.shutdown)
        a.artifacts = ArtifactVerifier(a.config.models.artifacts, a.model_policy, root=root,
                                       on_incident=a._on_artifact_incident)
        stage = a.load_label_skew_stage()
        self.assertIsInstance(stage, UnavailableStage)
        self.assertFalse(stage.approved)
        audit = [r for r in a.store.list_audit(50) if r["action"] == "ARTIFACT_REJECTED"]
        self.assertEqual(audit[0]["detail"]["reason"], "HASH_MISMATCH")
        a.start_session()
        out = a.process_frame(frame(GOOD), stage=stage)
        self.assertEqual(out.result, "NO_DECISION")

    def test_inventory_entry_is_never_treated_as_approved(self):  # owner rule (c)
        cfg = load_config(None)
        listed = cfg.models.artifacts[0]
        ref = ModelRef(listed["id"], listed["version"], listed["sha256"])
        policy = ModelPolicy.from_config(cfg)
        self.assertFalse(policy.is_model_approved(ref))
        self.assertEqual(policy.approved_models, frozenset())
        # even a policy that allows real loading does not approve what is only inventoried
        loading = ModelPolicy(approved_models=frozenset(), real_model_loading=True)
        self.assertFalse(loading.is_model_approved(ref))
        self.assertFalse(ArtifactVerifier(cfg.models.artifacts, loading).load(SKEW_ID, SKEW_VERSION).approved)
        self.assertFalse(specialist().approved)

    def test_inventory_schema_only_accepts_not_approved(self):  # owner rule (a)
        entry = dict(DEFAULT_ARTIFACTS[0])
        for status in ("APPROVED", "approved", "PENDING", "", None):
            with self.subTest(status=status), self.assertRaises(ConfigError):
                from acurix_edge.config.settings import validate

                validate(dataclasses.replace(AppConfig(), models=ModelsConfig(artifacts=({**entry,
                                                                                          "status": status},))))
        for bad in ({**entry, "path": "../x.json"}, {**entry, "sha256": "ABC"}, {k: v for k, v in entry.items()
                                                                                 if k != "path"}):
            with self.subTest(bad=bad), self.assertRaises(ConfigError):
                from acurix_edge.config.settings import validate

                validate(dataclasses.replace(AppConfig(), models=ModelsConfig(artifacts=(bad,))))


# -- the specialist ---------------------------------------------------------------------------------
class LabelSkewMeasurementTests(unittest.TestCase):
    params = specialist().params
    errors: list[tuple[float, float, float, float, str]] = []  # (product angle, true, measured, error, class)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.errors:
            worst = max(cls.errors, key=lambda e: e[3])
            print(f"\n[CV-007] skew fixtures: {len(cls.errors)} measured, worst error {worst[3]:.3f} deg "
                  f"(true {worst[1]}, measured {worst[2]:.3f}, product {worst[0]})")

    def test_angles_both_directions_within_error_budget(self):
        for pa in (0.0, 12.0, -20.0):
            for true in ANGLES:
                for sign in (1, -1):
                    skew = sign * true
                    with self.subTest(product_angle=pa, skew=skew):
                        m = measure(render(GOOD.with_(product_angle=pa, skew=skew, seed=int(10 * skew) + 1000)),
                                    self.params)
                        self.assertNotIsInstance(m, str)
                        assert not isinstance(m, str)
                        err = abs(m.skew_deg - skew)
                        cls = CLASS_SKEWED if abs(m.skew_deg) > 5.0 else CLASS_OK
                        type(self).errors.append((pa, skew, m.skew_deg, err, cls))
                        self.assertLessEqual(err, MAX_ERROR_DEG)
                        # class is guaranteed only outside the measurement error band around 5 deg
                        if abs(true - 5.0) > MAX_ERROR_DEG:
                            self.assertEqual(cls, CLASS_SKEWED if true > 5.0 else CLASS_OK)

    def test_rotated_product_with_aligned_label_is_not_skewed(self):
        for pa in (25.0, -35.0, 40.0, 90.0):
            with self.subTest(product_angle=pa):
                out = specialist().run(request_for(GOOD.with_(product_angle=pa, skew=0.0)))
                self.assertIs(out.status, InferenceStatus.OK)
                self.assertEqual(out.predictions[0].class_id, CLASS_OK)
                self.assertLess(abs(float(out.predictions[0].attributes["angle_deg"])), MAX_ERROR_DEG)

    def test_output_contract_borderline_and_technical_confidence(self):
        sp = specialist()
        for skew, cls, borderline in ((0.0, CLASS_OK, False), (4.0, CLASS_OK, True), (-4.0, CLASS_OK, True),
                                      (8.0, CLASS_SKEWED, False), (-15.0, CLASS_SKEWED, False)):
            with self.subTest(skew=skew):
                out = sp.run(request_for(GOOD.with_(skew=skew)))
                self.assertEqual(out.schema_version, INFERENCE_SCHEMA_VERSION)
                self.assertEqual(out.model, sp.model)
                p = out.predictions[0]
                self.assertEqual((p.class_id, p.attributes["borderline"]), (cls, borderline))
                self.assertTrue(0.0 <= p.confidence <= 1.0)
                self.assertEqual(p.attributes["confidence_kind"], "technical_geometric_not_probability")
                self.assertEqual(p.attributes["params_status"], "provisional")
                self.assertGreater(out.inference_ms, 0.0)
                self.assertEqual(InferenceResult.from_dict(json.loads(json.dumps(out.to_dict()))), out)

    def test_no_label_or_ambiguous_label_is_inconclusive(self):
        sp = specialist()
        for scene, code in ((GOOD.with_(label=False), "LABEL_NOT_FOUND"),
                            (GOOD.with_(label_size=(50.0, 50.0)), "LABEL_SHAPE_AMBIGUOUS"),
                            (GOOD.with_(product_shape="ellipse"), "PRODUCT_SHAPE_AMBIGUOUS")):  # review L8
            with self.subTest(code=code):
                out = sp.run(request_for(scene))
                self.assertIs(out.status, InferenceStatus.INCONCLUSIVE)
                self.assertEqual(out.error_code, code)
                self.assertEqual(out.predictions, ())

    def test_params_must_match_the_verified_artifact(self):
        art = verifier().load(SKEW_ID, SKEW_VERSION)
        from acurix_edge.vision.label_skew import ParamsError

        for key, value in (("skew_tolerance_deg", 50.0), ("status", "approved"), ("product_erode_px", "3"),
                           ("product_erode_px", 500), ("id", "other")):  # review L7
            with self.subTest(key=key, value=value):
                raw = json.loads(art.data)
                raw[key] = value
                bad = dataclasses.replace(art, data=json.dumps(raw).encode())
                with self.assertRaises(ParamsError):
                    SkewParams.from_artifact(bad)


# -- integration + properties (section 5) -----------------------------------------------------------
class SpyStage:
    """Counts calls; returns an OK fixture result (as if it were an approved model)."""

    stage_id, stage_kind = "specialist:spy", "specialist"

    def __init__(self, approved: bool) -> None:
        self.calls = 0
        self._approved = approved

    @property
    def model(self) -> ModelRef:
        return FIXTURE_MODEL

    @property
    def approved(self) -> bool:
        return self._approved

    def run(self, request: InferenceRequest) -> InferenceResult:
        self.calls += 1
        return InferenceResult(inspection_id=request.inspection_id, runtime=FIXTURE_RUNTIME, input=request.input,
                               model=FIXTURE_MODEL, status=InferenceStatus.OK,
                               predictions=(Prediction("fixture_class", 0.99),), inference_ms=0.1)


def random_scene(rng: random.Random) -> Scene:
    kw: dict[str, Any] = {"product_angle": rng.uniform(-30, 30), "skew": rng.choice([0.0, 2.0, 4.0, 6.0, 12.0,
                                                                                      -3.0, -9.0]),
                          "seed": rng.randrange(10_000)}
    if rng.random() < 0.5:
        kw[rng.choice(["blur_sigma", "gain", "reflection_radius", "center", "label"])] = None
    for key in [k for k, v in kw.items() if v is None]:
        kw[key] = {"blur_sigma": rng.uniform(2.5, 6), "gain": rng.choice([0.25, 0.35, 2.3, 2.8]),
                   "reflection_radius": rng.uniform(22, 35), "center": (rng.uniform(20, 70), 120.0),
                   "label": False}[key]
    return GOOD.with_(**kw)


class FrameAppCase(TempDirTestCase):
    """Helpers only (no tests)."""

    def app(self, **kw) -> Application:
        a = Application(AppConfig(), self.tmp, capture_probe=lambda: (C.OK, "fixture"), host_checks=HEALTHY_HOST,
                        **kw)
        a.boot()
        self.addCleanup(a.shutdown)
        a.start_session()
        return a

    def event(self, a, outcome):
        return a.store.get(outcome.record.event_id)


class FrameIntegrationTests(FrameAppCase):
    def test_invalid_frame_is_recorded_as_no_decision_and_no_stage_runs(self):
        a = self.app()
        spy = SpyStage(approved=False)
        out = a.process_frame(frame(GOOD.with_(blur_sigma=5.0)), stage=spy)
        self.assertEqual((out.result, out.reason), ("NO_DECISION", "INPUT_QUALITY"))
        self.assertEqual(spy.calls, 0)
        ev = self.event(a, out)
        self.assertEqual(ev.inference_status, "INPUT_INVALID")
        self.assertEqual(ev.extra["quality"]["status"], "INPUT_INVALID")
        self.assertEqual(ev.extra["quality"]["reasons"][0]["code"], "FOCUS_LOW")
        self.assertEqual(ev.extra["quality"]["envelope_version"], "0.1.0")
        self.assertNotIn("stages", ev.extra)

    def test_valid_frame_runs_the_unapproved_specialist_in_shadow_and_records_it(self):
        a = self.app()
        out = a.process_frame(frame(GOOD.with_(skew=8.0)), stage=a.load_label_skew_stage())
        self.assertEqual(out.result, "NO_DECISION")
        ev = self.event(a, out)
        self.assertEqual(ev.extra["quality"]["status"], "VALID")
        self.assertFalse(ev.extra["quality"]["envelope_validated"])
        st = ev.extra["stages"][0]
        self.assertEqual((st["stage_id"], st["approved"], st["shadow"]),
                         ("specialist:label-skew-geometric", False, True))
        self.assertEqual(st["result"]["predictions"][0]["class_id"], CLASS_SKEWED)
        self.assertEqual(st["decision_input_error"], "MODEL_NOT_APPROVED")
        self.assertIsNone(ev.model)  # nothing unapproved is presented as the deciding model

    def test_property_invalid_frames_never_reach_a_stage(self):
        a = self.app()
        rng = random.Random(7)
        stamper = FrameStamper("synthetic")
        seen_invalid = 0
        for _ in range(60):
            # an APPROVED spy is always called on a valid frame, so "never called on invalid" is meaningful
            # (an unapproved, non-inventoried spy would never be called at all: shadow condition (a))
            spy = SpyStage(approved=True)
            scene = random_scene(rng)
            out = a.process_frame(frame(scene, stamper), stage=spy)
            ev = self.event(a, out)
            if ev.extra["quality"]["status"] == "INPUT_INVALID":
                seen_invalid += 1
                self.assertEqual(spy.calls, 0, scene)
                self.assertEqual(out.reason, "INPUT_QUALITY")
            else:
                self.assertEqual(spy.calls, 1)
        self.assertGreater(seen_invalid, 10)

    def test_property_no_pass_or_defect_with_unapproved_specialist(self):
        # Most permissive setting available in tests: approved fixture rule + fixture adapter/policy,
        # healthy host, OBSERVING. The specialist is the real one (unapproved) or a spy claiming OK.
        a = self.app(inference=FixtureOkAdapter(), rule=FixtureApprovedRule(), model_policy=FIXTURE_POLICY)
        rng = random.Random(11)
        stamper = FrameStamper("synthetic")
        stages = [a.load_label_skew_stage(), SpyStage(approved=False)]
        results = set()
        for i in range(80):
            out = a.process_frame(frame(random_scene(rng), stamper), stage=stages[i % 2])
            results.add(out.result)
            self.assertNotIn(out.result, ("PASS", "DEFECT"))
        self.assertEqual(results, {"NO_DECISION"})

    def test_property_unapproved_stage_blocked_even_if_the_envelope_were_validated(self):  # review L11
        """Neutralizes the provisional-envelope downgrade (a 'validated' gate result is simulated) so
        the DEC-001 guard of the stage path is tested on its own."""
        from unittest import mock

        a = self.app(inference=FixtureOkAdapter(), rule=FixtureApprovedRule(), model_policy=FIXTURE_POLICY)
        real = a.quality_gate.evaluate

        def validated(image):
            return dataclasses.replace(real(image), envelope_status="validated", envelope_validated=True)

        rng = random.Random(5)
        stamper = FrameStamper("synthetic")
        with mock.patch.object(a.quality_gate, "evaluate", side_effect=validated):
            for i in range(40):
                scene = GOOD.with_(skew=rng.choice([0.0, 3.0, 8.0, -12.0]), seed=i)
                stage = a.load_label_skew_stage() if i % 2 else SpyStage(approved=False)
                out = a.process_frame(frame(scene, stamper), stage=stage)
                self.assertEqual((out.result, out.reason), ("NO_DECISION", "INFERENCE_NOT_OK"))
                self.assertEqual(self.event(a, out).extra["stages"][0]["decision_input_error"],
                                 "MODEL_NOT_APPROVED")
            # control: with the envelope neutralized, an APPROVED stage does reach a conclusive result,
            # so the assertions above are not vacuous
            ok = a.process_frame(frame(GOOD, stamper), stage=SpyStage(approved=True))
            self.assertEqual(ok.result, "PASS")

    def test_quality_gate_failure_is_recorded_as_input_invalid(self):  # review L5
        from unittest import mock

        a = self.app()
        spy = SpyStage(approved=False)
        with mock.patch.object(a.quality_gate, "evaluate", side_effect=RuntimeError("cv2 broke")):
            out = a.process_frame(frame(GOOD), stage=spy)
        self.assertEqual((out.result, out.reason), ("NO_DECISION", "INPUT_QUALITY"))
        self.assertEqual(spy.calls, 0)
        self.assertEqual(self.event(a, out).extra["quality"]["reasons"][0]["code"], "QUALITY_GATE_ERROR")

    def test_even_an_approved_stage_cannot_pass_with_a_provisional_envelope(self):
        a = self.app(inference=FixtureOkAdapter(), rule=FixtureApprovedRule(), model_policy=FIXTURE_POLICY)
        out = a.process_frame(frame(GOOD), stage=SpyStage(approved=True))
        self.assertEqual((out.result, out.reason), ("NO_DECISION", "INPUT_QUALITY"))
        self.assertIn("provisional", self.event(a, out).extra["decision_detail"])

    def test_guarded_run_contains_misbehaving_stages(self):
        req = request_for(GOOD)

        class Raises(SpyStage):
            def run(self, request):
                raise RuntimeError("boom")

        class WrongType(SpyStage):
            def run(self, request):
                return {"class": "OK"}

        for stage, code in ((Raises(True), "STAGE_EXCEPTION:RuntimeError"),
                            (WrongType(True), "STAGE_INVALID_RETURN_TYPE")):
            with self.subTest(code=code):
                run = guarded_run(stage, req)
                self.assertIsNot(run.for_decision.status, InferenceStatus.OK)
                self.assertEqual(run.raw.error_code, code)


# -- owner decisions 13/11/2026: shadow exception + review round-2 lows + M1-b --------------------
class ShadowModeTests(FrameAppCase):
    def test_a_only_inventoried_not_approved_stages_run_in_shadow(self):
        a = self.app()
        not_listed = SpyStage(approved=False)  # FIXTURE_MODEL is not in the inventory
        out = a.process_frame(frame(GOOD), stage=not_listed)
        self.assertEqual(not_listed.calls, 0)
        st = self.event(a, out).extra["stages"][0]
        self.assertEqual((st["shadow"], st["decision_input_error"]), (False, "MODEL_NOT_APPROVED"))
        self.assertIn("NOT CALLED", st["notice"])

        class ImpostorSpy(SpyStage):  # same id/version as the inventory, different sha256
            @property
            def model(self):
                return ModelRef(SKEW_ID, SKEW_VERSION, "f" * 64)

        impostor = ImpostorSpy(approved=False)
        a.process_frame(frame(GOOD.with_(seed=3)), stage=impostor)
        self.assertEqual(impostor.calls, 0)
        out = a.process_frame(frame(GOOD.with_(seed=4)), stage=a.load_label_skew_stage())
        st = self.event(a, out).extra["stages"][0]
        self.assertEqual((st["shadow"], st["approved"]), (True, False))
        self.assertTrue(st["notice"].startswith("SHADOW"))

    def test_c_property_shadow_result_never_changes_the_decision(self):
        """Same frames, with and without the shadow specialist: identical decision, and the shadow
        output never reaches the event's decision fields (result, reason, class, confidence, model)."""
        with_shadow = self.app()
        base_dir = self.tmp / "baseline"
        base_dir.mkdir()
        baseline = Application(AppConfig(), base_dir, capture_probe=lambda: (C.OK, "fixture"),
                               host_checks=HEALTHY_HOST)
        baseline.boot()
        self.addCleanup(baseline.shutdown)
        baseline.start_session()
        stage = with_shadow.load_label_skew_stage()
        rng = random.Random(23)
        shadow_seen = 0
        for _ in range(50):
            scene = random_scene(rng)
            s = with_shadow.process_frame(frame(scene), stage=stage)
            b = baseline.process_frame(frame(scene))
            self.assertEqual((s.result, s.reason), (b.result, b.reason), scene)
            ev = self.event(with_shadow, s)
            self.assertEqual(ev.result, InspectionResult.NO_DECISION)
            self.assertEqual((ev.class_id, ev.confidence, ev.model, ev.rule), (None, None, None, None))
            if "stages" in ev.extra:
                shadow_seen += 1
                self.assertTrue(ev.extra["stages"][0]["shadow"])
        self.assertGreater(shadow_seen, 5)
        # no PASS/DEFECT ever counted
        persisted = with_shadow.store.list_events(limit=100)
        self.assertFalse(any(e.result in (InspectionResult.PASS, InspectionResult.DEFECT) for e in persisted))


class GateFailureHealthTests(FrameAppCase):  # review round 2, L-d
    def entry(self, a):
        return next(r for r in a.status()["resources"] if r["component"] == "quality_envelope")

    def test_consecutive_gate_failures_degrade_health_and_recover(self):
        from unittest import mock

        a = self.app()
        with mock.patch.object(a.quality_gate, "evaluate", side_effect=RuntimeError("cv2 broke")):
            for _ in range(2):
                a.process_frame(frame(GOOD))
            self.assertEqual(self.entry(a)["status"], "OK")
            a.process_frame(frame(GOOD))
            e = self.entry(a)
            self.assertEqual((e["status"], e["code"]), ("DEGRADED", "AEX-SYS-001"))
            self.assertIn("3 consecutive", e["detail"])
        a.process_frame(frame(GOOD.with_(seed=9)))  # gate works again -> counter reset
        self.assertEqual(self.entry(a)["status"], "OK")


class EnvelopeHistoryTests(EnvelopeAppCase):
    def test_corrupted_history_has_its_own_message(self):  # review round 2, L-c
        a = self.app(AppConfig())
        a.store.audit(actor_ref="system", action="QUALITY_ENVELOPE_ACTIVATED", target="quality@0.1.0",
                      detail={"broken": True})
        a.shutdown()
        a = self.app(AppConfig())
        self.assertIn("corrupted", a._envelope_conflict or "")
        self.assertNotIn("different limits", a._envelope_conflict or "")
        rejected = self.actions(a)[0]
        self.assertEqual((rejected["action"], rejected["detail"]["reason"]),
                         ("QUALITY_ENVELOPE_REJECTED", "HISTORY_CORRUPTED"))

    def test_m1b_envelope_not_audited_at_boot_is_checked_on_reset(self):
        from acurix_edge.persistence.sqlite_store import PersistenceError, SqliteEventStore

        class FlakyStore(SqliteEventStore):
            fail_next_open = True

            def open(self) -> None:
                if FlakyStore.fail_next_open:
                    FlakyStore.fail_next_open = False
                    raise PersistenceError("disk not ready")
                super().open()

        store = FlakyStore(AppConfig().database_path(self.tmp))
        a = Application(AppConfig(), self.tmp, store=store, capture_probe=lambda: (C.OK, "fixture"),
                        host_checks=HEALTHY_HOST)
        a.boot()
        self.addCleanup(a.shutdown)
        from acurix_edge.domain.state_machine import SystemState

        self.assertIs(a.state.state, SystemState.FAULT)
        self.assertIn("not audited", a._envelope_conflict or "")
        a.reset_fault("disk back", actor_ref="owner")
        self.assertIsNone(a._envelope_conflict)
        self.assertEqual(self.actions(a)[0]["action"], "QUALITY_ENVELOPE_ACTIVATED")
        a.start_session()
        out = a.process_frame(frame(GOOD))
        self.assertEqual(out.result, "NO_DECISION")


class BorderAndRotationTests(unittest.TestCase):  # review round 2, H1 + rotation of non-rectangular objects
    def test_border_measured_inside_the_frame_edge(self):
        img = render(GOOD).copy()
        img[:, 0] = 0  # dead first column and first row of a cheap webcam
        img[0, :] = 0
        self.assertIs(QualityGate(QualityConfig()).evaluate(img).status, QualityStatus.VALID)
        edge = QualityGate(QualityConfig(border_inset_px=0)).evaluate(img)
        self.assertIn("BACKGROUND_NOT_UNIFORM", [r.code for r in edge.reasons])

    def test_round_product_rotation_not_evaluated_and_specialist_inconclusive(self):
        for pa in (0.0, 30.0, -40.0):
            with self.subTest(product_angle=pa):
                scene = GOOD.with_(product_shape="ellipse", product_angle=pa, label=False)
                r = QualityGate(QualityConfig()).evaluate(render(scene))
                self.assertNotIn("OBJECT_ROTATED", [x.code for x in r.reasons])
                self.assertEqual(r.metrics["object_rotation_measured"], 0.0)
                self.assertLess(r.metrics["object_rectangularity"], 0.85)
                out = specialist().run(request_for(scene.with_(label=True)))
                self.assertIs(out.status, InferenceStatus.INCONCLUSIVE)
                self.assertEqual(out.error_code, "PRODUCT_SHAPE_AMBIGUOUS")

    def test_rectangular_product_rotation_still_evaluated(self):
        r = QualityGate(QualityConfig()).evaluate(render(GOOD.with_(product_angle=20.0)))
        self.assertEqual(r.metrics["object_rotation_measured"], 1.0)
        self.assertIn("OBJECT_ROTATED", [x.code for x in r.reasons])


if __name__ == "__main__":
    unittest.main()
