"""Quality envelope 0.2.0 (owner decision 19/11/2026, QG-003): provisional displacement tolerance
`max_object_outside_px = 15` (0.1.0 had none: 0 px). Audited as a version change; an envelope
pinned to 0.1.0 keeps exactly its 0.1.0 limits and hash. Images are generated here (no camera)."""

import unittest

import cv2
import numpy as np

from acurix_edge.benchview import synthetic as syn
from acurix_edge.config.settings import ConfigError, QualityConfig, envelope_fingerprint, load_config
from acurix_edge.vision.label_skew import SkewParams, measure
from acurix_edge.vision.quality_gate import QualityGate
from tests.support import TempDirTestCase

# envelope_fingerprint(QualityConfig()) computed with the code of the previous commit (envelope 0.1.0)
HASH_0_1_0 = "c5da2f53cde35cc304d46d01da86f411f8e7ba1213efb78a3658a5dedebf9db2"
ROI_RIGHT = 96 + 448  # 640x480, roi_x 0.15, roi_w 0.70


def scene(outside_px: int):
    """Template scene whose box sticks out of the ROI's right edge by `outside_px` (0 = inside)."""
    rng = np.random.default_rng(3)
    img = np.full((480, 640), 90, np.uint8)
    right = ROI_RIGHT + outside_px if outside_px else ROI_RIGHT - 60
    cv2.rectangle(img, (right - 300, 130), (right, 350), 175, -1)
    noisy = np.clip(img.astype(np.int16) + rng.normal(0, 4, img.shape).astype(np.int16), 0, 255).astype(np.uint8)
    return cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR)


class EnvelopeVersionTests(TempDirTestCase):
    def test_new_default_is_0_2_0_provisional_with_15_px(self):
        q = QualityConfig()
        self.assertEqual((q.version, q.status, q.max_object_outside_px), ("0.2.0", "provisional", 15.0))
        self.assertNotEqual(envelope_fingerprint(q)[1], HASH_0_1_0)

    def test_a_file_pinned_to_0_1_0_keeps_its_limits_and_its_hash(self):
        p = self.tmp / "old.toml"
        p.write_text('[quality]\nversion = "0.1.0"\n', encoding="utf-8")
        q = load_config(p).quality
        self.assertEqual(q.max_object_outside_px, 0.0, "0.1.0 had no displacement tolerance")
        self.assertEqual(envelope_fingerprint(q), ("0.1.0", HASH_0_1_0), "same version, same limits, same hash")

    def test_0_1_0_with_a_new_limit_is_refused_by_the_audit(self):  # review 004, B4
        from acurix_edge.app import Application

        p = self.tmp / "old.toml"
        p.write_text('[quality]\nversion = "0.1.0"\n', encoding="utf-8")
        a = Application(load_config(p), self.tmp)
        a.boot()
        a.shutdown()
        p.write_text('[quality]\nversion = "0.1.0"\nmax_object_outside_px = 15.0\n', encoding="utf-8")
        a = Application(load_config(p), self.tmp)  # same version, different limits -> QG-003 refusal
        a.boot()
        try:
            audits = [r for r in a.store.list_audit(50) if r["action"].startswith("QUALITY_ENVELOPE")]
            self.assertEqual(audits[0]["action"], "QUALITY_ENVELOPE_REJECTED")
            self.assertIsNotNone(a._envelope_conflict)
        finally:
            a.shutdown()

    def test_code_built_0_1_0_config_needs_the_legacy_value(self):  # review 004, B1 (documented trap)
        built = QualityConfig(version="0.1.0")  # NOT through load_config: gets the 0.2.0 default (15 px)
        self.assertEqual(built.max_object_outside_px, 15.0)
        self.assertNotEqual(envelope_fingerprint(built)[1], HASH_0_1_0, "would be REJECTED, never accepted")
        legacy = QualityConfig(version="0.1.0", max_object_outside_px=0.0)
        self.assertEqual(envelope_fingerprint(legacy), ("0.1.0", HASH_0_1_0))

    def test_explicit_value_and_bounds(self):
        p = self.tmp / "q.toml"
        p.write_text('[quality]\nversion = "0.3.0"\nmax_object_outside_px = 25\n', encoding="utf-8")
        self.assertEqual(load_config(p).quality.max_object_outside_px, 25)
        for bad in ("-1", "101", '"15"', "true"):
            with self.subTest(value=bad):
                p.write_text(f'[quality]\nversion = "0.3.0"\nmax_object_outside_px = {bad}\n', encoding="utf-8")
                with self.assertRaises(ConfigError):
                    load_config(p)

    def test_upgrade_is_audited_as_an_envelope_change(self):
        from acurix_edge.app import Application
        from acurix_edge.config.settings import AppConfig

        p = self.tmp / "old.toml"
        p.write_text('[quality]\nversion = "0.1.0"\n', encoding="utf-8")
        old = Application(load_config(p), self.tmp)
        old.boot()
        old.shutdown()
        new = Application(AppConfig(), self.tmp)  # default = 0.2.0
        new.boot()
        try:
            audits = [r for r in new.store.list_audit(50) if r["action"].startswith("QUALITY_ENVELOPE")]
            self.assertEqual(audits[0]["action"], "QUALITY_ENVELOPE_CHANGED")
            self.assertEqual(audits[0]["target"], "quality@0.2.0")
            self.assertIsNone(new._envelope_conflict)
        finally:
            new.shutdown()


class GateToleranceTests(unittest.TestCase):
    def status(self, outside_px, q=None):
        r = QualityGate(q or QualityConfig()).evaluate(scene(outside_px))
        return r, [x.code for x in r.reasons], r.metrics["object_outside_roi_px"]

    def test_15_px_tolerance(self):
        r, codes, out = self.status(0)
        self.assertEqual((r.status.value, out), ("VALID", 0.0))
        r, codes, out = self.status(10)
        self.assertEqual(r.status.value, "VALID", codes)
        self.assertAlmostEqual(out, 10, delta=1)
        r, codes, out = self.status(30)
        self.assertIn("OBJECT_OUTSIDE_ROI", codes)
        reason = next(x for x in r.reasons if x.code == "OBJECT_OUTSIDE_ROI")
        self.assertEqual(reason.limit, 15.0)

    def test_exact_boundary_15_passes_16_is_refused(self):  # review 004, B4 (limit is strict: > 15)
        r, codes, out = self.status(14)  # the drawn box is inclusive: 14 -> measured 15 px
        self.assertEqual(out, 15.0)
        self.assertNotIn("OBJECT_OUTSIDE_ROI", codes)
        r, codes, out = self.status(15)
        self.assertEqual(out, 16.0)
        self.assertIn("OBJECT_OUTSIDE_ROI", codes)

    def test_0_1_0_envelope_still_refuses_any_displacement(self):
        _, codes, _ = self.status(10, QualityConfig(version="0.1.0", max_object_outside_px=0.0))
        self.assertIn("OBJECT_OUTSIDE_ROI", codes)


class SyntheticSceneAngleTests(unittest.TestCase):
    """The anti-aliased label keeps the specialist's angle measurement right (owner item 5)."""

    PARAMS = SkewParams(status="provisional", skew_tolerance_deg=5.0, borderline_low_deg=3.0, object_delta=30.0,
                        label_delta=40.0, product_erode_px=3, min_label_area_fraction=0.03,
                        max_label_area_fraction=0.85, min_label_elongation=1.4, product_square_elongation=1.15)

    def test_measured_angle_matches_the_drawn_angle(self):
        rng = np.random.default_rng(7)
        for t in (0.0, 2.0, 9.0, 12.0, 15.0):
            drawn, _ = syn.scene_angle_deg(t)
            with self.subTest(t=t, drawn=drawn):
                m = measure(syn.render_scene(t, rng), self.PARAMS)
                self.assertNotIsInstance(m, str, m)
                self.assertAlmostEqual(abs(m.skew_deg), abs(drawn), delta=0.3)


if __name__ == "__main__":
    unittest.main()
