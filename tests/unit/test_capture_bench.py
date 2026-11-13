"""AEX-CAPTURE-006 — CAP-005: bench capture tool and manifest (Anexo 04A §3.5, schema 1.0).

Acceptance: one valid manifest per session; sha256 checked against the written bytes; refusal
without the mandatory parameters; a session is never overwritten. Sources are synthetic."""

import contextlib
import copy
import hashlib
import io
import json
import re
import unittest
from datetime import UTC, datetime
from unittest import mock

import cv2
import numpy as np

from acurix_edge.__main__ import main
from acurix_edge.capture.bench import (
    COMPLETE,
    END_OF_STREAM,
    SOURCE_LOST,
    BenchError,
    BenchParams,
    capture_session,
)
from acurix_edge.capture.manifest import (
    MANIFEST_FILE,
    ManifestError,
    validate_manifest,
    verify_session,
)
from acurix_edge.capture.sources import ImageFolderSource, WebcamSource
from tests.capture_support import SyntheticCamera, write_image_folder
from tests.support import TempDirTestCase

PARAMS = BenchParams(sku="SKU-PET-01", unit_id="PET-01-u3", provisional_label="LABEL_SKEWED",
                     lighting_setup="luz-A-difusa-45", lighting_ambient="janela-fechada", pose="p2")
FIXED_NOW = lambda: datetime(2026, 11, 20, 13, 5, 0, tzinfo=UTC)  # noqa: E731
REQUIRED_CLI = {"--sku": "SKU-PET-01", "--unit-id": "PET-01-u3", "--label": "OK", "--lighting": "luz-A"}


class BenchCase(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.root = self.tmp / "datasets" / "local"
        self.originals = write_image_folder(self.tmp / "imgs", 4)

    def folder(self):
        return ImageFolderSource(self.tmp / "imgs")

    def run_capture(self, **kw):
        args = dict(frames=3, now=FIXED_NOW)
        args.update(kw)
        source = args.pop("source", None) or self.folder()
        params = args.pop("params", PARAMS)
        return capture_session(source, self.root, params, **args)

    def session_dirs(self):
        return sorted(p.name for p in self.root.iterdir()) if self.root.exists() else []


class BenchCaptureTests(BenchCase):
    def test_session_manifest_is_valid_and_hashes_match(self):
        r = self.run_capture()
        self.assertEqual((r.end_reason, r.frames_written), (COMPLETE, 3))
        self.assertEqual(r.session_dir.name, "cap-2026-11-20-01")
        self.assertEqual(r.session_dir.parent, self.root.resolve())
        data = json.loads((r.session_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
        validate_manifest(data)
        self.assertEqual(verify_session(r.session_dir), data)
        # exact Anexo 04A §3.5 shape
        self.assertEqual(set(data), {"manifest_schema", "session_id", "captured_by", "captured_at_utc",
                                     "optics_version", "camera", "lighting", "frames"})
        self.assertEqual(data["manifest_schema"], "1.0")
        self.assertEqual(data["captured_at_utc"], "2026-11-20T13:05:00Z")
        self.assertEqual(data["camera"]["resolution"], "64x48")
        self.assertEqual(data["lighting"], {"setup": "luz-A-difusa-45", "ambient": "janela-fechada"})
        self.assertEqual([f["file"] for f in data["frames"]], ["000001.png", "000002.png", "000003.png"])
        for entry, original in zip(data["frames"], self.originals, strict=False):  # 3 written of 4 originals
            path = r.session_dir / entry["file"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry["sha256"])
            decoded = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
            np.testing.assert_array_equal(decoded, original)  # PNG: lossless
            self.assertEqual((entry["sku"], entry["unit_id"], entry["provisional_label"], entry["pose"]),
                             ("SKU-PET-01", "PET-01-u3", "LABEL_SKEWED", "p2"))

    def test_sessions_are_never_overwritten(self):
        first = self.run_capture()
        before = {p.name: p.read_bytes() for p in first.session_dir.iterdir()}
        second = self.run_capture()
        self.assertEqual(second.session_dir.name, "cap-2026-11-20-02")  # next free id
        with self.assertRaises(BenchError) as ctx:
            self.run_capture(session_id="cap-2026-11-20-01")
        self.assertIn("never overwritten", str(ctx.exception))
        self.assertEqual({p.name: p.read_bytes() for p in first.session_dir.iterdir()}, before)
        self.assertEqual(self.session_dirs(), ["cap-2026-11-20-01", "cap-2026-11-20-02"])

    def test_mandatory_parameters_refused(self):
        bad = {"sku": "", "unit_id": " ", "provisional_label": "MAYBE", "lighting_setup": ""}
        for field, value in bad.items():
            with self.subTest(field=field):
                params = BenchParams(**{**{f: getattr(PARAMS, f) for f in PARAMS.__dataclass_fields__},
                                        field: value})
                with self.assertRaises(BenchError):
                    self.run_capture(params=params)
        self.assertEqual(self.session_dirs(), [], "a refused run creates nothing")

    def test_output_only_under_datasets_local(self):
        with self.assertRaises(BenchError):
            capture_session(self.folder(), self.tmp / "elsewhere", PARAMS, frames=1)
        self.assertFalse((self.tmp / "elsewhere").exists())

    def test_unavailable_source_creates_nothing(self):
        with self.assertRaises(BenchError) as ctx:
            self.run_capture(source=ImageFolderSource(self.tmp / "missing"))
        self.assertIn("AEX-CAP-001", str(ctx.exception))
        self.assertEqual(self.session_dirs(), [])

    def test_finite_source_ends_explicitly(self):
        r = self.run_capture(frames=10)
        self.assertEqual((r.end_reason, r.frames_written, r.frames_requested), (END_OF_STREAM, 4, 10))
        verify_session(r.session_dir)

    def test_lost_camera_keeps_written_frames_in_manifest(self):
        cam = SyntheticCamera(frames=2)  # synthetic camera "unplugged" after 2 frames
        r = self.run_capture(source=WebcamSource(0, capture_factory=cam.factory), frames=5)
        self.assertEqual((r.end_reason, r.frames_written), (SOURCE_LOST, 2))
        self.assertEqual(verify_session(r.session_dir)["camera"]["device"], "webcam-usb-0")

    def test_invalid_frame_count(self):
        for frames in (0, -1, 10_001, True):
            with self.subTest(frames=frames), self.assertRaises(BenchError):
                self.run_capture(frames=frames)


class InterruptingSource:
    """Delivers `ok` synthetic frames, then the operator presses Ctrl+C."""

    live = False
    source_id = "interrupting"

    def __init__(self, ok: int) -> None:
        self.ok, self.n, self.is_open = ok, 0, False

    def open(self):
        self.is_open = True

    def read(self):
        from tests.capture_support import synth_image

        if self.n >= self.ok:
            raise KeyboardInterrupt
        self.n += 1
        return synth_image(self.n)

    def close(self):
        self.is_open = False


class BenchReviewRegressionTests(BenchCase):
    def test_m5_interrupted_run_still_lists_written_frames(self):
        src = InterruptingSource(2)
        with self.assertRaises(KeyboardInterrupt):
            self.run_capture(source=src, frames=5)
        self.assertFalse(src.is_open)
        (session,) = [self.root / name for name in self.session_dirs()]
        data = verify_session(session)  # valid manifest, hashes match, no stray .tmp
        self.assertEqual(len(data["frames"]), 2)

    def test_m5_interrupt_before_any_frame_leaves_nothing(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_capture(source=InterruptingSource(0), frames=3)
        self.assertEqual(self.session_dirs(), [])

    def test_l1_frame_renamed_but_not_listed_is_removed_on_interrupt(self):
        real = hashlib.sha256
        calls = {"n": 0}

        def sha_then_interrupt(data):
            calls["n"] += 1
            if calls["n"] == 3:  # frame 3 already renamed to its final name, not yet listed
                raise KeyboardInterrupt
            return real(data)

        with mock.patch("acurix_edge.capture.bench.hashlib.sha256", side_effect=sha_then_interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_capture(frames=4)
        (session,) = [self.root / name for name in self.session_dirs()]
        self.assertEqual(len(verify_session(session)["frames"]), 2)  # no unlisted 000003.png left

    def test_b7_session_ids_that_windows_would_alter_are_refused(self):
        for bad in ("abc.", "CON", "nul.txt", "a b", "../x", ".hidden", "COM0", "lpt0"):
            with self.subTest(bad=bad), self.assertRaises(BenchError):
                self.run_capture(session_id=bad)
        self.assertEqual(self.session_dirs(), [])
        self.run_capture(session_id="bench.v1-a")

    def test_device_label_validated_before_anything_is_created(self):
        with self.assertRaises(BenchError):
            self.run_capture(device_label="x" * 201)
        self.assertEqual(self.session_dirs(), [])


class ManifestValidationTests(BenchCase):
    def setUp(self):
        super().setUp()
        self.result = self.run_capture(frames=2)
        self.data = json.loads((self.result.session_dir / MANIFEST_FILE).read_text(encoding="utf-8"))

    def mutated(self, fn):
        d = copy.deepcopy(self.data)
        fn(d)
        return d

    def test_schema_violations_are_reported(self):
        cases = {
            "missing key": lambda d: d.pop("optics_version"),
            "unknown key": lambda d: d.update(extra=1),
            "wrong schema": lambda d: d.update(manifest_schema="2.0"),
            "camera missing focus": lambda d: d["camera"].pop("focus"),
            "bad resolution": lambda d: d["camera"].update(resolution="64 x 48"),
            "lighting unknown key": lambda d: d["lighting"].update(color="x"),
            "bad sha": lambda d: d["frames"][0].update(sha256="ABC"),
            "bad label": lambda d: d["frames"][0].update(provisional_label="DEFECT"),
            "bad file name": lambda d: d["frames"][0].update(file="../x.png"),
            "duplicated file": lambda d: d["frames"].append(dict(d["frames"][0])),
            "empty frames": lambda d: d.update(frames=[]),
            "bad timestamp": lambda d: d.update(captured_at_utc="2026-11-20 13:05"),
            "multiline note": lambda d: d["frames"][0].update(notes="a\nb"),
            "not an object": lambda d: d["frames"].__setitem__(0, "x"),
        }
        for name, fn in cases.items():
            with self.subTest(case=name), self.assertRaises(ManifestError):
                validate_manifest(self.mutated(fn))
        with self.assertRaises(ManifestError):
            validate_manifest([])

    def test_all_problems_reported_together(self):
        d = self.mutated(lambda d: (d.pop("session_id"), d["frames"][0].update(sha256="x")))
        with self.assertRaises(ManifestError) as ctx:
            validate_manifest(d)
        self.assertGreaterEqual(len(ctx.exception.problems), 2)

    def test_tampered_file_extra_file_and_missing_file_detected(self):
        sd = self.result.session_dir
        first = sd / "000001.png"
        original = first.read_bytes()
        first.write_bytes(original + b"\x00")
        with self.assertRaises(ManifestError) as ctx:
            verify_session(sd)
        self.assertTrue(any("sha256 mismatch" in p for p in ctx.exception.problems))
        first.write_bytes(original)
        (sd / "intruder.png").write_bytes(b"x")
        with self.assertRaises(ManifestError) as ctx:
            verify_session(sd)
        self.assertTrue(any("not in manifest" in p for p in ctx.exception.problems))
        (sd / "intruder.png").unlink()
        (sd / "000002.png").unlink()
        with self.assertRaises(ManifestError):
            verify_session(sd)


class CliTests(BenchCase):
    def cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        code: int | str | None
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = main(list(args))
            except SystemExit as exc:  # argparse refusal
                code = exc.code
        return code, out.getvalue(), err.getvalue()

    def base(self, **override):
        args = ["capture", "--source", "folder", "--path", str(self.tmp / "imgs"), "--frames", "2",
                "--base-dir", str(self.tmp)]
        for k, v in {**REQUIRED_CLI, **override}.items():
            if v is not None:
                args += [k, v]
        return args

    def test_cli_capture_and_verify(self):
        code, out, err = self.cli(*self.base())
        self.assertEqual(code, 0, err)
        result = json.loads(out)
        self.assertEqual((result["frames_written"], result["end_reason"]), (2, "COMPLETE"))
        self.assertRegex(result["session_id"], r"^cap-\d{4}-\d{2}-\d{2}-01$")
        code, out, _ = self.cli("capture-verify", "--session-dir", str(self.root / result["session_id"]))
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(out)["valid"])

    def test_cli_refuses_without_each_mandatory_parameter(self):
        for missing in REQUIRED_CLI:
            with self.subTest(missing=missing):
                code, _, err = self.cli(*self.base(**{missing: None}))
                self.assertEqual(code, 2)
                self.assertIn(missing.lstrip("-"), err.replace("_", "-"))
        code, _, _ = self.cli(*self.base(**{"--label": "PASS"}))  # not an Anexo 04A class
        self.assertEqual(code, 2)
        self.assertEqual(self.session_dirs(), [])

    def test_cli_refuses_inconsistent_source_options(self):
        code, _, err = self.cli("capture", "--source", "webcam", "--path", "x", "--base-dir", str(self.tmp),
                                *[a for kv in REQUIRED_CLI.items() for a in kv])
        self.assertEqual(code, 2)
        self.assertIn("--device", err)

    def test_cli_verify_reports_invalid(self):
        (self.tmp / "s").mkdir()
        code, _, err = self.cli("capture-verify", "--session-dir", str(self.tmp / "s"))
        self.assertEqual(code, 2)
        self.assertIn("manifest.json not found", err)

    def test_existing_session_id_refused_by_cli(self):
        code, out, _ = self.cli(*self.base(), "--session-id", "bench-s1")
        self.assertEqual(code, 0)
        code, _, err = self.cli(*self.base(), "--session-id", "bench-s1")
        self.assertEqual(code, 2)
        self.assertTrue(re.search("never overwritten", err))

    def test_cli_device_label_recorded_in_manifest(self):
        code, out, err = self.cli(*self.base(), "--session-id", "lbl-1", "--device-label",
                                  "Integrated Camera (embutida, teste tecnico)")
        self.assertEqual(code, 0, err)
        data = verify_session(self.root / "lbl-1")
        self.assertEqual(data["camera"]["device"], "Integrated Camera (embutida, teste tecnico)")
        code, _, _ = self.cli(*self.base(), "--session-id", "lbl-2", "--device-label", "x" * 201)
        self.assertEqual(code, 2)  # invalid label refused before anything is created
        self.assertNotIn("lbl-2", self.session_dirs())
        code, _, _ = self.cli(*self.base(), "--session-id", "lbl-3", "--device-label", "")
        self.assertEqual(code, 2)  # explicit empty label is refused, never silently replaced
        self.assertNotIn("lbl-3", self.session_dirs())


if __name__ == "__main__":
    unittest.main()
