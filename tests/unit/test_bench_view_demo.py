"""bench-view --demo-sintetico (owner approval 19/11/2026): simulated source, never a real camera.

Conditions under test: (1) fixed, unhideable "FONTE SIMULADA" banner at the top of the page;
(2) status JSON and recorded events mark the source as simulated; (3) no real camera is ever
opened and no camera profile is required; (4) data in a disposable folder by default."""

import contextlib
import http.client
import io
import json
import re
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from acurix_edge.__main__ import main
from acurix_edge.benchview import synthetic as syn
from acurix_edge.benchview.app_factory import DEMO_MARKER, BenchViewError, build_bench_view
from acurix_edge.benchview.server import HOST, BenchViewServer
from acurix_edge.benchview.view_model import SIMULATED_NOTICE
from acurix_edge.config.settings import QualityConfig
from acurix_edge.vision.quality_gate import QualityGate
from tests.capture_support import SyntheticCamera
from tests.support import HEALTHY_HOST, TempDirTestCase

STATIC_DIR = Path(__file__).resolve().parents[2] / "src" / "acurix_edge" / "benchview" / "static"
NOTICE = "FONTE SIMULADA — imagem gerada por código, não é câmera real"


def wait_until(pred, timeout=5.0, step=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(step)
    return bool(pred())


def no_real_camera():
    """Any attempt to reach the real camera driver fails the test."""
    boom = AssertionError("a real camera was opened")
    return contextlib.ExitStack(), [
        mock.patch("acurix_edge.capture.sources.cv2.VideoCapture", side_effect=boom),
        mock.patch("acurix_edge.capture.sources._default_webcam_factory", side_effect=boom),
    ]


class DemoCase(TempDirTestCase):
    def start_demo(self, config_path=None):
        stack, patches = no_real_camera()
        self.addCleanup(stack.close)
        self.camera_mocks = [stack.enter_context(p) for p in patches]
        self.server: BenchViewServer = build_bench_view(config_path=config_path, base_dir=self.tmp, port=0, fps=15.0,
                                                        stream_width=320, host_checks=HEALTHY_HOST, synthetic=True)
        self.assertTrue(self.server.engine.start(), self.server.engine.problem)
        t = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        t.start()

        def cleanup():
            self.server.engine.stop()
            self.server.shutdown()
            self.server.server_close()

        self.addCleanup(cleanup)
        return self.server

    def get(self, path):
        conn = http.client.HTTPConnection(HOST, self.server.port, timeout=5)
        try:
            conn.request("GET", path)
            resp = conn.getresponse()
            return resp.status, resp.read()
        finally:
            conn.close()

    def status(self):
        return json.loads(self.get("/api/status")[1])


class DemoServerTests(DemoCase):
    def test_never_opens_a_real_camera_and_needs_no_profile(self):
        self.start_demo()  # no --config at all: no camera profile anywhere
        self.assertTrue(wait_until(lambda: self.status()["frames_processed"] >= 3))
        for m in self.camera_mocks:
            m.assert_not_called()
        s = self.status()
        self.assertEqual(s["source"], {"kind": "SIMULATED", "simulated": True, "notice": NOTICE})
        self.assertEqual(s["strip"]["camera"], "OK (simulada)")
        self.assertEqual(self.server.engine.app.capture.source.source_id, "demo-sintetico")
        self.assertEqual(s["privacy_notice"], "Fonte simulada: nenhuma câmera em uso. Nenhuma imagem é gravada.")
        self.assertEqual((s["tool_name"], s["tool_caption"]),
                         ("Acurix Edge · Visualizador de bancada", "ferramenta de desenvolvimento"))

    def test_no_decision_reason_is_the_real_cause(self):  # owner review 19/11/2026, item 3
        self.start_demo()
        self.assertTrue(wait_until(lambda: self.status()["frames_processed"] >= 2))
        s = self.status()
        self.assertEqual(s["verdict"]["value"], "NO_DECISION")
        reason = s["verdict"]["reason"]
        self.assertNotIn("fora de observação", reason)
        # 0.3.x: no approved model and no approved rule -> DEGRADED (real health entries, listed by name)
        self.assertTrue(reason.startswith(("Sistema em DEGRADED: nenhum modelo aprovado",
                                           "Imagem recusada pela porta de qualidade")), reason)

    def test_page_has_fixed_banner_at_the_top_and_tag_on_the_image(self):
        self.start_demo()
        code, page = self.get("/")
        self.assertEqual(code, 200)
        text = page.decode("utf-8")
        banner = re.search(r'<div class="ax-banner ax-banner--p2 bv-simulated" role="note">([^<]+)</div>', text)
        self.assertIsNotNone(banner)
        self.assertEqual(banner.group(1), NOTICE)
        self.assertLess(banner.start(), text.index('<header class="ax-strip'), "banner is the first thing on top")
        self.assertIn('<div class="bv-sim-tag" aria-hidden="true">FONTE SIMULADA</div>', text)
        self.assertNotIn("<!--SIMULATED-SOURCE", text)
        css = (STATIC_DIR / "viewer.css").read_text(encoding="utf-8")
        self.assertRegex(css, r"\.bv-simulated\s*\{[^}]*position:\s*sticky;[^}]*top:\s*0;")
        js = (STATIC_DIR / "viewer.js").read_text(encoding="utf-8")
        for name in ("bv-simulated", "bv-sim-tag", "role=\"note\"", "querySelector", "document.body", ".remove()"):
            self.assertNotIn(name, js, "the page script never touches (or can remove) the simulated banner")

    def test_recorded_events_mark_the_source_as_simulated(self):
        self.start_demo()
        self.assertTrue(wait_until(lambda: self.status()["frames_processed"] >= 3))
        events = self.server.engine.app.store.list_events(limit=50)
        self.assertGreaterEqual(len(events), 3)
        for ev in events:
            self.assertEqual(ev.extra["input_source"]["simulated"], True)
            self.assertEqual(ev.extra["input_source"]["source_id"], "demo-sintetico")
            self.assertIn("SIMULATED SOURCE", ev.extra["input_source"]["notice"])
            self.assertNotIn(ev.result.value, ("PASS", "DEFECT"))

    def test_real_profile_in_the_config_is_ignored(self):
        p = self.tmp / "real.toml"
        p.write_text('[capture]\ncamera_backend = "DSHOW"\ncamera_default_resolution = "640x480"\n'
                     'camera_resolutions = ["800x600"]\ncamera_device_label = "GENERAL WEBCAM"\n', encoding="utf-8")
        self.start_demo(config_path=p)
        self.assertTrue(wait_until(lambda: self.status()["frames_processed"] >= 1))
        self.assertEqual(self.server.engine.app.config.capture.camera_backend, "SYNTH-DEMO")


class RealModeIsUnchangedTests(TempDirTestCase):
    def test_real_mode_page_has_no_simulated_banner_and_events_have_no_input_source(self):
        cam = SyntheticCamera()
        p = self.tmp / "cam.toml"
        prof = cam.profile("Bench")
        res = ", ".join(f'"{r}"' for r in prof.supported_resolutions)
        p.write_text(f'[capture]\nframe_timeout_s = 0.3\ncamera_backend = "{prof.backend}"\n'
                     f'camera_default_resolution = "{prof.default_resolution}"\ncamera_resolutions = [{res}]\n'
                     f'camera_device_label = "Bench"\n', encoding="utf-8")
        server = build_bench_view(config_path=p, base_dir=self.tmp, port=0, capture_factory=cam.factory,
                                  host_checks=HEALTHY_HOST)
        self.addCleanup(server.server_close)
        self.addCleanup(server.engine.stop)
        page = server.static["/"][0].decode("utf-8")
        self.assertNotIn("FONTE SIMULADA", page)
        self.assertNotIn("<!--SIMULATED-SOURCE", page)
        self.assertTrue(server.engine.start())
        self.assertTrue(wait_until(lambda: server.engine.frames_processed >= 1))
        self.assertEqual(server.engine.snapshot()["source"]["kind"], "CAMERA")
        for ev in server.engine.app.store.list_events(limit=10):
            self.assertNotIn("input_source", ev.extra)


class DemoFactoryAndCliTests(TempDirTestCase):
    def cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        code: int | str | None
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = main(list(args))
            except SystemExit as exc:
                code = exc.code
        return code, out.getvalue(), err.getvalue()

    def test_demo_refuses_another_capture_factory_and_real_mode_requires_config(self):
        with self.assertRaises(BenchViewError):
            build_bench_view(config_path=None, base_dir=self.tmp, port=0, synthetic=True,
                             capture_factory=SyntheticCamera().factory)
        with self.assertRaises(BenchViewError) as ctx:
            build_bench_view(config_path=None, base_dir=self.tmp, port=0)
        self.assertIn("--config", ctx.exception.problems[0])

    def test_simulated_and_real_data_never_share_a_folder(self):  # review 002, MÉDIO
        station = self.tmp / "station"
        (station / ".data").mkdir(parents=True)
        (station / ".data" / "acurix.db").write_bytes(b"real station data")
        with self.assertRaises(BenchViewError) as ctx:
            build_bench_view(config_path=None, base_dir=station, port=0, synthetic=True, host_checks=HEALTHY_HOST)
        self.assertIn("estação real", ctx.exception.problems[0])
        self.assertFalse((station / DEMO_MARKER).exists(), "a refused run marks nothing")
        demo_dir = self.tmp / "demo"
        server = build_bench_view(config_path=None, base_dir=demo_dir, port=0, synthetic=True,
                                  host_checks=HEALTHY_HOST)
        server.engine.stop()
        server.server_close()
        self.assertTrue((demo_dir / DEMO_MARKER).exists())
        again = build_bench_view(config_path=None, base_dir=demo_dir, port=0, synthetic=True,
                                 host_checks=HEALTHY_HOST)  # the demo may reuse ITS OWN folder
        again.engine.stop()
        again.server_close()
        cam = SyntheticCamera()
        cfg = self.tmp / "cam.toml"
        prof = cam.profile("Bench")
        res = ", ".join(f'"{r}"' for r in prof.supported_resolutions)
        cfg.write_text(f'[capture]\ncamera_backend = "{prof.backend}"\ncamera_default_resolution = '
                       f'"{prof.default_resolution}"\ncamera_resolutions = [{res}]\ncamera_device_label = "Bench"\n',
                       encoding="utf-8")
        with self.assertRaises(BenchViewError) as ctx:  # real mode never writes into a demo folder
            build_bench_view(config_path=cfg, base_dir=demo_dir, port=0, capture_factory=cam.factory,
                             host_checks=HEALTHY_HOST)
        self.assertIn("demo sintético", ctx.exception.problems[0])
        self.assertEqual(cam.opens, 0)

    def test_fault_banner_in_demo_has_no_cable_instruction(self):
        from acurix_edge.benchview import view_model as vm

        b = vm.banner_view("FAULT", "FAULT", None, None, simulated=True)
        assert b is not None
        self.assertEqual(b["title"], "Fonte simulada interrompida — inspeção parada")
        self.assertNotIn("cabo USB, encerre", b["text"])
        self.assertEqual(b["text"], vm.RESET_HELP_SIMULATED)

    def test_cli_refuses_device_with_demo_and_missing_config_without_demo(self):
        code, _, err = self.cli("bench-view", "--demo-sintetico", "--device", "0", "--port", "0")
        self.assertEqual(code, 2)
        self.assertIn("--device", err)
        code, _, err = self.cli("bench-view", "--port", "0")
        self.assertEqual(code, 2)
        self.assertIn("--config", err)

    def test_cli_demo_uses_a_disposable_folder_deleted_at_exit(self):
        boom = AssertionError("a real camera was opened")
        with mock.patch("acurix_edge.capture.sources.cv2.VideoCapture", side_effect=boom), \
                mock.patch.object(BenchViewServer, "serve_forever", side_effect=KeyboardInterrupt):
            code, out, err = self.cli("bench-view", "--demo-sintetico", "--port", "0", "--fps", "4")
        self.assertEqual(code, 0, err)
        lines = [json.loads(line) for line in out.splitlines() if line.strip()]
        started, stopped = lines[0], lines[-1]
        self.assertEqual(started["source"], NOTICE)
        self.assertTrue(started["data_dir_disposable"])
        self.assertIn("acurix-demo-sintetico-", started["data_dir"])
        self.assertTrue(started["session_started"])
        self.assertFalse(Path(started["data_dir"]).exists(), "disposable folder deleted at exit")
        self.assertTrue(stopped["disposable_data_removed"])

    def test_cli_demo_keeps_an_explicit_base_dir(self):
        with mock.patch.object(BenchViewServer, "serve_forever", side_effect=KeyboardInterrupt):
            code, out, err = self.cli("bench-view", "--demo-sintetico", "--port", "0",
                                      "--base-dir", str(self.tmp / "keep"))
        self.assertEqual(code, 0, err)
        started = json.loads(out.splitlines()[0])
        self.assertFalse(started["data_dir_disposable"])
        self.assertTrue((self.tmp / "keep").is_dir(), "an explicit --base-dir is never deleted")


class SceneTests(unittest.TestCase):
    def test_scene_cycle_and_quality_gate(self):
        self.assertAlmostEqual(syn.scene_angle_deg(12.0)[0], 5.0)
        self.assertEqual(syn.scene_angle_deg(17.0)[1], 200)
        self.assertEqual(syn.scene_angle_deg(21.0), syn.scene_angle_deg(1.0))
        cap = syn.SceneCapture(clock=iter([0.0, 1.0, 17.0]).__next__, pace=False)
        gate = QualityGate(QualityConfig())
        ok, aligned = cap.read()
        self.assertTrue(ok)
        self.assertEqual(aligned.shape, (480, 640, 3))
        self.assertEqual(gate.evaluate(aligned).status.value, "VALID")
        _, outside = cap.read()
        self.assertEqual(gate.evaluate(outside).status.value, "INPUT_INVALID")
        self.assertEqual(cap.getBackendName(), "SYNTH-DEMO")
        cap.release()
        self.assertEqual(cap.read(), (False, None))

    def test_synthetic_source_cannot_use_a_real_capture_factory(self):
        src = syn.SyntheticSceneSource()
        self.assertTrue(src.simulated)
        self.assertIs(src._factory, syn._scene_factory)
        self.assertEqual(SIMULATED_NOTICE, NOTICE)


if __name__ == "__main__":
    unittest.main()
