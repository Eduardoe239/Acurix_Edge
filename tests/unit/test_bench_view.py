"""AEX-BENCH-VIEW-001 (Ordem 5) — bench viewer: local server, MJPEG, status, safety properties.

The real camera is never opened: the webcam source uses the synthetic camera factory."""

import http.client
import json
import os
import re
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from acurix_edge.benchview import view_model as vm
from acurix_edge.benchview.app_factory import BenchViewError, build_bench_view
from acurix_edge.benchview.engine import p95
from acurix_edge.benchview.server import HOST, MAX_STREAMS, STATIC, BenchViewServer
from acurix_edge.config.settings import QualityConfig
from acurix_edge.contracts.decision import InspectionResult
from tests.capture_support import SyntheticCamera, profile_config
from tests.support import HEALTHY_HOST, TempDirTestCase

VERDICT_WORDS = re.compile(rb"\b(PASS|DEFECT)\b")
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".mjpg"}
STATIC_DIR = Path(__file__).resolve().parents[2] / "src" / "acurix_edge" / "benchview" / "static"
DS_DIR = Path(__file__).resolve().parents[2] / "resources" / "design-system"


def wait_until(pred, timeout=3.0, step=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(step)
    return bool(pred())


def toml_value(v):
    if isinstance(v, tuple):
        return "[" + ", ".join(f'"{x}"' for x in v) + "]"
    return f'"{v}"'


class BenchViewCase(TempDirTestCase):
    LABEL = "Bench Cam (USB, sintética)"

    def write_config(self, with_profile=True, extra=""):
        lines = ["[capture]", "frame_timeout_s = 0.3", "verify_timeout_s = 1.0", "open_timeout_s = 1.0"]
        if with_profile:
            profile = profile_config(SyntheticCamera().profile(self.LABEL))
            lines += [f"{k} = {toml_value(v)}" for k, v in profile.items()]
        p = self.tmp / "bench.toml"
        p.write_text("\n".join(lines) + "\n" + extra, encoding="utf-8")
        return p

    def start(self, cam=None, fps=15.0):
        self.cam = cam or SyntheticCamera()
        self.server: BenchViewServer = build_bench_view(
            config_path=self.write_config(), base_dir=self.tmp, port=0, fps=fps, stream_width=160,
            capture_factory=self.cam.factory, host_checks=HEALTHY_HOST)
        self.started = self.server.engine.start()
        t = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        t.start()
        self.serve_thread = t

        def cleanup():
            self.server.engine.stop()
            self.server.shutdown()
            self.server.server_close()
            self.cam.unstall()

        self.addCleanup(cleanup)
        return self.server

    def request(self, method, path, headers=None, host=None):
        conn = http.client.HTTPConnection(HOST, self.server.port, timeout=5)
        try:
            if host is not None:
                conn.putrequest(method, path, skip_host=True)
                conn.putheader("Host", host)
                for k, v in (headers or {}).items():
                    conn.putheader(k, v)
                conn.endheaders()
            else:
                conn.request(method, path, headers=headers or {})
            resp = conn.getresponse()
            return resp.status, dict(resp.getheaders()), resp.read()
        finally:
            conn.close()

    def status(self):
        code, _, body = self.request("GET", "/api/status")
        self.assertEqual(code, 200)
        return json.loads(body)


class ServerTests(BenchViewCase):
    def test_listens_only_on_loopback_and_status_reports_processing(self):
        server = self.start()
        self.assertTrue(self.started)
        self.assertEqual(server.server_address[0], "127.0.0.1")
        self.assertEqual(server.url, f"http://127.0.0.1:{server.port}/")
        self.assertTrue(wait_until(lambda: self.status()["frames_processed"] >= 3))
        s = self.status()
        self.assertEqual(s["strip"]["camera"], "OK")
        self.assertIn(s["strip"]["system"], ("OBSERVING", "DEGRADED"))
        self.assertIn("provisório", s["strip"]["envelope_text"])
        self.assertTrue(s["strip"]["latency_text"].endswith(" ms"))
        self.assertEqual(s["shadow"]["title"], "Prévia (modo sombra) — não decide")
        self.assertEqual(s["privacy_notice"], "Câmera apontada só para a bancada. Nenhuma imagem é gravada.")
        self.assertEqual(s["verdict"]["value"], "NO_DECISION")
        self.assertEqual(s["verdict"]["phrase"], "O sistema ainda não decide: especialista em validação")

    def test_quality_refusal_shows_the_reason(self):
        self.start()  # tiny synthetic frames do not match the template: the gate refuses them
        self.assertTrue(wait_until(lambda: self.status()["quality"]["valid"] is False))
        s = self.status()
        self.assertEqual(s["quality"]["status"], "INPUT_INVALID")
        self.assertTrue(s["quality"]["reasons"])
        self.assertTrue(all(r in vm.QUALITY_REASON_PT.values() for r in s["quality"]["reasons"]))
        self.assertEqual([b["label"] for b in s["quality"]["bars"]],
                         ["Nitidez", "Brilho", "Saturação", "Posição", "Rotação do produto"])
        self.assertIn("recusada", s["shadow"]["text"])  # the specialist is not called on a refused image
        self.assertEqual(s["trail"][0]["kind"], "warn")
        self.assertEqual([t["detail"] for t in s["trail"][1:3]], ["não implementada", "não implementada"])

    def test_mjpeg_stream_serves_jpeg_frames(self):
        self.start()
        conn = http.client.HTTPConnection(HOST, self.server.port, timeout=5)
        try:
            conn.request("GET", "/stream.mjpg")
            resp = conn.getresponse()
            self.assertEqual(resp.status, 200)
            self.assertTrue(resp.getheader("Content-Type").startswith("multipart/x-mixed-replace"))
            chunk = b""
            deadline = time.monotonic() + 5
            while b"\xff\xd9" not in chunk and time.monotonic() < deadline:
                chunk += resp.read1(4096)
            self.assertIn(b"Content-Type: image/jpeg", chunk)
            self.assertIn(b"\xff\xd8", chunk)  # JPEG start of image
        finally:
            conn.close()

    def test_streams_are_bounded(self):
        self.start()
        conns = []
        try:
            for _ in range(MAX_STREAMS):
                c = http.client.HTTPConnection(HOST, self.server.port, timeout=5)
                c.request("GET", "/stream.mjpg")
                self.assertEqual(c.getresponse().status, 200)
                conns.append(c)
            code, _, _ = self.request("GET", "/stream.mjpg")
            self.assertEqual(code, 503)
        finally:
            for c in conns:
                c.close()

    def test_host_header_routes_and_methods_are_closed(self):
        self.start()
        self.assertEqual(self.request("GET", "/api/status", host="evil.example:80")[0], 403)  # DNS rebinding
        self.assertEqual(self.request("GET", "/api/status", host=f"localhost:{self.server.port}")[0], 200)
        for path in ("/static/../app.py", "/static/x.css", "/etc/passwd", "/api/statusx", "/static/%2e%2e/app.py"):
            with self.subTest(path=path):
                self.assertEqual(self.request("GET", path)[0], 404)
        self.assertEqual(self.request("POST", "/api/shutdown")[0], 403)  # custom header required (CSRF)
        self.assertEqual(self.request("POST", "/api/status", headers={"X-Acurix-Action": "shutdown"})[0], 404)
        for method in ("PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"):
            with self.subTest(method=method):
                code, headers, _ = self.request(method, "/api/status")
                self.assertEqual(code, 405)
                self.assertIn("default-src 'none'", headers["Content-Security-Policy"])  # review B8
        self.assertFalse(self.server.engine.stopped)
        code, headers, _ = self.request("GET", "/")
        self.assertEqual(code, 200)
        self.assertIn("default-src 'none'", headers["Content-Security-Policy"])
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(headers["Cache-Control"], "no-store")

    def test_other_sites_cannot_embed_the_stream_or_read_status(self):  # review M2
        self.start()
        for site in ("cross-site", "same-site"):
            for path in ("/stream.mjpg", "/api/status", "/"):
                with self.subTest(site=site, path=path):
                    code, headers, _ = self.request("GET", path, headers={"Sec-Fetch-Site": site})
                    self.assertEqual(code, 403)
                    self.assertEqual(headers["Cross-Origin-Resource-Policy"], "same-origin")
        for site in ("same-origin", "none"):
            with self.subTest(site=site):
                self.assertEqual(self.request("GET", "/api/status", headers={"Sec-Fetch-Site": site})[0], 200)
        self.assertEqual(self.request("POST", "/api/shutdown", headers={
            "X-Acurix-Action": "shutdown", "Sec-Fetch-Site": "cross-site"})[0], 403)
        self.assertFalse(self.server.engine.stopped)

    def test_stalled_camera_shows_fault_within_the_page_budget(self):  # review M4, default 1.0 s
        cam = SyntheticCamera()
        self.cam = cam
        cfg_path = self.tmp / "stall.toml"
        cfg_path.write_text(self.write_config().read_text(encoding="utf-8").replace(
            "frame_timeout_s = 0.3", "frame_timeout_s = 1.0"), encoding="utf-8")
        self.server = build_bench_view(config_path=cfg_path, base_dir=self.tmp, port=0,
                                       capture_factory=cam.factory, host_checks=HEALTHY_HOST)
        self.assertTrue(self.server.engine.start())
        t = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        t.start()

        def cleanup():
            cam.unstall()
            self.server.engine.stop()
            self.server.shutdown()
            self.server.server_close()

        self.addCleanup(cleanup)
        self.assertTrue(wait_until(lambda: self.status()["frames_processed"] >= 1))
        stalled_at = time.monotonic()
        cam.stall()
        self.assertTrue(wait_until(lambda: self.status()["strip"]["system"] == "FAULT", 2.0))
        elapsed = time.monotonic() - stalled_at
        self.assertLess(elapsed, 1.5, f"{elapsed:.2f} s + page poll 0.5 s must stay < 2 s")
        print(f"\n[BENCH-VIEW] stall -> FAULT in status: {elapsed * 1000:.0f} ms (frame_timeout_s=1.0)")

    def test_concurrent_stops_wait_for_the_release(self):  # review B6
        self.start()
        self.assertTrue(wait_until(lambda: self.status()["frames_processed"] >= 1))
        threads = [threading.Thread(target=self.server.engine.stop) for _ in range(3)]
        for th in threads:
            th.start()
        for th in threads:
            th.join(10)
        self.assertTrue(all(not c.isOpened() for c in self.cam.caps), "every stop() returned after the release")

    def test_static_assets_served(self):
        self.start()
        for path, (_, ctype) in STATIC.items():
            with self.subTest(path=path):
                code, headers, body = self.request("GET", path)
                self.assertEqual(code, 200)
                self.assertEqual(headers["Content-Type"], ctype)
                self.assertTrue(body)

    def test_no_response_ever_contains_pass_or_defect(self):
        self.start()
        self.assertTrue(wait_until(lambda: self.status()["frames_processed"] >= 2))
        for path in [*STATIC, "/api/status", "/api/unknown"]:
            with self.subTest(path=path):
                _, _, body = self.request("GET", path)
                self.assertIsNone(VERDICT_WORDS.search(body), f"{path} contains a conclusive verdict word")

    def test_camera_loss_shows_fault_within_2s_and_nothing_reopens(self):
        self.start()
        self.assertTrue(wait_until(lambda: self.status()["frames_processed"] >= 2))
        opens = self.cam.opens
        lost_at = time.monotonic()
        self.cam.disconnect()
        self.assertTrue(wait_until(lambda: self.status()["strip"]["system"] == "FAULT", 2.0), "FAULT within 2 s")
        self.assertLess(time.monotonic() - lost_at, 2.0)
        s = self.status()
        self.assertEqual(s["strip"]["camera"], "FAULT")
        self.assertEqual(s["banner"]["title"], "Câmera desconectada — inspeção parada")
        self.assertEqual(s["banner"]["code"], "AEX-CAP-001")
        self.assertIn("reset-fault", s["banner"]["text"])
        self.cam.reconnect()
        time.sleep(0.5)
        self.assertEqual(self.cam.opens, opens, "the viewer never reopens a lost camera (AEX-CAPTURE-007)")
        self.assertEqual(self.status()["strip"]["system"], "FAULT")

    def test_shutdown_button_stops_cleanly_and_releases_the_camera(self):
        self.start()
        self.assertTrue(wait_until(lambda: self.status()["frames_processed"] >= 1))
        code, _, _ = self.request("POST", "/api/shutdown", headers={"X-Acurix-Action": "shutdown"})
        self.assertEqual(code, 202)
        self.serve_thread.join(5)
        self.assertFalse(self.serve_thread.is_alive(), "server stopped")
        self.assertTrue(self.server.engine.stopped)
        self.assertTrue(all(not c.isOpened() for c in self.cam.caps), "camera released")

    def test_no_image_file_is_written(self):
        before = time.time()
        with mock.patch("cv2.imwrite", side_effect=AssertionError("image written to disk")) as imwrite:
            self.start()
            self.assertTrue(wait_until(lambda: self.status()["frames_processed"] >= 5))
            self.server.engine.stop()
        imwrite.assert_not_called()
        created = []
        for root in (self.tmp, Path.cwd()):
            for dirpath, dirnames, files in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in {".venv", ".git", "archive", "pdf", "docs"}]
                for f in files:
                    p = Path(dirpath) / f
                    if p.suffix.lower() in IMAGE_EXT and p.stat().st_mtime >= before:
                        created.append(p)
        self.assertEqual(created, [])

    def test_start_refused_when_camera_profile_differs(self):
        server = self.start(cam=SyntheticCamera(backend="OTHER"))
        self.assertFalse(self.started)
        s = self.status()
        self.assertEqual(s["banner"]["title"], "Inspeção não iniciada")
        self.assertIn("não corresponde à câmera configurada", s["banner"]["text"])
        self.assertEqual(s["banner"]["code"], "AEX-CAP-001")
        self.assertEqual(s["verdict"]["value"], "NO_DECISION")
        self.assertEqual(server.engine.frames_processed, 0)


class FactoryTests(BenchViewCase):
    def test_refused_without_camera_profile_or_with_bad_parameters(self):
        with self.assertRaises(BenchViewError) as ctx:
            build_bench_view(config_path=self.write_config(with_profile=False), base_dir=self.tmp, port=0)
        self.assertIn("perfil da câmera", ctx.exception.problems[0])
        for kw in ({"fps": 0.5}, {"fps": 30.0}, {"stream_width": 100}, {"stream_width": 4000}, {"port": 80},
                   {"device": -1}):
            with self.subTest(**kw), self.assertRaises(BenchViewError):
                build_bench_view(config_path=self.write_config(), base_dir=self.tmp,
                                 **{"port": 0, **kw})
        slow = self.write_config().read_text(encoding="utf-8").replace("frame_timeout_s = 0.3", "frame_timeout_s = 1.5")
        p = self.tmp / "slow.toml"
        p.write_text(slow, encoding="utf-8")
        with self.assertRaises(BenchViewError) as ctx:  # review M4: FAULT on the page within 2 s
            build_bench_view(config_path=p, base_dir=self.tmp, port=0)
        self.assertIn("frame_timeout_s", ctx.exception.problems[0])

    def test_busy_port_is_refused(self):
        server = self.start()
        with self.assertRaises(BenchViewError) as ctx:
            build_bench_view(config_path=self.write_config(), base_dir=self.tmp / "b", port=server.port,
                             capture_factory=SyntheticCamera().factory, host_checks=HEALTHY_HOST)
        self.assertIn("indisponível", ctx.exception.problems[0])


class ViewModelTests(unittest.TestCase):
    Q = QualityConfig()

    def test_verdict_never_pass_or_defect(self):
        for result in [*(r.value for r in InspectionResult), None, "ANYTHING"]:
            with self.subTest(result=result):
                v = vm.verdict_view(result, "RULE_APPLIED")
                self.assertNotIn(v["value"], ("PASS", "DEFECT"))
                self.assertIsNone(VERDICT_WORDS.search(json.dumps(v).encode()))
        self.assertTrue(vm.verdict_view("PASS", None)["hidden"])
        self.assertEqual(vm.verdict_view("NO_DECISION", "INPUT_QUALITY")["value"], "NO_DECISION")

    def full(self, result, stages, quality=None):
        quality = quality or {"status": "VALID", "reasons": [], "metrics": {"object_area_fraction_roi": 0.5,
                                                                         "object_rotation_measured": 1.0},
                              "roi_px": [10, 5, 70, 40]}
        return vm.build_status(system_state="OBSERVING", camera_status="OK", latency_p95_ms=12.0,
                               fps=8.0, ram_free_gb=1.2, quality_config=self.Q, result=result, reason="RULE_APPLIED",
                               detail={"quality": quality, "stages": stages}, frame_size=(100, 50), problem=None,
                               problem_code=None, frames_processed=1, running=True)

    def shadow_stage(self, cls, angle):
        return [{"shadow": True, "result": {"status": "OK", "error_code": None, "predictions": [
            {"class_id": cls, "confidence": 0.9, "attributes": {"angle_deg": angle, "tolerance_deg": 5.0,
                                                                "borderline_low_deg": 3.0}}]}}]

    def test_full_status_never_contains_conclusive_words(self):
        for result in [r.value for r in InspectionResult]:
            for cls, angle in (("OK", 1.2), ("LABEL_SKEWED", -7.5)):
                with self.subTest(result=result, cls=cls):
                    s = self.full(result, self.shadow_stage(cls, angle))
                    self.assertIsNone(VERDICT_WORDS.search(json.dumps(s, ensure_ascii=False).encode()))

    def test_shadow_preview_is_neutral_and_labelled(self):
        s = vm.shadow_view(self.shadow_stage("LABEL_SKEWED", -7.5), True)
        self.assertEqual(s["title"], "Prévia (modo sombra) — não decide")
        self.assertEqual(s["text"], "Etiqueta inclinada")
        self.assertEqual(s["angle_text"], "-7,5°")
        self.assertEqual(s["bar"]["band_pct"], [30.0, 50.0])  # 3–5° on a 0–10° scale
        self.assertEqual(s["bar"]["thresholds_pct"], [50.0])
        self.assertTrue(s["bar"]["attention"])
        ok = vm.shadow_view(self.shadow_stage("OK", 1.2), True)
        self.assertEqual(ok["text"], "Etiqueta alinhada")
        self.assertFalse(ok["bar"]["attention"])

    def test_shadow_text_always_matches_the_marker_colour(self):  # owner review 19/11/2026, item 1
        expected = {2.9: ("Etiqueta alinhada", False), 3.0: ("Caso limite (3–5°)", True),
                    4.2: ("Caso limite (3–5°)", True), 5.0: ("Caso limite (3–5°)", True),
                    5.1: ("Etiqueta inclinada", True), 0.0: ("Etiqueta alinhada", False)}
        for angle, (text, attention) in expected.items():
            for sign in (1, -1):
                with self.subTest(angle=sign * angle):
                    cls = "LABEL_SKEWED" if angle > 5 else "OK"
                    s = vm.shadow_view(self.shadow_stage(cls, sign * angle), True)
                    self.assertEqual(s["text"], text)
                    self.assertEqual(s["bar"]["attention"], attention)
                    self.assertEqual(s["tone"] == "review", attention, "text tone == marker colour")
        for tenth in range(0, 101):  # property: every angle 0..10° in 0,1° steps
            s = vm.shadow_view(self.shadow_stage("OK", tenth / 10), True)
            self.assertEqual(s["bar"]["attention"], s["text"] != "Etiqueta alinhada")
            self.assertNotEqual(s["text"] == "Etiqueta alinhada", tenth >= 30)
        inconclusive = vm.shadow_view([{"shadow": True, "result": {"status": "INCONCLUSIVE",
                                                                    "error_code": "LABEL_NOT_FOUND"}}], True)
        self.assertEqual(inconclusive["text"], "Inconclusiva: etiqueta não encontrada.")
        self.assertIsNone(inconclusive["bar"])
        self.assertIn("recusada", vm.shadow_view(None, False)["text"])

    def quality(self, status, metrics, reasons=()):
        return vm.quality_view({"status": status, "metrics": metrics,
                                "reasons": [{"code": c, "metric": m} for c, m in reasons]}, self.Q)

    def bar(self, view, label):
        return next(b for b in view["bars"] if b["label"] == label)

    def test_position_is_displacement_in_px_and_rotation_is_separate(self):  # item 2
        shifted = self.quality("INPUT_INVALID", {
            "object_area_fraction_roi": 0.4, "object_outside_roi_px": 166.0, "object_rotation_measured": 1.0,
            "object_rotation_deg": 0.0, "object_rotation_abs_deg": 0.0},
            reasons=[("OBJECT_OUTSIDE_ROI", "object_outside_roi_px")])
        pos, rot = self.bar(shifted, "Posição"), self.bar(shifted, "Rotação do produto")
        self.assertEqual(pos["value_text"], "166 px")
        self.assertTrue(pos["attention"])
        self.assertEqual(pos["detail"], "limite 0 px · peça deslocada para fora do gabarito")
        self.assertEqual(pos["scale_text"], ["0 px", "100+ px"])
        self.assertEqual(rot["value_text"], "—", "never 0,0° when the refusal is a displacement")
        self.assertIsNone(rot["value_pct"], "no marker for a value that was not evaluated")
        self.assertIn("peça fora do gabarito", rot["detail"])
        self.assertNotIn("0,0°", json.dumps(shifted, ensure_ascii=False))
        inside = self.quality("VALID", {
            "object_area_fraction_roi": 0.4, "object_outside_roi_px": 0.0, "object_rotation_measured": 1.0,
            "object_rotation_deg": -2.4, "object_rotation_abs_deg": 2.4})
        self.assertEqual(self.bar(inside, "Posição")["value_text"], "0 px")
        self.assertEqual(self.bar(inside, "Posição")["detail"], "limite 0 px · dentro do gabarito")
        self.assertEqual(self.bar(inside, "Rotação do produto")["value_text"], "-2,4°")
        self.assertEqual(self.bar(inside, "Rotação do produto")["detail"], "limite 10°")
        missing = self.quality("INPUT_INVALID", {"object_area_fraction_roi": 0.0},
                               reasons=[("OBJECT_NOT_FOUND", "object_area_fraction_roi")])
        for label in ("Posição", "Rotação do produto"):
            with self.subTest(label=label):
                self.assertEqual(self.bar(missing, label)["value_text"], "—")
                self.assertIsNone(self.bar(missing, label)["value_pct"])

    def test_saturation_caption_uses_real_values(self):  # item 6
        v = self.quality("VALID", {"bright_fraction": 0.032, "dark_fraction": 0.0})
        sat = self.bar(v, "Saturação")
        self.assertEqual(sat["detail"], f"limite {vm.fmt(self.Q.max_bright_fraction * 100, 0)} % · sombras 0,0 %")
        self.assertEqual(sat["value_text"], "3,2 %")

    def test_no_decision_reason_tells_the_real_cause(self):  # item 3
        degraded = [("models", "DEGRADED", ""), ("memory", "DEGRADED", "AEX-CAS-004"),
                    ("decision_rule", "DEGRADED", "")]
        cases = [
            (("NO_DECISION", "SYSTEM_NOT_OBSERVING", "DEGRADED", degraded, ()),
             "Sistema em DEGRADED: nenhum modelo aprovado; RAM livre abaixo de 1,5 GB; "
             "nenhuma regra de decisão aprovada"),
            (("NO_DECISION", "HEALTH_NOT_OK", "DEGRADED", [("gpu", "DEGRADED", "")], ()),
             "Sistema em DEGRADED: placa de vídeo: DEGRADED"),
            (("NO_DECISION", "SYSTEM_NOT_OBSERVING", "DEGRADED", [("weird_entry", "UNKNOWN", "")], ()),
             "Sistema em DEGRADED: weird_entry: não medido"),
            (("NO_DECISION", "SYSTEM_NOT_OBSERVING", "OBSERVING", (), ()), "Aguardando o próximo quadro"),
            (("NO_DECISION", "INPUT_QUALITY", "DEGRADED", degraded, ("Peça fora da área do gabarito",)),
             "Imagem recusada pela porta de qualidade: Peça fora da área do gabarito"),
            (("NO_DECISION", "INFERENCE_NOT_OK", "OBSERVING", (), ()),
             "Especialista em validação: nenhum modelo aprovado para decidir"),
            (("NO_DECISION", "SYSTEM_NOT_OBSERVING", "IDLE", degraded, ()), "Sessão de inspeção não iniciada"),
            (("NO_DECISION", "INPUT_QUALITY", "FAULT", degraded, ("x",)), "Sistema em FAULT: inspeção parada"),
            ((None, None, "IDLE", (), ()), "Sessão de inspeção não iniciada"),
            ((None, None, "OBSERVING", (), ()), "Aguardando o primeiro quadro"),
        ]
        for (result, reason, state, issues, q_reasons), text in cases:
            with self.subTest(reason=reason, state=state):
                v = vm.verdict_view(result, reason, system_state=state, health_issues=issues,
                                    quality_reasons=q_reasons)
                self.assertEqual(v["reason"], text)
                self.assertNotIn("fora de observação", v["reason"])

    def test_health_phrase_never_states_a_cause_the_status_does_not_support(self):  # review 003, M1
        cases = {
            ("memory", "UNKNOWN", ""): "RAM livre: não medido",           # not "abaixo de 1,5 GB"
            ("memory", "DEGRADED", "AEX-CAS-004"): "RAM livre abaixo de 1,5 GB",
            ("cpu", "UNKNOWN", ""): "processador: não medido",            # never "sobrecarregado"
            ("quality_envelope", "DEGRADED", "AEX-SYS-001"): "porta de qualidade com erros seguidos",
            ("quality_envelope", "DEGRADED", "AEX-CFG-001"): "envelope de qualidade recusado",
            ("camera", "FAULT", "AEX-CAP-001"): "câmera: FAULT",
            ("models", "DEGRADED", "AEX-CFG-002"): "modelo carregado fora da lista aprovada",
        }
        for issue, text in cases.items():
            with self.subTest(issue=issue):
                self.assertEqual(vm.health_phrase(issue), text)

    def test_preview_classifies_the_value_shown(self):  # review 003, M2
        for angle, shown, text in ((2.96, "3,0°", "Caso limite (3–5°)"), (2.94, "2,9°", "Etiqueta alinhada"),
                                   (4.96, "5,0°", "Caso limite (3–5°)"), (5.04, "5,0°", "Caso limite (3–5°)"),
                                   (5.06, "5,1°", "Etiqueta inclinada"), (-2.96, "-3,0°", "Caso limite (3–5°)")):
            with self.subTest(angle=angle):
                s = vm.shadow_view(self.shadow_stage("OK", angle), True)
                self.assertEqual((s["angle_text"], s["text"]), (shown, text))
                self.assertEqual(s["bar"]["value_text"], shown)

    def test_limits_keep_their_decimals_and_bad_attributes_do_not_break(self):  # review 003, B5 + hypothesis
        q = QualityConfig(max_object_rotation_deg=0.5, max_bright_fraction=0.155)
        v = vm.quality_view({"status": "VALID", "reasons": [], "metrics": {
            "object_area_fraction_roi": 0.5, "object_outside_roi_px": 0.0, "object_rotation_measured": 1.0,
            "object_rotation_deg": 0.2, "object_rotation_abs_deg": 0.2, "bright_fraction": 0.0,
            "dark_fraction": 0.0}}, q)
        self.assertEqual(self.bar(v, "Rotação do produto")["detail"], "limite 0,5°")
        self.assertTrue(self.bar(v, "Saturação")["detail"].startswith("limite 15,5 %"))
        s = vm.shadow_view([{"shadow": True, "result": {"status": "OK", "predictions": [
            {"class_id": "OK", "attributes": {"angle_deg": 1.0, "tolerance_deg": None}}]}}], True)
        self.assertEqual(s["text"], "Etiqueta alinhada")  # missing tolerance -> default, no exception

    def test_roi_and_trail(self):
        s = self.full("NO_DECISION", self.shadow_stage("OK", 0.5))
        self.assertEqual(s["roi_pct"], {"x": 10.0, "y": 10.0, "w": 70.0, "h": 80.0})
        self.assertEqual([t["name"] for t in s["trail"]],
                         ["Qualidade da imagem", "Anomalia", "Roteador", "Especialista — etiqueta (sombra)",
                          "Regra final"])
        self.assertEqual(s["trail"][-1]["value"], "NO_DECISION")
        self.assertEqual(s["trail"][3]["value"], "0,5°")

    def test_last_barrier_scrubs_any_conclusive_word_from_any_string(self):  # review B3
        s = vm.build_status(system_state="DEGRADED", camera_status="OK", latency_p95_ms=None, fps=None,
                            ram_free_gb=None, quality_config=self.Q, result="NO_DECISION", reason="SOMETHING PASS",
                            detail={"quality": {"status": "INPUT_INVALID", "metrics": {},
                                                "reasons": [{"code": "DEFECT_LIKE", "metric": "x"}]},
                                    "stages": [{"shadow": False, "result": {"status": "PASS"}}]},
                            frame_size=None, problem="exception text with DEFECT inside", problem_code="AEX-X",
                            frames_processed=3, running=False, image_stale=True)
        self.assertIsNone(VERDICT_WORDS.search(json.dumps(s, ensure_ascii=False).encode()))
        self.assertEqual(s["banner"]["text"], vm.SCRUBBED)
        self.assertEqual(s["banner"]["title"], "Inspeção interrompida")  # review B4
        self.assertIn("congelada", s["image_note"])
        self.assertEqual(vm.scrub({"a": ["PASS", "ok"], "PASSAGE": 1}), {"a": [vm.SCRUBBED, "ok"], "PASSAGE": 1})

    def test_fault_banner(self):
        b = vm.banner_view("FAULT", "FAULT", "Captura interrompida", "AEX-STATE-002")
        self.assertEqual(b, {"title": "Câmera desconectada — inspeção parada", "text": vm.RESET_HELP,
                             "code": "AEX-CAP-001"})
        self.assertIsNone(vm.banner_view("OBSERVING", "OK", None, None))

    def test_numbers_are_pt_br(self):
        self.assertEqual(vm.fmt(8.25, 1), "8,2")
        self.assertEqual(vm.fmt(float("nan")), "—")
        self.assertEqual(p95([float(i) for i in range(1, 101)]), 95.0)
        self.assertIsNone(p95([]))


class StaticAssetTests(unittest.TestCase):
    def test_own_css_uses_only_tokens(self):
        css = (STATIC_DIR / "viewer.css").read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"#[0-9a-fA-F]{3,8}\b", css), "no loose hex color")
        self.assertIsNone(re.search(r"\b\d+(\.\d+)?px\b", css), "no literal px size")
        self.assertIsNone(re.search(r"rgba?\(|hsla?\(", css), "no literal color function")

    def test_page_has_no_external_resource_or_inline_code(self):
        for name in ("index.html", "viewer.js", "viewer.css"):
            text = (STATIC_DIR / name).read_text(encoding="utf-8")
            with self.subTest(file=name):
                self.assertIsNone(re.search(r"https?://", text), "offline: no CDN / external URL")
                self.assertNotIn("innerHTML", text)
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("style=", html)
        self.assertIsNone(re.search(r"<script>(?!</script>)", html), "no inline script (CSP)")
        self.assertIn("<!--PRIVACY-NOTICE-->", html)  # filled by the server per mode (see ServedPageTests)
        self.assertIn("Luís Eduardo de Sousa Barbosa", html)

    def test_identification_at_the_start_of_the_status_strip(self):  # item 4
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        strip = html[html.index('<header class="ax-strip"'):html.index("</header>")]
        first = strip[strip.index("<span"):]
        self.assertTrue(first.startswith('<span class="ax-strip__item bv-ident"><b>Acurix Edge · Visualizador '
                                         'de bancada</b><span class="ax-limit__scale bv-caption">ferramenta de '
                                         'desenvolvimento</span></span>'))
        bundle = (STATIC_DIR / "bundle.css").read_text(encoding="utf-8")  # caption = 12/16, text-muted
        self.assertRegex(bundle, r"\.ax-limit__scale\s*\{[^}]*color:\s*var\(--text-muted\)")


class ServedPageTests(unittest.TestCase):
    """The privacy notice is written by the server, right below the image, per mode (item 5)."""

    def privacy(self, simulated):
        from acurix_edge.benchview.server import load_static

        page = load_static(simulated=simulated)["/"][0].decode("utf-8")
        self.assertNotIn("<!--PRIVACY-NOTICE-->", page)
        found = re.findall(r'<p class="ax-card bv-privacy" role="note">([^<]+)</p>', page)
        self.assertEqual(len(found), 1, "exactly one privacy notice")
        left = page[page.index('<section class="bv-left"'):page.index("</section>")]
        self.assertLess(left.index('id="media"'), left.index("bv-privacy"), "directly below the image")
        footer = page[page.index("<footer"):]
        self.assertNotIn("Nenhuma imagem é gravada", footer, "not hidden at the bottom anymore")
        return found[0]

    def test_real_mode_notice(self):
        self.assertEqual(self.privacy(False), "Câmera apontada só para a bancada. Nenhuma imagem é gravada.")

    def test_demo_mode_notice(self):
        self.assertEqual(self.privacy(True), "Fonte simulada: nenhuma câmera em uso. Nenhuma imagem é gravada.")

    @unittest.skipUnless(DS_DIR.is_dir(), "design-system source folder not present (not versioned)")
    def test_design_system_copies_are_identical(self):
        for name in ("tokens.css", "bundle.css", "Inter-Variable-latin.woff2"):
            with self.subTest(file=name):
                self.assertEqual((STATIC_DIR / name).read_bytes(), (DS_DIR / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
