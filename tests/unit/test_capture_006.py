"""AEX-CAPTURE-006 — CAP-003 (bounded buffer, counted drops), HLT-002 (camera verified before
OBSERVING), HLT-004 (loss -> FAULT within 2 s), CAP-004 (reconnection never resumes the session).

All sources are synthetic (generated images, fake camera). The real camera is never opened."""

import dataclasses
import threading
import time
import unittest
from typing import Any
from unittest import mock

from acurix_edge.app import Application, SessionError
from acurix_edge.capture.buffer import DROP_QUEUE_FULL, FrameBuffer
from acurix_edge.capture.errors import CaptureLost, CaptureUnavailable, EndOfStream, StaleSession
from acurix_edge.capture.frames import FrameStamper
from acurix_edge.capture.service import CaptureService, CaptureState
from acurix_edge.capture.sources import ImageFolderSource, SourceOpenError, WebcamSource
from acurix_edge.config.settings import AppConfig, CaptureConfig, ConfigError, load_config
from acurix_edge.contracts.errors import ErrorCode
from acurix_edge.domain.state_machine import SystemState
from tests.capture_support import SyntheticCamera, synth_image, write_image_folder
from tests.support import FIXTURE_POLICY, HEALTHY_HOST, FixtureApprovedRule, FixtureOkAdapter, TempDirTestCase

HLT004_TARGET_S = 2.0
FAST = CaptureConfig(queue_max_frames=4, drop_alarm_threshold=0, frame_timeout_s=0.3, verify_timeout_s=1.0,
                     reconnect_interval_s=0.05, reconnect_max_attempts=40)


def wait_until(predicate, timeout=3.0, step=0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return bool(predicate())


def invariant_holds(c) -> bool:
    return bool(c.captured == c.delivered + c.dropped + c.depth)


class FrameBufferTests(unittest.TestCase):
    def frames(self, n):
        stamper = FrameStamper("synthetic")
        return [stamper.stamp(synth_image(i), 0.0) for i in range(n)]

    def test_live_drop_oldest_is_counted_and_alarm_fires_once(self):
        alarms: list[Any] = []
        buf = FrameBuffer(3, 2, drop_oldest=True, on_alarm=alarms.append)
        fs = self.frames(10)
        for f in fs:
            buf.put(f)
        c = buf.counters()
        self.assertEqual((c.captured, c.depth, c.dropped_by_cause), (10, 3, {DROP_QUEUE_FULL: 7}))
        self.assertTrue(c.over_limit)
        self.assertEqual(len(alarms), 1, "AEX-CAP-002 alarm once when crossing the limit")
        self.assertEqual(alarms[0].dropped, 3)  # crossed at 3 > threshold 2
        self.assertEqual([buf.get(0).sequence for _ in range(3)], [8, 9, 10])  # newest kept, oldest dropped
        self.assertTrue(invariant_holds(buf.counters()))

    def test_non_live_back_pressure_never_drops(self):
        buf = FrameBuffer(2, 0, drop_oldest=False)
        fs = self.frames(6)
        got = []

        def consume():
            for _ in range(6):
                got.append(buf.get(2.0))

        t = threading.Thread(target=consume)
        t.start()
        for f in fs:
            buf.put(f)
        t.join(5)
        self.assertEqual([f.sequence for f in got], list(range(1, 7)))
        self.assertEqual(buf.counters().dropped, 0)

    def test_close_counts_pending_frames_with_cause(self):
        buf = FrameBuffer(4, 0, drop_oldest=True)
        for f in self.frames(3):
            buf.put(f)
        buf.close("SOURCE_LOST", discard_pending=True)
        c = buf.counters()
        self.assertEqual(c.dropped_by_cause, {"SOURCE_LOST": 3})
        self.assertFalse(buf.put(self.frames(1)[0]))  # after close: refused AND counted
        self.assertEqual(buf.counters().dropped_by_cause, {"SOURCE_LOST": 4})
        self.assertTrue(invariant_holds(buf.counters()))


class CaptureConfigTests(TempDirTestCase):
    def test_defaults_valid_and_bounds_enforced(self):
        AppConfig()  # defaults validated by load_config(None)
        load_config(None)
        bad = {"queue_max_frames": 0, "drop_alarm_threshold": -1, "frame_timeout_s": 2.5, "verify_timeout_s": 0,
               "reconnect_interval_s": 0, "reconnect_max_attempts": 1001}
        for key, value in bad.items():
            with self.subTest(key=key):
                p = self.tmp / f"{key}.toml"
                p.write_text(f"[capture]\n{key} = {value}\n", encoding="utf-8")
                with self.assertRaises(ConfigError):
                    load_config(p)
        p = self.tmp / "unknown.toml"
        p.write_text("[capture]\nqueue = 3\n", encoding="utf-8")
        with self.assertRaises(ConfigError):
            load_config(p)


class CaptureAppCase(TempDirTestCase):
    def make(self, cam: SyntheticCamera | None = None, config: CaptureConfig = FAST, source=None):
        self.cam = cam or SyntheticCamera()
        src = source or WebcamSource(0, capture_factory=self.cam.factory)
        self.capture = CaptureService(src, config)
        a = Application(AppConfig(), self.tmp, model_policy=FIXTURE_POLICY, inference=FixtureOkAdapter(),
                        rule=FixtureApprovedRule(), host_checks=HEALTHY_HOST, capture=self.capture)
        a.boot()
        self.addCleanup(a.shutdown)
        self.addCleanup(self.cam.unstall)  # never leave a fake read blocked
        self.transitions: list[Any] = []
        a.state.subscribe(lambda r: self.transitions.append((time.monotonic(), r)))
        return a

    def camera_entry(self, a):
        return next(c for c in a.status()["components"] if c["component"] == "camera")

    def audits(self, a, action):
        return [r for r in a.store.list_audit(200) if r["action"] == action]

    def fault_time(self):
        return next((t for t, r in self.transitions if r.accepted and r.target is SystemState.FAULT), None)


class CameraVerificationTests(CaptureAppCase):  # HLT-002
    def test_camera_unknown_until_verified(self):
        a = self.make()
        cam = self.camera_entry(a)
        self.assertEqual((cam["status"], cam["code"]), ("UNKNOWN", "AEX-CAP-001"))
        self.assertEqual(self.cam.opens, 0, "status never opens the camera")

    def test_unavailable_camera_blocks_observing(self):
        a = self.make(SyntheticCamera(open_ok=False))
        with self.assertRaises(SessionError) as ctx:
            a.start_session()
        self.assertIs(ctx.exception.code, ErrorCode.CAP_001)
        self.assertIs(a.state.state, SystemState.IDLE)
        self.assertIsNone(a.session_id)
        self.assertNotIn(SystemState.OBSERVING, [r.target for _, r in self.transitions if r.accepted])
        self.assertEqual(self.camera_entry(a)["status"], "UNKNOWN")
        audit = self.audits(a, "CAMERA_UNAVAILABLE")
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit[0]["detail"]["code"], "AEX-CAP-001")

    def test_camera_that_opens_but_never_delivers_blocks_observing(self):
        cam = SyntheticCamera()
        cam.stall()
        a = self.make(cam)
        started = time.monotonic()
        with self.assertRaises(SessionError) as ctx:
            a.start_session()
        self.assertIs(ctx.exception.code, ErrorCode.CAP_001)
        self.assertLess(time.monotonic() - started, FAST.verify_timeout_s + 1.0)
        self.assertIs(a.state.state, SystemState.IDLE)

    def test_verified_camera_reaches_observing_and_delivers_frames(self):
        a = self.make()
        sid = a.start_session()
        self.assertIs(a.state.state, SystemState.OBSERVING)
        self.assertEqual(self.camera_entry(a)["status"], "OK")
        seqs = [a.next_frame(2.0).sequence for _ in range(3)]
        self.assertEqual(seqs, sorted(seqs))
        self.assertEqual(self.capture.session_id, sid)


class DropCounterTests(CaptureAppCase):  # CAP-003 acceptance
    def test_fast_source_drop_counter_grows_and_nothing_is_lost_unrecorded(self):
        a = self.make()
        a.start_session()
        self.assertTrue(wait_until(lambda: self.capture.counters().dropped >= 20, 5.0))
        first = self.capture.counters().dropped
        self.assertTrue(wait_until(lambda: self.capture.counters().dropped > first, 5.0), "counter keeps growing")
        delivered = sum(1 for _ in range(3) if a.next_frame(1.0) is not None)
        self.assertEqual(delivered, 3)
        c = self.capture.counters()
        self.assertTrue(invariant_holds(c) or c.captured - (c.delivered + c.dropped + c.depth) == 1,
                        "every captured frame is delivered, queued or counted (one may be mid-put)")
        self.assertEqual(set(c.dropped_by_cause), {DROP_QUEUE_FULL})
        q = next(r for r in a.status()["resources"] if r["component"] == "queue")
        self.assertEqual((q["status"], q["code"]), ("DEGRADED", "AEX-CAP-002"))
        self.assertIn("dropped", q["detail"])
        st = a.status()
        self.assertGreaterEqual(st["counters"]["frames_dropped"], c.dropped)
        self.assertEqual(st["capture"]["dropped_by_cause"].keys(), {DROP_QUEUE_FULL})
        alarm = self.audits(a, "CAPTURE_DROP_LIMIT_EXCEEDED")
        self.assertEqual(len(alarm), 1)
        self.assertEqual(alarm[0]["detail"]["code"], "AEX-CAP-002")
        a.close_session()
        c = self.capture.counters()
        self.assertTrue(invariant_holds(c))
        self.assertEqual(c.depth, 0)

    def test_drops_within_limit_are_counted_but_not_alarmed(self):
        a = self.make(config=dataclasses.replace(FAST, drop_alarm_threshold=1_000_000))
        a.start_session()
        self.assertTrue(wait_until(lambda: self.capture.counters().dropped >= 5, 5.0))
        q = next(r for r in a.status()["resources"] if r["component"] == "queue")
        self.assertEqual(q["status"], "OK")
        self.assertIn("QUEUE_FULL_DROP_OLDEST=", q["detail"])
        self.assertEqual(self.audits(a, "CAPTURE_DROP_LIMIT_EXCEEDED"), [])


class CameraLossTests(CaptureAppCase):  # HLT-004
    def assert_fault_within_target(self, a, injected_at):
        # wait for the recorded transition (listeners run just after the state changes)
        self.assertTrue(wait_until(lambda: self.fault_time() is not None, HLT004_TARGET_S + 1.0))
        elapsed = self.fault_time() - injected_at
        self.assertLess(elapsed, HLT004_TARGET_S, f"FAULT after {elapsed:.3f} s")
        self.assertTrue(wait_until(lambda: self.audits(a, "CAMERA_LOST") != []))
        audit = self.audits(a, "CAMERA_LOST")
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit[0]["detail"]["code"], "AEX-CAP-001")
        self.assertEqual(audit[0]["correlation_id"], self.sid)
        transitions = [r for r in self.audits(a, "STATE_TRANSITION") if r["target"].endswith("->FAULT")]
        self.assertIn("AEX-CAP-001", transitions[0]["detail"]["reason"])
        return elapsed

    def test_disconnect_reaches_fault_within_2s(self):
        a = self.make()
        self.sid = a.start_session()
        a.next_frame(2.0)
        injected = time.monotonic()
        self.cam.disconnect()
        elapsed = self.assert_fault_within_target(a, injected)
        print(f"\n[HLT-004] disconnect -> FAULT: {elapsed * 1000:.1f} ms")

    def test_stalled_camera_reaches_fault_within_2s(self):
        a = self.make()
        self.sid = a.start_session()
        a.next_frame(2.0)
        injected = time.monotonic()
        self.cam.stall()
        elapsed = self.assert_fault_within_target(a, injected)
        self.assertGreaterEqual(elapsed, FAST.frame_timeout_s * 0.9)
        print(f"\n[HLT-004] stall -> FAULT: {elapsed * 1000:.1f} ms (frame_timeout_s={FAST.frame_timeout_s})")

    def test_default_timeout_also_meets_target(self):
        cfg = CaptureConfig(reconnect_max_attempts=0)  # default frame_timeout_s = 1.0
        a = self.make(config=cfg)
        self.sid = a.start_session()
        a.next_frame(2.0)
        injected = time.monotonic()
        self.cam.stall()
        elapsed = self.assert_fault_within_target(a, injected)
        print(f"\n[HLT-004] stall -> FAULT (default 1.0 s timeout): {elapsed * 1000:.1f} ms")

    def test_fault_blocks_frames_and_queued_frames_are_counted(self):
        a = self.make()
        self.sid = a.start_session()
        self.cam.disconnect()
        self.assertTrue(wait_until(lambda: a.state.state is SystemState.FAULT))
        with self.assertRaises(SessionError) as ctx:
            a.next_frame(0.1)
        self.assertIs(ctx.exception.code, ErrorCode.STATE_002)
        with self.assertRaises(CaptureLost):
            self.capture.next_frame(self.sid, 0.1)
        c = self.capture.counters()
        self.assertEqual(c.depth, 0)
        self.assertTrue(invariant_holds(c))


class ReconnectionTests(CaptureAppCase):  # CAP-004
    def test_reconnection_never_resumes_the_old_session(self):
        a = self.make()
        old = a.start_session()
        self.sid = old
        a.next_frame(2.0)
        self.cam.disconnect()
        self.assertTrue(wait_until(lambda: a.state.state is SystemState.FAULT))
        self.assertTrue(wait_until(lambda: self.capture.state is CaptureState.RECONNECTING))
        self.cam.reconnect()
        self.assertTrue(wait_until(lambda: self.capture.state is CaptureState.RECONNECTED, 3.0))
        self.assertIn("new session is required", self.capture.detail)
        # Reconnected, but the old session stays dead: FAULT latched and no frame for it.
        self.assertIs(a.state.state, SystemState.FAULT)
        with self.assertRaises(CaptureLost):
            self.capture.next_frame(old, 0.2)
        with self.assertRaises(StaleSession):
            self.capture.next_frame("never-bound", 0.1)
        with self.assertRaises(SessionError):
            a.next_frame(0.1)
        a.reset_fault("camera replaced on bench", actor_ref="owner")
        self.assertIs(a.state.state, SystemState.IDLE)
        self.assertIsNone(a.session_id)
        new = a.start_session()
        self.assertNotEqual(new, old)
        self.assertIs(a.state.state, SystemState.OBSERVING)
        frame = a.next_frame(2.0)
        self.assertEqual(frame.sequence, 1, "new session = new sequence")
        with self.assertRaises(CaptureLost):
            self.capture.next_frame(old, 0.1)  # still refused after the new session started

    def test_old_session_id_can_never_be_rebound(self):
        a = self.make()
        old = a.start_session()
        self.cam.disconnect()
        self.assertTrue(wait_until(lambda: self.capture.state is CaptureState.RECONNECTING))
        self.cam.reconnect()
        self.assertTrue(wait_until(lambda: self.capture.state is CaptureState.RECONNECTED, 3.0))
        a.reset_fault("bench", actor_ref="owner")
        self.capture.verify()
        with self.assertRaises(StaleSession):
            self.capture.begin(old)

    def test_failed_reconnection_stays_lost_and_start_is_refused(self):
        cfg = CaptureConfig(frame_timeout_s=0.3, verify_timeout_s=0.3, reconnect_interval_s=0.02,
                            reconnect_max_attempts=3)
        a = self.make(config=cfg)
        a.start_session()
        self.cam.disconnect()
        self.assertTrue(wait_until(lambda: self.capture.state is CaptureState.LOST
                                   and "reconnection failed" in self.capture.detail, 3.0))
        self.assertEqual(self.camera_entry(a)["status"], "FAULT")
        a.reset_fault("bench", actor_ref="owner")
        with self.assertRaises(SessionError) as ctx:
            a.start_session()
        self.assertIs(ctx.exception.code, ErrorCode.CAP_001)
        self.assertIs(a.state.state, SystemState.IDLE)


class FiniteSourceSessionTests(CaptureAppCase):
    def test_folder_session_delivers_every_image_then_ends_without_error(self):
        write_image_folder(self.tmp / "imgs", 5)
        a = self.make(source=ImageFolderSource(self.tmp / "imgs"))
        a.start_session()
        seqs = [a.next_frame(2.0).sequence for _ in range(5)]
        self.assertEqual(seqs, [1, 2, 3, 4, 5], "verification does not consume the first image")
        with self.assertRaises(EndOfStream):
            a.next_frame(1.0)
        with self.assertRaises(EndOfStream):
            a.next_frame(0.1)  # stays ended, no repeated frame
        self.assertIs(a.state.state, SystemState.OBSERVING)  # not a fault
        self.assertEqual(self.camera_entry(a)["status"], "OK")
        self.assertEqual(self.capture.counters().dropped, 0)
        self.assertEqual(self.audits(a, "CAMERA_LOST"), [])

    def test_missing_folder_blocks_session(self):
        a = self.make(source=ImageFolderSource(self.tmp / "missing"))
        with self.assertRaises(SessionError):
            a.start_session()
        self.assertIsInstance(self.capture, CaptureService)
        with self.assertRaises(CaptureUnavailable):
            self.capture.verify()


class ReviewRegressionTests(CaptureAppCase):
    """Findings of the independent review (round 1): each failure injected and pinned."""

    def test_h1_worker_crash_is_a_loss_not_a_silent_stop(self):
        a = self.make()
        with mock.patch("acurix_edge.capture.service.FrameStamper.stamp", side_effect=MemoryError):
            self.sid = a.start_session()
            self.assertTrue(wait_until(lambda: self.fault_time() is not None, HLT004_TARGET_S))
        self.assertIn("capture worker failed (MemoryError)", self.audits(a, "CAMERA_LOST")[0]["detail"]["detail"])
        self.assertNotEqual(self.camera_entry(a)["status"], "OK")

    def test_m1_unexpected_verification_error_returns_to_idle(self):
        a = self.make()
        with mock.patch.object(CaptureService, "_probe_source", side_effect=OSError("share gone")):
            with self.assertRaises(SessionError) as ctx:
                a.start_session()
        self.assertIs(ctx.exception.code, ErrorCode.CAP_001)
        self.assertIs(a.state.state, SystemState.IDLE)
        cam = SyntheticCamera()

        def bad_set(*_a):
            raise RuntimeError("driver refused property")

        def factory(target, *_args):
            c = cam.factory(target)
            c.set = bad_set  # opening succeeds, configuring the camera fails
            return c

        src = WebcamSource(1, width=320, capture_factory=factory)
        with self.assertRaises(SourceOpenError):
            src.open()
        self.assertFalse(src.is_open)

    def test_m4_begin_failure_never_leaves_observing_without_session(self):
        a = self.make()
        with mock.patch.object(CaptureService, "begin", side_effect=RuntimeError("can't start new thread")):
            with self.assertRaises(RuntimeError):
                a.start_session()
        self.assertIs(a.state.state, SystemState.IDLE)
        self.assertIsNone(a.session_id)
        self.assertEqual(len(self.audits(a, "SESSION_START_ABORTED")), 1)
        a.start_session()  # the station is usable again
        self.assertIs(a.state.state, SystemState.OBSERVING)

    def test_m2_reconnection_errors_end_in_final_lost(self):
        a = self.make(config=dataclasses.replace(FAST, reconnect_max_attempts=3, reconnect_interval_s=0.02))
        a.start_session()
        with mock.patch.object(CaptureService, "_probe_source", side_effect=OSError("usb hub")):
            self.cam.disconnect()
            self.assertTrue(wait_until(lambda: "reconnection failed" in self.capture.detail, 3.0))
        self.assertIs(self.capture.state, CaptureState.LOST)
        with mock.patch.object(CaptureService, "_try_reconnect", side_effect=RuntimeError("boom")):
            self.capture._reconnect(self.capture._gen)
        self.assertIs(self.capture.state, CaptureState.LOST)
        self.assertIn("reconnection aborted", self.capture.detail)

    def test_m3_close_never_releases_during_an_inflight_read(self):
        cam = SyntheticCamera()
        src = WebcamSource(0, capture_factory=cam.factory)
        src.open()
        cap = src._cap
        cam.stall()
        errors = []

        def reader():
            try:
                src.read()
            except Exception as exc:
                errors.append(exc)

        t = threading.Thread(target=reader)
        t.start()
        time.sleep(0.1)
        src.close()
        self.assertTrue(cap.isOpened(), "release deferred while read is in flight")
        cam.unstall()
        t.join(2)
        self.assertFalse(cap.isOpened(), "released by the reader once its read returned")
        self.assertEqual([type(e).__name__ for e in errors], ["SourceError"])

    def test_b2_normal_session_end_is_counted_but_not_alarmed(self):
        write_image_folder(self.tmp / "imgs", 8)
        a = self.make(source=ImageFolderSource(self.tmp / "imgs"))
        sid = a.start_session()
        self.assertTrue(wait_until(lambda: self.capture.counters().depth == FAST.queue_max_frames))
        a.close_session()
        c = self.capture.counters()
        # the queued frames, plus possibly one the producer held while waiting for space
        self.assertIn(c.dropped_by_cause.get("SESSION_ENDED"), (FAST.queue_max_frames, FAST.queue_max_frames + 1))
        self.assertEqual(set(c.dropped_by_cause), {"SESSION_ENDED"})
        self.assertFalse(c.over_limit)
        self.assertEqual(self.audits(a, "CAPTURE_DROP_LIMIT_EXCEEDED"), [])
        self.assertTrue(invariant_holds(c))
        with self.assertRaises(StaleSession):
            self.capture.next_frame(sid, 0.1)

    def test_l3_watchdog_crash_is_a_loss(self):
        a = self.make()
        with mock.patch.object(CaptureService, "_watch", side_effect=RuntimeError("clock broke")):
            a.start_session()
            self.assertTrue(wait_until(lambda: self.fault_time() is not None, HLT004_TARGET_S))
        self.assertIn("watchdog failed", self.capture.detail + str(self.audits(a, "CAMERA_LOST")))

    def test_l5_start_aborts_to_idle_even_if_capture_end_raises(self):
        a = self.make()
        with mock.patch.object(CaptureService, "begin", side_effect=RuntimeError("begin")), \
                mock.patch.object(CaptureService, "end", side_effect=RuntimeError("end")):
            with self.assertRaises(RuntimeError):
                a.start_session()
        self.assertIs(a.state.state, SystemState.IDLE)
        self.assertIsNone(a.session_id)

    def test_b4_frame_timeout_capped_below_target(self):
        p = self.tmp / "t.toml"
        p.write_text("[capture]\nframe_timeout_s = 1.95\n", encoding="utf-8")
        with self.assertRaises(ConfigError):
            load_config(p)


if __name__ == "__main__":
    unittest.main()
