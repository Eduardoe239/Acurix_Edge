"""AEX-CAPTURE-006/007 — CAP-003 (bounded buffer, counted drops), HLT-002 (camera verified before
OBSERVING), HLT-004 (loss -> FAULT within 2 s), CAP-004 (a lost camera is never reopened by
itself; recovery = reset-fault + new session) and the camera profile check (owner decision
19/11/2026: a different camera is refused with AEX-CAP-001, never reaches OBSERVING).

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
from acurix_edge.capture.profile import MISMATCH_MESSAGE, NOT_CONFIGURED_MESSAGE, CameraMismatch
from acurix_edge.capture.service import CaptureService, CaptureState
from acurix_edge.capture.sources import ImageFolderSource, SourceOpenError, WebcamSource
from acurix_edge.config.settings import AppConfig, CaptureConfig, ConfigError, load_config
from acurix_edge.contracts.errors import ErrorCode
from acurix_edge.domain.state_machine import SystemState
from tests.capture_support import SyntheticCamera, profile_config, synth_image, write_image_folder
from tests.support import FIXTURE_POLICY, HEALTHY_HOST, FixtureApprovedRule, FixtureOkAdapter, TempDirTestCase

HLT004_TARGET_S = 2.0
SYNTH_PROFILE = SyntheticCamera().profile()  # what WebcamSource(0) measures on the default synthetic camera
FAST = CaptureConfig(queue_max_frames=4, drop_alarm_threshold=0, frame_timeout_s=0.3, verify_timeout_s=1.0,
                     **profile_config(SYNTH_PROFILE))


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
        bad = {"queue_max_frames": 0, "drop_alarm_threshold": -1, "frame_timeout_s": 2.5, "verify_timeout_s": 0}
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

    def test_config_with_utf8_bom_is_accepted(self):
        p = self.tmp / "bom.toml"
        p.write_bytes(b"\xef\xbb\xbf" + b"[capture]\nqueue_max_frames = 3\n")
        self.assertEqual(load_config(p).capture.queue_max_frames, 3)

    def test_reconnect_keys_are_refused_with_the_reason(self):
        for key in ("reconnect_max_attempts = 10", "reconnect_interval_s = 1.0"):
            with self.subTest(key=key):
                p = self.tmp / "old.toml"
                p.write_text(f"[capture]\n{key}\n", encoding="utf-8")
                with self.assertRaises(ConfigError) as ctx:
                    load_config(p)
                self.assertIn("never reopened automatically", str(ctx.exception))

    def test_camera_profile_all_or_nothing_and_well_formed(self):
        good = ('camera_backend = "DSHOW"\ncamera_default_resolution = "640x480"\n'
                'camera_resolutions = ["640x480", "320x240"]\ncamera_device_label = "GENERAL WEBCAM"\n')
        p = self.tmp / "good.toml"
        p.write_text("[capture]\n" + good, encoding="utf-8")
        cfg = load_config(p)
        self.assertEqual(cfg.capture.camera_backend, "DSHOW")
        cases = {
            "partial": 'camera_backend = "DSHOW"\n',
            "bad resolution": good.replace('"640x480"\ncamera_res', '"640*480"\ncamera_res'),
            "duplicate resolutions": good.replace('"320x240"]', '"640x480"]'),
            "empty list": good.replace('["640x480", "320x240"]', "[]"),
            "label outer spaces": good.replace('"GENERAL WEBCAM"', '" GENERAL WEBCAM"'),
            "bad backend": good.replace('"DSHOW"', '"DS HOW"'),
        }
        for name, body in cases.items():
            with self.subTest(case=name):
                p.write_text("[capture]\n" + body, encoding="utf-8")
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
        cfg = CaptureConfig(**profile_config(SYNTH_PROFILE))  # default frame_timeout_s = 1.0
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


class NoAutomaticReconnectionTests(CaptureAppCase):  # CAP-004, owner decision 19/11/2026 (option a)
    def test_lost_camera_is_never_reopened_and_recovery_needs_reset_and_new_session(self):
        a = self.make()
        old = a.start_session()
        self.sid = old
        a.next_frame(2.0)
        opens_before_loss = self.cam.opens
        self.cam.disconnect()
        self.assertTrue(wait_until(lambda: a.state.state is SystemState.FAULT))
        self.cam.reconnect()  # the cable is back...
        time.sleep(0.5)       # ...and nothing reopens the source by itself
        self.assertEqual(self.cam.opens, opens_before_loss, "no automatic reopen after a loss")
        self.assertIs(self.capture.state, CaptureState.LOST)
        self.assertFalse(self.capture.source.is_open, "lost source is closed")
        self.assertIn("not reopened automatically", self.capture.detail)
        self.assertEqual(self.camera_entry(a)["status"], "FAULT")
        self.assertIs(a.state.state, SystemState.FAULT)
        with self.assertRaises(CaptureLost):
            self.capture.next_frame(old, 0.2)
        with self.assertRaises(StaleSession):
            self.capture.next_frame("never-bound", 0.1)
        with self.assertRaises(SessionError):
            a.next_frame(0.1)
        with self.assertRaises(SessionError):
            a.start_session()  # FAULT latched: no new session without reset-fault
        self.assertEqual(self.cam.opens, opens_before_loss)
        a.reset_fault("camera reconnected on bench", actor_ref="owner")
        self.assertIs(a.state.state, SystemState.IDLE)
        self.assertIsNone(a.session_id)
        self.assertEqual(self.cam.opens, opens_before_loss, "reset-fault alone does not open the camera")
        new = a.start_session()
        self.assertEqual(self.cam.opens, opens_before_loss + 1, "only the new session opens the camera")
        self.assertNotEqual(new, old)
        self.assertIs(a.state.state, SystemState.OBSERVING)
        frame = a.next_frame(2.0)
        # new session = sequence restarts at 1; with a fast live source the oldest frames may be
        # dropped (and counted) before the first get, so the first delivered is 1 + drops so far.
        self.assertLessEqual(frame.sequence, 1 + self.capture.counters().dropped, "new session = new sequence")
        with self.assertRaises(CaptureLost):
            self.capture.next_frame(old, 0.1)  # still refused after the new session started
        reset = self.audits(a, "FAULT_RESET")
        self.assertEqual(len(reset), 1)
        self.assertEqual(reset[0]["actor_ref"], "owner")

    def test_old_session_id_can_never_be_rebound(self):
        a = self.make()
        old = a.start_session()
        self.cam.disconnect()
        self.assertTrue(wait_until(lambda: self.capture.state is CaptureState.LOST))
        self.cam.reconnect()
        a.reset_fault("bench", actor_ref="owner")
        self.capture.verify()
        with self.assertRaises(StaleSession):
            self.capture.begin(old)

    def test_camera_still_absent_after_reset_refuses_the_new_session(self):
        a = self.make()
        a.start_session()
        self.cam.disconnect()
        self.assertTrue(wait_until(lambda: self.capture.state is CaptureState.LOST))
        a.reset_fault("bench", actor_ref="owner")
        with self.assertRaises(SessionError) as ctx:
            a.start_session()
        self.assertIs(ctx.exception.code, ErrorCode.CAP_001)
        self.assertIs(a.state.state, SystemState.IDLE)


class SwitchableCamera:
    """What a device index points to can change (DirectShow renumbers on unplug)."""

    def __init__(self, cam: SyntheticCamera) -> None:
        self.current = cam

    def factory(self, target, *args):
        return self.current.factory(target, *args)


class CameraProfileTests(CaptureAppCase):  # owner decision 19/11/2026 (option b)
    def assert_refused(self, a, fragment):
        before = len(self.transitions)
        with self.assertRaises(SessionError) as ctx:
            a.start_session()
        self.assertIs(ctx.exception.code, ErrorCode.CAP_001)
        self.assertIn(fragment, str(ctx.exception))
        self.assertIs(a.state.state, SystemState.IDLE)
        self.assertIsNone(a.session_id)
        self.assertNotIn(SystemState.OBSERVING, [r.target for _, r in self.transitions[before:] if r.accepted])
        audit = self.audits(a, "CAMERA_UNAVAILABLE")[0]  # list_audit: newest first
        self.assertEqual(audit["detail"]["code"], "AEX-CAP-001")
        self.assertIn(fragment, audit["detail"]["detail"])
        self.assertFalse(self.capture.source.is_open, "refused camera is released")

    def test_same_camera_is_accepted(self):
        a = self.make()
        self.capture.verify()
        self.assertIn("camera profile matched", self.camera_entry(a)["detail"])
        self.capture.end()
        a.start_session()
        self.assertIs(a.state.state, SystemState.OBSERVING)
        self.assertEqual(self.capture.profile, SYNTH_PROFILE)
        frame = a.next_frame(2.0)
        self.assertEqual(frame.pixels.shape[:2], (48, 64), "configured/default size restored after profiling")

    def test_other_camera_is_refused(self):
        variants = {"backend": SyntheticCamera(backend="MSMF"),
                    "resolutions": SyntheticCamera(sizes=((64, 48), (128, 96))),
                    "default resolution": SyntheticCamera(sizes=((32, 24), (96, 72)))}
        for name, cam in variants.items():
            with self.subTest(differs=name):
                a = self.make(cam)
                self.assert_refused(a, MISMATCH_MESSAGE)
                self.assertIn(name, self.audits(a, "CAMERA_UNAVAILABLE")[0]["detail"]["detail"])

    def test_other_device_label_is_refused(self):
        a = self.make(source=WebcamSource(0, capture_factory=SyntheticCamera().factory, device_label="Other"))
        self.assert_refused(a, "device label")

    def test_no_expected_profile_is_refused(self):
        a = self.make(config=CaptureConfig(frame_timeout_s=0.3, verify_timeout_s=1.0))
        self.assert_refused(a, NOT_CONFIGURED_MESSAGE)

    def test_camera_swapped_while_lost_is_refused_on_the_new_session(self):
        """The real-hardware finding of 19/11/2026: the USB camera is unplugged and the same index
        now opens the integrated camera. The new session must be refused, never OBSERVING on it."""
        switch = SwitchableCamera(SyntheticCamera())
        a = self.make(source=WebcamSource(0, capture_factory=switch.factory))
        self.sid = a.start_session()
        a.next_frame(2.0)
        switch.current.disconnect()
        self.assertTrue(wait_until(lambda: a.state.state is SystemState.FAULT))
        switch.current = SyntheticCamera(backend="SYNTH", sizes=((64, 48), (80, 60)))  # another camera
        a.reset_fault("bench", actor_ref="owner")
        self.assert_refused(a, MISMATCH_MESSAGE)
        switch.current = SyntheticCamera()  # the configured camera is back
        a.start_session()
        self.assertIs(a.state.state, SystemState.OBSERVING)

    def test_folder_source_is_not_profiled(self):
        write_image_folder(self.tmp / "imgs", 2)
        a = self.make(source=ImageFolderSource(self.tmp / "imgs"),
                      config=CaptureConfig(frame_timeout_s=0.3, verify_timeout_s=1.0))
        a.start_session()
        self.assertIsNone(self.capture.profile)
        self.assertIs(a.state.state, SystemState.OBSERVING)

    def test_stuck_profile_measurement_is_bounded_and_never_touches_the_next_camera_object(self):
        """Review M1/M2: the driver blocks inside cap.set during profiling. The start is refused
        within profile_timeout_s; the next session opens a NEW capture object that the stale
        measurement thread must never touch once it is released."""
        cam = SyntheticCamera()
        a = self.make(cam, config=dataclasses.replace(FAST, profile_timeout_s=0.3))
        gate = threading.Event()
        self.addCleanup(gate.set)
        cam.set_gate = gate
        started = time.monotonic()
        with self.assertRaises(SessionError) as ctx:
            a.start_session()
        self.assertLess(time.monotonic() - started, 0.3 + 1.0, "bounded by profile_timeout_s")
        self.assertIn("camera profile", str(ctx.exception))
        self.assertIs(a.state.state, SystemState.IDLE)
        stale_cap = cam.caps[-1]
        self.assertTrue(stale_cap.isOpened(), "release deferred: a native call is still in flight")
        cam.set_gate = None
        a.start_session()
        self.assertIs(a.state.state, SystemState.OBSERVING)
        new_cap = cam.caps[-1]
        self.assertIsNot(new_cap, stale_cap)
        sets_before = new_cap.set_calls
        gate.set()  # the stuck driver call returns
        self.assertTrue(wait_until(lambda: not stale_cap.isOpened(), 2.0), "stale object released afterwards")
        time.sleep(0.2)
        self.assertEqual(new_cap.set_calls, sets_before, "stale measurement never touches the new object")
        self.assertEqual(a.next_frame(2.0).pixels.shape[:2], (48, 64))

    def test_close_during_profile_measurement_does_not_block_and_aborts(self):
        cam = SyntheticCamera()
        src = WebcamSource(0, capture_factory=cam.factory)
        src.open()
        cap = src._cap
        gate = threading.Event()
        self.addCleanup(gate.set)
        cam.set_gate = gate
        errors: list[BaseException] = []

        def measure():
            try:
                src.measure_profile()
            except BaseException as exc:
                errors.append(exc)

        t = threading.Thread(target=measure)
        t.start()
        time.sleep(0.1)
        t0 = time.monotonic()
        src.close()
        self.assertLess(time.monotonic() - t0, 0.5, "close never waits for a stuck driver call (M1)")
        self.assertTrue(cap.isOpened(), "release deferred while the native call is in flight")
        gate.set()
        t.join(2)
        self.assertFalse(cap.isOpened(), "released when the measurement let go of it")
        self.assertEqual([type(e).__name__ for e in errors], ["SourceError"])

    def test_failing_size_restore_never_leaks_the_handle(self):  # review 2, M-1
        cam = SyntheticCamera()
        src = WebcamSource(0, capture_factory=cam.factory)
        src.open()
        cap = src._cap
        calls = {"n": 0}
        original_set = cap.set

        def flaky_set(prop, value):
            calls["n"] += 1
            if calls["n"] > 2 * 5:  # the 5 candidates (width+height) pass; the restore fails
                raise RuntimeError("driver refused restore")
            return original_set(prop, value)

        cap.set = flaky_set
        profile = src.measure_profile()  # restore failure is logged, never masks nor raises
        self.assertEqual(profile, cam.profile())
        src.close()
        self.assertFalse(cap.isOpened(), "handle released: _unpin ran despite the failing restore")

    def test_driver_without_default_resolution_is_refused(self):  # review 2, B-2
        cam = SyntheticCamera()

        def factory(target, *args):
            c = cam.factory(target, *args)
            c.get = lambda prop: 0.0
            return c

        a = self.make(source=WebcamSource(0, capture_factory=factory))
        self.assert_refused(a, "did not report a default resolution")

    def test_live_source_without_profile_support_is_refused(self):
        class LiveFolder(ImageFolderSource):
            live = True

        write_image_folder(self.tmp / "imgs", 2)
        a = self.make(source=LiveFolder(self.tmp / "imgs"))
        self.assert_refused(a, "without camera profile support")

    def test_mismatch_error_is_a_source_open_error(self):
        self.assertTrue(issubclass(CameraMismatch, SourceOpenError))


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

    def test_m2_loss_is_final_and_probe_is_never_called_after_it(self):
        a = self.make()
        a.start_session()
        with mock.patch.object(CaptureService, "_probe_source", side_effect=AssertionError("reopened")) as probe:
            self.cam.disconnect()
            self.assertTrue(wait_until(lambda: self.capture.state is CaptureState.LOST))
            time.sleep(0.3)
        probe.assert_not_called()
        self.assertIs(self.capture.state, CaptureState.LOST)

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
