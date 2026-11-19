"""Regression (20/11/2026, real USB webcam): on Windows a DirectShow capture object stops working
when the thread that CREATED it ends (measured: get -> 0x0 and read fails). The bounded open of
AEX-CAPTURE-007/Ordem 5 created it in a short-lived thread, which broke every real camera.

`AffineCamera` reproduces that behaviour: the bounded open must keep the creating thread alive
while the object exists, and let it end only after the release. Synthetic images only."""

import threading
import time
import unittest
from unittest import mock

import cv2

from acurix_edge.capture.profile import CameraProfile
from acurix_edge.capture.service import CaptureService
from acurix_edge.capture.sources import SourceOpenError, WebcamSource, open_with_timeout
from acurix_edge.config.settings import CaptureConfig
from tests.capture_support import profile_config, synth_image


def wait_until(pred, timeout=3.0, step=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(step)
    return bool(pred())


class AffineCap:
    def __init__(self, report_size=True):
        self.creator = threading.current_thread()
        self.is_open = True
        self.report_size = report_size  # False: default size is reported, sizes after a set() are not
        self.creator_alive_at_release = None
        self.sets = 0
        self.read_gate: threading.Event | None = None  # not None -> read blocks until set

    def _usable(self):
        return self.is_open and self.creator.is_alive()

    def isOpened(self):  # noqa: N802
        return self.is_open

    def getBackendName(self):  # noqa: N802
        return "AFFINE"

    def get(self, prop):
        if not self._usable() or (not self.report_size and self.sets > 0):
            return 0.0
        return {cv2.CAP_PROP_FRAME_WIDTH: 64.0, cv2.CAP_PROP_FRAME_HEIGHT: 48.0}.get(prop, -1.0)

    def set(self, _prop, _value):
        self.sets += 1
        return self._usable()

    def read(self):
        if self.read_gate is not None:
            self.read_gate.wait(5)
        return (True, synth_image(0)) if self._usable() else (False, None)

    def release(self):
        self.creator_alive_at_release = self.creator.is_alive()
        self.is_open = False


class AffineCamera:
    def __init__(self, report_size=True):
        self.caps = []
        self.report_size = report_size

    def factory(self, _target, *_args):
        cap = AffineCap(self.report_size)
        self.caps.append(cap)
        return cap


class ThreadAffinityTests(unittest.TestCase):
    def test_fake_reproduces_the_driver(self):
        """Control: an object created in a thread that ENDS is dead (the measured DirectShow behaviour)."""
        cam = AffineCamera()
        t = threading.Thread(target=cam.factory, args=(0,))
        t.start()
        t.join()
        self.assertEqual(cam.caps[0].read(), (False, None))
        self.assertEqual(cam.caps[0].get(cv2.CAP_PROP_FRAME_WIDTH), 0.0)

    def test_bounded_open_keeps_the_object_usable(self):
        cam = AffineCamera()
        src = WebcamSource(0, capture_factory=cam.factory)
        open_with_timeout(src, 2.0)
        self.assertTrue(cam.caps[0].creator.is_alive(), "creating thread kept alive")
        self.assertIsNotNone(src.read(), "reads work from another thread")
        profile = src.measure_profile()
        self.assertEqual((profile.default_resolution, profile.supported_resolutions), ("64x48", ("64x48",)))
        creator = cam.caps[0].creator
        src.close()
        self.assertTrue(cam.caps[0].creator_alive_at_release, "released BEFORE its thread ends")
        self.assertTrue(wait_until(lambda: not creator.is_alive()), "and then the thread ends (no leak)")

    def test_session_on_a_thread_affine_camera_delivers_frames(self):
        cam = AffineCamera()
        prof = CameraProfile("AFFINE", "64x48", ("64x48",), "webcam-usb-0")
        svc = CaptureService(WebcamSource(0, capture_factory=cam.factory),
                             CaptureConfig(frame_timeout_s=0.5, verify_timeout_s=1.0, **profile_config(prof)))
        svc.verify()
        svc.begin("s1")
        try:
            frame = svc.next_frame("s1", 2.0)
            self.assertIsNotNone(frame, "the capture worker reads frames (would fail before the fix)")
        finally:
            svc.end("s1")
        self.assertTrue(wait_until(lambda: not cam.caps[0].creator.is_alive()))
        svc.shutdown()

    def test_driver_reporting_no_resolution_is_refused(self):
        cam = AffineCamera(report_size=False)
        src = WebcamSource(0, capture_factory=cam.factory)
        open_with_timeout(src, 2.0)
        try:
            with self.assertRaises(SourceOpenError) as ctx:
                src.measure_profile()
            self.assertIn("reported no resolution", str(ctx.exception))
        finally:
            src.close()

    def test_interrupt_between_open_and_install_releases_the_camera(self):  # review CAPTURE-008, M1
        cam = AffineCamera()
        src = WebcamSource(0, capture_factory=cam.factory)
        with mock.patch.object(WebcamSource, "_install_unless_closed", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                open_with_timeout(src, 2.0)
        self.assertFalse(src.is_open)
        self.assertFalse(cam.caps[0].is_open, "camera released, not left open")
        self.assertTrue(wait_until(lambda: not cam.caps[0].creator.is_alive()), "creating thread not stuck")

    def test_close_during_the_open_discards_and_ends_the_thread(self):  # review CAPTURE-008, B3
        gate = threading.Event()
        cam = AffineCamera()

        def slow(target, *args):
            gate.wait(5)
            return cam.factory(target, *args)

        src = WebcamSource(0, capture_factory=slow)
        errors: list[BaseException] = []
        opener = threading.Thread(target=lambda: self._try(lambda: open_with_timeout(src, 3.0), errors))
        opener.start()
        time.sleep(0.1)
        src.close()
        gate.set()
        opener.join(5)
        self.assertEqual([type(e).__name__ for e in errors], ["SourceOpenError"])
        self.assertFalse(src.is_open)
        self.assertTrue(wait_until(lambda: cam.caps and not cam.caps[0].is_open))
        self.assertTrue(wait_until(lambda: not cam.caps[0].creator.is_alive()))

    def test_deferred_release_after_an_inflight_read_ends_the_thread(self):  # review CAPTURE-008, B3
        cam = AffineCamera()
        src = WebcamSource(0, capture_factory=cam.factory)
        open_with_timeout(src, 2.0)
        cap = cam.caps[0]
        cap.read_gate = threading.Event()
        reader = threading.Thread(target=lambda: self._try(src.read, []))
        reader.start()
        time.sleep(0.1)
        src.close()  # read in flight: release deferred to the reader
        self.assertTrue(cap.is_open, "not released during the native read")
        self.assertTrue(cap.creator.is_alive())
        cap.read_gate.set()
        reader.join(5)
        self.assertTrue(wait_until(lambda: not cap.is_open))
        self.assertTrue(wait_until(lambda: not cap.creator.is_alive()), "creating thread ends after the release")

    @staticmethod
    def _try(fn, errors):
        try:
            fn()
        except BaseException as exc:
            errors.append(exc)

    def test_abandoned_late_open_releases_and_ends_its_thread(self):
        gate = threading.Event()
        cam = AffineCamera()

        def slow(target, *args):
            gate.wait(5)
            return cam.factory(target, *args)

        src = WebcamSource(0, capture_factory=slow)
        with self.assertRaises(SourceOpenError):
            open_with_timeout(src, 0.2)
        gate.set()
        self.assertTrue(wait_until(lambda: len(cam.caps) == 1 and not cam.caps[0].is_open))
        self.assertTrue(wait_until(lambda: not cam.caps[0].creator.is_alive()), "late opener does not linger")
        self.assertFalse(src.is_open)


if __name__ == "__main__":
    unittest.main()
