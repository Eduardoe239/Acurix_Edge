"""AEX-CAPTURE-006 — CAP-001/CAP-002: one source contract, three implementations.

The SAME contract tests run against the image folder, the video file and the webcam source (the
webcam backed by a synthetic camera; the real camera is never opened). Every frame becomes a
valid CapturedFrame 1.1 with increasing sequence, increasing monotonic time, UTC, source and
resolution; termination is explicit and never repeats a frame."""

import unittest
from typing import TYPE_CHECKING

import numpy as np

from acurix_edge.capture.frames import FrameStamper, InvalidImageError, check_image, quality_metrics
from acurix_edge.capture.sources import (
    FrameSource,
    Image,
    ImageFolderSource,
    SourceError,
    SourceLostError,
    SourceOpenError,
    VideoFileSource,
    WebcamSource,
)
from acurix_edge.contracts.capture import CapturedFrame, CaptureStatus
from acurix_edge.contracts.common import validate_utc
from acurix_edge.contracts.inference import ImageQuality
from tests.capture_support import H, SyntheticCamera, W, synth_image, write_image_folder, write_video
from tests.support import TempDirTestCase

N = 5


# The mixin is typed as a TestCase for mypy only; at runtime it is a plain object (not collected).
if TYPE_CHECKING:
    _ContractBase = unittest.TestCase
else:
    _ContractBase = object


class SourceContract(_ContractBase):
    """Mixin: subclasses provide make_source(n) and make_missing()."""

    def make_source(self, n: int) -> FrameSource:
        raise NotImplementedError

    def make_missing(self) -> FrameSource:
        raise NotImplementedError

    def read_all(self, src: FrameSource, n: int) -> list[Image | None]:
        src.open()
        self.addCleanup(src.close)
        return [src.read() for _ in range(n)]

    # -- the contract ------------------------------------------------------------------
    def test_implements_protocol(self):
        src = self.make_source(N)
        self.assertIsInstance(src, FrameSource)
        self.assertTrue(src.source_id)

    def test_frames_are_valid_captured_frames_with_monotonic_sequence(self):
        src = self.make_source(N)
        stamper = FrameStamper(src.source_id)
        frames: list[CapturedFrame] = [stamper.stamp(img, 1.0) for img in self.read_all(src, N)]
        self.assertEqual([f.sequence for f in frames], list(range(1, N + 1)))
        monos = [f.monotonic_ns for f in frames]
        self.assertEqual(monos, sorted(set(monos)), "monotonic_ns must strictly increase")
        for f in frames:
            validate_utc(f.captured_at_utc)
            self.assertEqual(f.source_id, src.source_id)
            self.assertEqual((f.input.width, f.input.height), (W, H))
            self.assertIs(f.input.quality, ImageQuality.UNKNOWN)  # metrics only; the gate is 0.3.0
            self.assertIs(f.status, CaptureStatus.OK)
            self.assertIsNotNone(f.quality_metrics)
            self.assertEqual(CapturedFrame.from_dict(f.to_dict()), f)

    def test_open_failure_is_explicit(self):
        with self.assertRaises(SourceOpenError):
            self.make_missing().open()

    def test_read_before_open_and_after_close_is_an_error(self):
        src = self.make_source(N)
        with self.assertRaises(SourceError):
            src.read()
        src.open()
        src.close()
        src.close()  # idempotent
        self.assertFalse(src.is_open)
        with self.assertRaises(SourceError):
            src.read()

    def test_termination_is_explicit_and_never_repeats(self):
        src = self.make_source(N)
        images = self.read_all(src, N)
        self.assertTrue(all(img is not None for img in images))
        if src.live:  # a camera never "ends": no more frames = loss
            with self.assertRaises(SourceLostError):
                src.read()
        else:
            for _ in range(3):
                self.assertIsNone(src.read())  # end of stream, stable, no error, no repeated frame


class ImageFolderContractTests(SourceContract, TempDirTestCase):
    def make_source(self, n):
        self.originals = write_image_folder(self.tmp / "imgs", n)
        return ImageFolderSource(self.tmp / "imgs")

    def make_missing(self):
        return ImageFolderSource(self.tmp / "nope")

    def test_lossless_order_and_content(self):
        src = self.make_source(N)
        for got, expected in zip(self.read_all(src, N), self.originals, strict=False):
            np.testing.assert_array_equal(got, expected)

    def test_empty_folder_and_undecodable_file(self):
        (self.tmp / "empty").mkdir()
        with self.assertRaises(SourceOpenError):
            ImageFolderSource(self.tmp / "empty").open()
        bad = self.tmp / "bad"
        bad.mkdir()
        (bad / "a.png").write_bytes(b"not a png")
        src = ImageFolderSource(bad)
        src.open()
        with self.assertRaises(SourceLostError):
            src.read()


class VideoFileContractTests(SourceContract, TempDirTestCase):
    def make_source(self, n):
        write_video(self.tmp / "v.avi", n)
        return VideoFileSource(self.tmp / "v.avi")

    def make_missing(self):
        return VideoFileSource(self.tmp / "missing.avi")


class WebcamContractTests(SourceContract, unittest.TestCase):
    """The synthetic camera plays the webcam (real camera only with owner authorization)."""

    def make_source(self, n):
        self.cam = SyntheticCamera(frames=n)
        return WebcamSource(0, capture_factory=self.cam.factory)

    def make_missing(self):
        return WebcamSource(3, capture_factory=SyntheticCamera(open_ok=False).factory)

    def test_invalid_device_index(self):
        for bad in (-1, True, "0"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                WebcamSource(bad)


class QualityMetricsTests(unittest.TestCase):
    def test_brightness_orders_dark_and_bright(self):
        dark = quality_metrics(np.full((H, W, 3), 10, np.uint8))
        bright = quality_metrics(np.full((H, W, 3), 240, np.uint8))
        self.assertAlmostEqual(dark.mean_brightness, 10.0, places=3)
        self.assertAlmostEqual(bright.mean_brightness, 240.0, places=3)

    def test_sharpness_orders_sharp_and_blurred(self):
        import cv2

        sharp = synth_image(3)
        blurred = cv2.GaussianBlur(sharp, (9, 9), 4)
        self.assertGreater(quality_metrics(sharp).sharpness, quality_metrics(blurred).sharpness)
        self.assertEqual(quality_metrics(np.zeros((H, W), np.uint8)).sharpness, 0.0)

    def test_grayscale_and_bgra_accepted(self):
        quality_metrics(np.zeros((H, W), np.uint8))
        quality_metrics(np.zeros((H, W, 4), np.uint8))
        quality_metrics(np.zeros((H, W, 1), np.uint8))

    def test_invalid_images_rejected(self):
        for bad in (None, b"x", np.zeros((H, W), np.float32), np.zeros((0, W, 3), np.uint8),
                    np.zeros((H, W, 2), np.uint8), np.zeros((2, 2, 2, 2), np.uint8)):
            with self.subTest(bad=type(bad).__name__), self.assertRaises(InvalidImageError):
                check_image(bad)

    def test_stamper_forces_strictly_increasing_monotonic_time(self):
        stamper = FrameStamper("s", monotonic_ns=lambda: 42)  # a frozen clock
        a, b = stamper.stamp(synth_image(0), 0.0), stamper.stamp(synth_image(1), 0.0)
        self.assertLess(a.monotonic_ns, b.monotonic_ns)


if __name__ == "__main__":
    unittest.main()
