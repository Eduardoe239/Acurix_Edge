"""Turns a raw image into a `CapturedFrame` (CAP-002): sequence, monotonic + UTC timestamps,
source, resolution and basic quality measurements. Quality stays UNKNOWN (gate is 0.3.0)."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable

import cv2
import numpy as np

from acurix_edge.contracts.capture import CapturedFrame, CaptureStatus, FrameQualityMetrics
from acurix_edge.contracts.common import utc_now_iso
from acurix_edge.contracts.inference import MAX_IMAGE_SIDE, ImageQuality, InputDescriptor


class InvalidImageError(ValueError):
    """The source delivered something that is not a usable 8-bit image."""


def check_image(image: object) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise InvalidImageError(f"image must be numpy.ndarray, got {type(image).__name__}")
    if image.dtype != np.uint8:
        raise InvalidImageError(f"image dtype must be uint8, got {image.dtype}")
    if image.ndim == 3 and image.shape[2] in (1, 3, 4):
        pass
    elif image.ndim != 2:
        raise InvalidImageError(f"unsupported image shape {image.shape}")
    h, w = image.shape[:2]
    if not (0 < h <= MAX_IMAGE_SIDE and 0 < w <= MAX_IMAGE_SIDE):
        raise InvalidImageError(f"image size {w}x{h} outside 1..{MAX_IMAGE_SIDE}")
    return image


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    channels = image.shape[2]
    if channels == 1:
        return image[:, :, 0]
    return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY if channels == 4 else cv2.COLOR_BGR2GRAY)


def quality_metrics(image: np.ndarray) -> FrameQualityMetrics:
    """Mean brightness (0..255) and variance of the Laplacian (simple sharpness)."""
    gray = _gray(check_image(image))
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return FrameQualityMetrics(mean_brightness=min(brightness, 255.0), sharpness=max(sharpness, 0.0))


class FrameStamper:
    """One per source per session. Sequence starts at 1 and strictly increases; monotonic_ns is
    forced strictly increasing too (a coarse clock can return equal readings)."""

    def __init__(self, source_id: str, *, monotonic_ns: Callable[[], int] = time.monotonic_ns,
                 utc_now: Callable[[], str] = utc_now_iso) -> None:
        self.source_id = source_id
        self._mono, self._utc = monotonic_ns, utc_now
        self._sequence = 0
        self._last_mono = -1
        self._lock = threading.Lock()

    @property
    def last_sequence(self) -> int:
        with self._lock:
            return self._sequence

    def stamp(self, image: object, capture_latency_ms: float) -> CapturedFrame:
        img = check_image(image)
        metrics = quality_metrics(img)
        h, w = img.shape[:2]
        with self._lock:
            mono = max(self._mono(), self._last_mono + 1)
            self._last_mono = mono
            self._sequence += 1
            seq = self._sequence
        return CapturedFrame(
            frame_id=str(uuid.uuid4()), source_id=self.source_id, sequence=seq, monotonic_ns=mono,
            captured_at_utc=self._utc(), capture_latency_ms=max(0.0, float(capture_latency_ms)),
            input=InputDescriptor(w, h, ImageQuality.UNKNOWN), status=CaptureStatus.OK,
            quality_metrics=metrics, pixels=img,
        )
