"""Image quality gate (QG-001, QG-002): runs on EVERY frame before any cascade stage.

Deterministic, OpenCV only. Checks (each failure is a reason with metric, value and limit):
- focus:      variance of the Laplacian of the gray ROI          < min_sharpness        -> FOCUS_LOW
- exposure:   mean gray of the frame                             < min_brightness       -> UNDEREXPOSED
                                                                 > max_brightness       -> OVEREXPOSED
              fraction of pixels <= dark_level                   > max_dark_fraction    -> CRUSHED_SHADOWS
              fraction of pixels >= bright_level                 > max_bright_fraction  -> BLOWN_HIGHLIGHTS
- reflection: fraction of ROI pixels with ALL channels >= specular_level
                                                                 > max_specular_fraction -> REFLECTION
- position (method, chosen for being simple and explainable): the template has a uniform
  background; its level is the MEDIAN of the frame border pixels. Object pixels differ from it by
  more than object_delta; the object is the LARGEST contour of that mask (filled).
              |border median - background_level|                 > background_tolerance -> BACKGROUND_OUT_OF_RANGE
              border pixels off the background by > object_delta > max_border_outlier_fraction -> BACKGROUND_NOT_UNIFORM
              (both catch wrong framing: camera too close, object or clutter on the border)
              no object, or object area / ROI area               < min_object_fraction  -> OBJECT_NOT_FOUND
              object area / ROI area                             > max_object_fraction  -> OBJECT_TOO_LARGE
              object bounding box not entirely inside the ROI    (pixels outside > 0)   -> OBJECT_OUTSIDE_ROI
              |object axis vs image axes| (mod 90°)              > max_object_rotation_deg -> OBJECT_ROTATED
              Limit: a round object has no axis; its rotation reads the min-area rectangle (provisional).

Any reason -> INPUT_INVALID. The result also carries every measured metric and the envelope
version; `envelope_validated` is False unless the envelope status is "validated" (impossible in
0.3.0 by configuration schema).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from acurix_edge.capture.frames import check_image
from acurix_edge.capture.sources import Image
from acurix_edge.config.settings import QualityConfig, envelope_fingerprint
from acurix_edge.contracts.quality import Comparison, QualityReason, QualityResult, QualityStatus
from acurix_edge.vision.geometry import object_axis_deg, rectangularity, wrap90

Mask = npt.NDArray[np.uint8]
BORDER_PX = 2
DEFAULT_INSET = 3  # same as QualityConfig.border_inset_px default
MIN_RECTANGULARITY = 0.85  # below: no reliable object axis (shared with the label-skew specialist)
SQUARE_ELONGATION = 1.15  # below: near-square object, its axis comes from the min-area rectangle


@dataclass(frozen=True, slots=True)
class ObjectRegion:
    """The dominant object of the frame (filled mask of its largest contour)."""

    mask: Mask                         # 255 inside the object, full-frame size
    contour: npt.NDArray[np.int32]
    bbox: tuple[int, int, int, int]    # x, y, w, h (pixels)
    area: float
    background_level: float


def to_gray(image: Image) -> Image:
    img = check_image(image)
    if img.ndim == 2:
        return img
    if img.shape[2] == 1:
        return img[:, :, 0]
    code = cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY
    return np.asarray(cv2.cvtColor(img, code), dtype=np.uint8)


def roi_pixels(q: QualityConfig, width: int, height: int) -> tuple[int, int, int, int]:
    """ROI in pixels, always inside the frame and at least 1x1 (review L4)."""
    x = min(int(round(q.roi_x * width)), width - 1)
    y = min(int(round(q.roi_y * height)), height - 1)
    w, h = int(round(q.roi_w * width)), int(round(q.roi_h * height))
    return x, y, max(1, min(w, width - x)), max(1, min(h, height - y))


def _border(gray: Image, inset: int = DEFAULT_INSET) -> Image:
    """Border band BORDER_PX wide, sampled `inset` px inside the frame edge (a dead or colored edge
    row/column of a cheap webcam is skipped). The inset is clamped so the band stays in the frame."""
    h, w = gray.shape[:2]
    i = max(0, min(inset, h // 2 - BORDER_PX, w // 2 - BORDER_PX))
    b = BORDER_PX
    return np.concatenate([gray[i:i + b, i:w - i].ravel(), gray[h - i - b:h - i, i:w - i].ravel(),
                           gray[i:h - i, i:i + b].ravel(), gray[i:h - i, w - i - b:w - i].ravel()])


def border_stats(gray: Image, delta: float, inset: int = DEFAULT_INSET) -> tuple[float, float]:
    """(median, outlier fraction) of the frame border: the template background level, and the
    fraction of border pixels that differ from it by more than `delta` (object/clutter on the border)."""
    border = _border(gray, inset)
    median = float(np.median(border))
    outliers = float(np.count_nonzero(np.abs(border.astype(np.int16) - int(round(median))) > delta))
    return median, outliers / float(border.size)


def find_object(gray: Image, delta: float, inset: int = DEFAULT_INSET) -> ObjectRegion | None:
    """Largest region that differs from the border (background) level by more than `delta`."""
    background = float(np.median(_border(gray, inset)))
    diff = np.abs(gray.astype(np.int16) - int(round(background))).astype(np.uint8)
    mask = np.where(diff > delta, 255, 0).astype(np.uint8)
    mask = np.asarray(cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), dtype=np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(gray)
    cv2.drawContours(filled, [biggest], -1, 255, thickness=cv2.FILLED)
    x, y, w, h = cv2.boundingRect(biggest)
    return ObjectRegion(filled, np.asarray(biggest, dtype=np.int32), (x, y, w, h),
                        float(cv2.countNonZero(filled)), background)


class QualityGate:
    def __init__(self, envelope: QualityConfig) -> None:
        self.envelope = envelope
        self.envelope_sha256 = envelope_fingerprint(envelope)[1]

    @property
    def validated(self) -> bool:
        return self.envelope.status == "validated"

    def evaluate(self, image: Image) -> QualityResult:
        started = time.perf_counter()
        q = self.envelope
        img = check_image(image)
        gray = to_gray(img)
        height, width = gray.shape[:2]
        rx, ry, rw, rh = roi_pixels(q, width, height)
        roi_gray = gray[ry:ry + rh, rx:rx + rw]
        total = float(gray.size)

        sharpness = float(cv2.Laplacian(roi_gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        dark = float(np.count_nonzero(gray <= q.dark_level)) / total
        bright = float(np.count_nonzero(gray >= q.bright_level)) / total
        roi_color = img[ry:ry + rh, rx:rx + rw]
        if roi_color.ndim == 3 and roi_color.shape[2] >= 3:
            specular_px = np.all(roi_color[:, :, :3] >= q.specular_level, axis=2)
        else:
            specular_px = roi_gray >= q.specular_level
        specular = float(np.count_nonzero(specular_px)) / float(roi_gray.size)

        bg_level, border_outliers = border_stats(gray, q.object_delta, q.border_inset_px)
        obj = find_object(gray, q.object_delta, q.border_inset_px)
        roi_area = float(rw * rh)
        obj_fraction = (obj.area / roi_area) if obj else 0.0
        outside_px = 0.0
        rotation = 0.0
        rect = 0.0
        rotation_measured = False
        if obj is not None:
            x, y, w, h = obj.bbox
            outside_px = float(max(0, rx - x, ry - y, (x + w) - (rx + rw), (y + h) - (ry + rh)))
            rect = rectangularity(obj.contour)
            # A poorly rectangular object (round, irregular) has no reliable axis: its rotation is NOT
            # evaluated here (never INPUT_INVALID for that); the specialist reports it as inconclusive.
            if rect >= MIN_RECTANGULARITY:
                rotation = wrap90(object_axis_deg(obj.mask, obj.contour, SQUARE_ELONGATION))
                rotation_measured = True

        metrics = {"sharpness_laplacian_var": sharpness, "brightness_mean": brightness,
                   "dark_fraction": dark, "bright_fraction": bright, "specular_fraction_roi": specular,
                   "background_level": bg_level, "background_offset": abs(bg_level - q.background_level),
                   "border_outlier_fraction": border_outliers, "object_area_fraction_roi": obj_fraction,
                   "object_outside_roi_px": outside_px, "object_rotation_deg": rotation,
                   "object_rotation_abs_deg": abs(rotation), "object_rectangularity": rect,
                   "object_rotation_measured": 1.0 if rotation_measured else 0.0}

        reasons: list[QualityReason] = []

        def fail(code: str, metric: str, cmp: Comparison, limit: float) -> None:
            reasons.append(QualityReason(code, metric, metrics[metric], float(limit), cmp))

        if sharpness < q.min_sharpness:
            fail("FOCUS_LOW", "sharpness_laplacian_var", Comparison.BELOW_MIN, q.min_sharpness)
        if brightness < q.min_brightness:
            fail("UNDEREXPOSED", "brightness_mean", Comparison.BELOW_MIN, q.min_brightness)
        if brightness > q.max_brightness:
            fail("OVEREXPOSED", "brightness_mean", Comparison.ABOVE_MAX, q.max_brightness)
        if dark > q.max_dark_fraction:
            fail("CRUSHED_SHADOWS", "dark_fraction", Comparison.ABOVE_MAX, q.max_dark_fraction)
        if bright > q.max_bright_fraction:
            fail("BLOWN_HIGHLIGHTS", "bright_fraction", Comparison.ABOVE_MAX, q.max_bright_fraction)
        if specular > q.max_specular_fraction:
            fail("REFLECTION", "specular_fraction_roi", Comparison.ABOVE_MAX, q.max_specular_fraction)
        # Template framing (review M2): the border must be the expected, uniform background.
        if metrics["background_offset"] > q.background_tolerance:
            fail("BACKGROUND_OUT_OF_RANGE", "background_offset", Comparison.ABOVE_MAX, q.background_tolerance)
        if border_outliers > q.max_border_outlier_fraction:
            fail("BACKGROUND_NOT_UNIFORM", "border_outlier_fraction", Comparison.ABOVE_MAX,
                 q.max_border_outlier_fraction)
        if obj is None or obj_fraction < q.min_object_fraction:  # explicit when nothing was found (L3)
            fail("OBJECT_NOT_FOUND", "object_area_fraction_roi", Comparison.NOT_FOUND, q.min_object_fraction)
        else:
            if obj_fraction > q.max_object_fraction:
                fail("OBJECT_TOO_LARGE", "object_area_fraction_roi", Comparison.ABOVE_MAX, q.max_object_fraction)
            if outside_px > 0:
                fail("OBJECT_OUTSIDE_ROI", "object_outside_roi_px", Comparison.ABOVE_MAX, 0.0)
            if rotation_measured and abs(rotation) > q.max_object_rotation_deg:  # review M3 (Anexo 04A §3.3)
                fail("OBJECT_ROTATED", "object_rotation_abs_deg", Comparison.ABOVE_MAX, q.max_object_rotation_deg)

        return QualityResult(
            status=QualityStatus.INPUT_INVALID if reasons else QualityStatus.VALID,
            reasons=tuple(reasons), metrics=metrics, elapsed_ms=(time.perf_counter() - started) * 1000.0,
            envelope_version=q.version, envelope_status=q.status, envelope_validated=self.validated,
            roi_px=(rx, ry, rw, rh), envelope_sha256=self.envelope_sha256)
