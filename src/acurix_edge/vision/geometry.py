"""Shared geometry for the quality gate and the label-skew specialist (deterministic, OpenCV).

Angles are COUNTER-CLOCKWISE AS DISPLAYED (the image y axis points down). Axes are defined modulo
90° for rectangles (a rectangle has two perpendicular symmetry axes), so differences and rotations
are reported in [-45°, 45°)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from acurix_edge.capture.sources import Image


@dataclass(frozen=True, slots=True)
class Axis:
    angle_deg: float      # counter-clockwise as displayed, in (-90, 90]
    elongation: float     # sqrt(major/minor) second-moment ratio (1 = isotropic)


def soft_weights(gray: Image, hard: Any, level: float, contrast: float, pad: int = 2) -> Any:
    """Sub-pixel membership: inside the hard mask dilated by `pad` px, weight = |gray - level| /
    contrast clipped to [0, 1] (anti-aliased edge pixels count partially). Outside: 0."""
    region = cv2.dilate(hard, np.ones((2 * pad + 1, 2 * pad + 1), np.uint8)) > 0
    diff = np.abs(gray.astype(np.float32) - np.float32(level))
    w = np.clip(diff / np.float32(max(contrast, 1.0)), 0.0, 1.0)
    return np.where(region, w, np.float32(0.0)).astype(np.float32)


def moment_axis(mask: Any) -> Axis | None:
    """Principal axis from image moments; `mask` may be binary (uint8) or soft weights (float32)."""
    m = cv2.moments(mask, binaryImage=False) if mask.dtype == np.float32 else cv2.moments(mask, binaryImage=True)
    if m["m00"] <= 0:
        return None
    mu20, mu02, mu11 = m["mu20"] / m["m00"], m["mu02"] / m["m00"], m["mu11"] / m["m00"]
    theta = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)  # image coords (y down): clockwise as displayed
    common = math.sqrt(((mu20 - mu02) / 2.0) ** 2 + mu11 ** 2)
    lam1, lam2 = (mu20 + mu02) / 2.0 + common, (mu20 + mu02) / 2.0 - common
    elongation = math.sqrt(lam1 / lam2) if lam2 > 1e-12 else float("inf")
    return Axis(-math.degrees(theta), elongation)


def rect_axis(contour: Any) -> float:
    """Edge direction of the minimum-area rectangle (used for near-square shapes)."""
    pts = cv2.boxPoints(cv2.minAreaRect(contour))
    (x0, y0), (x1, y1) = pts[0], pts[1]
    return math.degrees(math.atan2(-(float(y1) - float(y0)), float(x1) - float(x0)))


def rectangularity(contour: Any) -> float:
    """Contour area / minimum-area-rectangle area (1 = perfect rectangle; a disc is ~0.785)."""
    (_, _), (w, h), _ = cv2.minAreaRect(contour)
    return min(1.0, float(cv2.contourArea(contour)) / max(1.0, float(w) * float(h)))


def wrap90(angle: float) -> float:
    """Axis angle/difference modulo 90°, in [-45, 45)."""
    return ((angle + 45.0) % 90.0) - 45.0


def object_axis_deg(mask: Any, contour: Any, square_elongation: float) -> float:
    """Orientation of a (rectangular) object: moments when elongated, else min-area rectangle."""
    axis = moment_axis(mask)
    if axis is not None and axis.elongation >= square_elongation:
        return axis.angle_deg
    return rect_axis(contour)
