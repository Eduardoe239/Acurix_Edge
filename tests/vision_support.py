"""Synthetic packaging scenes (NFR-010): every image is DRAWN here — uniform background, a box
(product) and a label rectangle, with controlled rotation, blur, noise, exposure and reflection.
No real photo. Default size 320x240 (ADR-014: RAM below 1.5 GB)."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import cv2
import numpy as np
import numpy.typing as npt

Img = npt.NDArray[np.uint8]
W, H = 320, 240
SHIFT = 4  # sub-pixel drawing (coordinates x 16)


@dataclass(frozen=True)
class Scene:
    product_angle: float = 0.0           # counter-clockwise as displayed (degrees)
    skew: float = 0.0                    # label angle relative to the product (CCW, degrees)
    center: tuple[float, float] = (W / 2, H / 2)
    product_size: tuple[float, float] = (180.0, 120.0)
    label_size: tuple[float, float] = (110.0, 44.0)
    label: bool = True
    background: int = 90
    product_bgr: tuple[int, int, int] = (150, 165, 180)
    label_bgr: tuple[int, int, int] = (235, 232, 228)
    blur_sigma: float = 0.0
    noise_sigma: float = 2.0
    gain: float = 1.0                    # exposure multiplier
    reflection_radius: float = 0.0       # white specular blob on the label
    clutter: bool = False                # dark object lying on the frame border (wrong framing)
    product_shape: str = "rect"          # "rect" or "ellipse" (round product: no reference axis)
    seed: int = 0

    def with_(self, **kw: object) -> Scene:
        return replace(self, **kw)


def _rect(center: tuple[float, float], size: tuple[float, float], angle_deg: float) -> npt.NDArray[np.int32]:
    cx, cy = center
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    pts = []
    for dx, dy in ((-size[0] / 2, -size[1] / 2), (size[0] / 2, -size[1] / 2),
                   (size[0] / 2, size[1] / 2), (-size[0] / 2, size[1] / 2)):
        # CCW as displayed (image y axis points down)
        pts.append((cx + dx * c + dy * s, cy - dx * s + dy * c))
    return np.round(np.array(pts) * (1 << SHIFT)).astype(np.int32)


SUPER = 4  # supersampling factor: geometrically exact pixel coverage after INTER_AREA reduction


def render(scene: Scene) -> Img:
    """Draw at SUPER x resolution without anti-aliasing, then reduce by area averaging: every edge
    pixel gets its true coverage (cv2 LINE_AA fills carry a sub-pixel bias of their own)."""
    s = SUPER
    big = np.full((H * s, W * s, 3), scene.background, np.uint8)
    cx, cy = scene.center[0] * s, scene.center[1] * s

    def poly(size: tuple[float, float], angle: float) -> npt.NDArray[np.int32]:
        return _rect((cx - 0.5 * (s - 1), cy - 0.5 * (s - 1)), (size[0] * s, size[1] * s), angle)

    if scene.product_shape == "ellipse":
        cv2.ellipse(big, (int(cx), int(cy)), (int(scene.product_size[0] * s / 2), int(scene.product_size[1] * s / 2)),
                    -scene.product_angle, 0, 360, scene.product_bgr, -1)
    else:
        cv2.fillPoly(big, [poly(scene.product_size, scene.product_angle)], scene.product_bgr, shift=SHIFT)
    if scene.clutter:
        cv2.rectangle(big, (20 * s, 0), (200 * s, 14 * s), (30, 30, 30), -1)
    if scene.label:
        cv2.fillPoly(big, [poly(scene.label_size, scene.product_angle + scene.skew)], scene.label_bgr, shift=SHIFT)
    if scene.reflection_radius > 0:
        cv2.circle(big, (int(cx), int(cy)), int(scene.reflection_radius * s), (255, 255, 255), -1)
    img = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)
    del big
    out = img.astype(np.float32) * scene.gain
    if scene.noise_sigma > 0:
        out += np.random.default_rng(abs(scene.seed)).normal(0.0, scene.noise_sigma, out.shape).astype(np.float32)
    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    if scene.blur_sigma > 0:
        out_u8 = np.asarray(cv2.GaussianBlur(out_u8, (0, 0), scene.blur_sigma), dtype=np.uint8)
    return out_u8


GOOD = Scene()
