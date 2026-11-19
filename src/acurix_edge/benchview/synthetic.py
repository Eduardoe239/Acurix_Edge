"""Synthetic bench scene for the bench viewer demo (`bench-view --demo-sintetico`, owner approval
19/11/2026). NEVER opens a real camera: the capture object is generated in code and the source
cannot be given another capture factory.

Scene (640x480, ~15 frames/s, 20 s cycle): grey background, light box, dark label that rotates:
  0-8 s   label almost aligned          -> shadow preview "alinhada"
  8-16 s  label goes 2° -> 8°           -> limit band 3-5°, then "inclinada"
  16-20 s box pushed out of the template -> quality gate refuses (INPUT_INVALID)
Mild noise keeps the image realistic for the quality gate (sharpness).
"""

from __future__ import annotations

import math
import time
from typing import Any

import cv2
import numpy as np

from acurix_edge.capture.sources import WebcamSource

SYNTHETIC_BACKEND = "SYNTH-DEMO"
SYNTHETIC_LABEL = "Fonte simulada (imagem gerada por código)"
SYNTHETIC_SOURCE_ID = "demo-sintetico"
WIDTH, HEIGHT = 640, 480
FRAME_PERIOD_S = 1 / 15
CYCLE_S = 20.0
SUBPIXEL_SHIFT = 4
SUBPIXEL = 1 << SUBPIXEL_SHIFT


def scene_angle_deg(t: float) -> tuple[float, int]:
    """(label angle in degrees, horizontal shift of the box in px) at time t of the cycle."""
    t %= CYCLE_S
    if t < 8:
        return 1.5 * math.sin(t / 8 * 2 * math.pi), 0
    if t < 16:
        return 2.0 + (t - 8) / 8 * 6.0, 0
    return 0.0, 200


def render_scene(t: float, rng: np.random.Generator) -> Any:
    angle, dx = scene_angle_deg(t)
    img = np.full((HEIGHT, WIDTH), 90, np.uint8)
    cx, cy = WIDTH // 2 + dx, HEIGHT // 2
    cv2.rectangle(img, (cx - 150, cy - 110), (cx + 150, cy + 110), 175, -1)  # box
    # Label: anti-aliased, sub-pixel corners (shift=4 -> 1/16 px), so its edges look like a camera's.
    box = np.round(cv2.boxPoints(((cx, cy), (180, 60), -angle)) * SUBPIXEL).astype(np.int32)
    cv2.fillPoly(img, [box], 70, lineType=cv2.LINE_AA, shift=SUBPIXEL_SHIFT)
    noisy = img.astype(np.int16) + rng.normal(0, 4, img.shape).astype(np.int16)
    return cv2.cvtColor(np.clip(noisy, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


class SceneCapture:
    """cv2.VideoCapture stand-in: always open, fixed 640x480, frames generated in code."""

    def __init__(self, clock: Any = time.monotonic, pace: bool = True) -> None:
        self._clock = clock
        self._t0 = clock()
        self._rng = np.random.default_rng(7)
        self._open = True
        self._pace = pace

    def isOpened(self) -> bool:  # noqa: N802 (cv2 API name)
        return self._open

    def getBackendName(self) -> str:  # noqa: N802 (cv2 API name)
        return SYNTHETIC_BACKEND

    def get(self, prop: int) -> float:
        return {cv2.CAP_PROP_FRAME_WIDTH: float(WIDTH), cv2.CAP_PROP_FRAME_HEIGHT: float(HEIGHT)}.get(prop, -1.0)

    def set(self, prop: int, value: float) -> bool:
        return False  # fixed resolution

    def release(self) -> None:
        self._open = False

    def read(self) -> tuple[bool, Any]:
        if not self._open:
            return False, None
        if self._pace:
            time.sleep(FRAME_PERIOD_S)
        return True, render_scene(self._clock() - self._t0, self._rng)


def _scene_factory(_target: object, *_args: object) -> SceneCapture:
    return SceneCapture()


class SyntheticSceneSource(WebcamSource):
    """Live source whose capture object is ALWAYS the generated scene (never cv2.VideoCapture)."""

    simulated = True

    def __init__(self) -> None:
        super().__init__(0, source_id=SYNTHETIC_SOURCE_ID, capture_factory=_scene_factory,
                         device_label=SYNTHETIC_LABEL)


def synthetic_profile_config() -> dict[str, Any]:
    """`[capture] camera_*` values that match SyntheticSceneSource (set in code, not by the user)."""
    res = f"{WIDTH}x{HEIGHT}"
    return {"camera_backend": SYNTHETIC_BACKEND, "camera_default_resolution": res,
            "camera_resolutions": (res,), "camera_device_label": SYNTHETIC_LABEL}
