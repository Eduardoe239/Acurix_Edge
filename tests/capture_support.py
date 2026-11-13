"""Synthetic capture fixtures (NFR-010): every image is GENERATED here from shapes and colors.
No real photo, no person, no real camera. Sizes are tiny (64x48) to respect ADR-014.

`SyntheticCamera` is a cv2.VideoCapture stand-in for `WebcamSource(capture_factory=...)`: it
produces frames on demand and can be told to fail to open, disconnect (read -> False), stall
(read blocks) or come back (reconnect)."""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np

W, H = 64, 48


def synth_image(i: int, w: int = W, h: int = H) -> np.ndarray:
    """Deterministic BGR image: background shade + a rectangle and a circle placed by `i`."""
    img = np.full((h, w, 3), (40 + 7 * i) % 200, dtype=np.uint8)
    x = 4 + (i * 5) % max(1, w - 24)
    cv2.rectangle(img, (x, 6), (x + 16, 22), (0, 0, 255), -1)
    cv2.circle(img, (w - 12, h - 12), 6 + i % 4, (255, (60 * i) % 256, 0), -1)
    return img


def write_image_folder(folder: Path, n: int, **kw: int) -> list[np.ndarray]:
    folder.mkdir(parents=True, exist_ok=True)
    images = [synth_image(i, **kw) for i in range(n)]
    for i, img in enumerate(images):
        ok, buf = cv2.imencode(".png", img)
        assert ok
        (folder / f"img_{i:03d}.png").write_bytes(buf.tobytes())
    return images


def write_video(path: Path, n: int, fps: float = 10.0) -> None:
    """MJPG/AVI is encoded by OpenCV itself (no external codec)."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))
    assert writer.isOpened(), "OpenCV MJPG writer unavailable"
    for i in range(n):
        writer.write(synth_image(i))
    writer.release()


class SyntheticCamera:
    """Controls shared by every capture object the factory creates (survives reopen)."""

    def __init__(self, *, frames: int | None = None, open_ok: bool = True) -> None:
        self.frames = frames  # None = endless (a live camera); int = disconnect after N reads
        self.open_ok = open_ok
        self.connected = True
        self.reads = 0
        self.opens = 0
        self._stall = threading.Event()   # set -> read blocks
        self._unstall = threading.Event()
        self.lock = threading.Lock()

    # test controls -------------------------------------------------------------
    def disconnect(self) -> None:
        with self.lock:
            self.connected = False

    def reconnect(self) -> None:
        with self.lock:
            self.connected, self.open_ok, self.frames = True, True, None
        self.unstall()

    def stall(self) -> None:
        self._unstall.clear()
        self._stall.set()

    def unstall(self) -> None:
        self._stall.clear()
        self._unstall.set()

    def factory(self, _target: object, *_args: object) -> "_Cap":
        with self.lock:
            self.opens += 1
        return _Cap(self)


class _Cap:
    def __init__(self, cam: SyntheticCamera) -> None:
        self.cam = cam
        with cam.lock:
            self._open = cam.open_ok and cam.connected

    def isOpened(self) -> bool:  # noqa: N802 (cv2 API name)
        return self._open

    def set(self, _prop: int, _value: float) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        cam = self.cam
        if cam._stall.is_set():
            cam._unstall.wait(5.0)
        with cam.lock:
            if not self._open or not cam.connected:
                return False, None
            if cam.frames is not None and cam.reads >= cam.frames:
                return False, None
            i = cam.reads
            cam.reads += 1
        return True, synth_image(i)

    def release(self) -> None:
        self._open = False
