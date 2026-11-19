"""Synthetic capture fixtures (NFR-010): every image is GENERATED here from shapes and colors.
No real photo, no person, no real camera. Sizes are tiny (64x48) to respect ADR-014.

`SyntheticCamera` is a cv2.VideoCapture stand-in for `WebcamSource(capture_factory=...)`: it
produces frames on demand and can be told to fail to open, disconnect (read -> False), stall
(read blocks) or come back (reconnect). It also emulates a camera PROFILE (backend name,
default size, sizes it delivers), so a "different camera" is just different parameters."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from acurix_edge.capture.profile import PROFILE_CANDIDATES, CameraProfile, resolution, sort_resolutions

W, H = 64, 48
SYNTH_BACKEND = "SYNTH"
SYNTH_SIZES: tuple[tuple[int, int], ...] = ((W, H), (96, 72))  # tiny on purpose (ADR-014)


def synth_image(i: int, w: int = W, h: int = H) -> npt.NDArray[np.uint8]:
    """Deterministic BGR image: background shade + a rectangle and a circle placed by `i`."""
    img = np.full((h, w, 3), (40 + 7 * i) % 200, dtype=np.uint8)
    x = 4 + (i * 5) % max(1, w - 24)
    cv2.rectangle(img, (x, 6), (x + 16, 22), (0, 0, 255), -1)
    cv2.circle(img, (w - 12, h - 12), 6 + i % 4, (255, (60 * i) % 256, 0), -1)
    return img


def write_image_folder(folder: Path, n: int, **kw: int) -> list[npt.NDArray[np.uint8]]:
    folder.mkdir(parents=True, exist_ok=True)
    images = [synth_image(i, **kw) for i in range(n)]
    for i, img in enumerate(images):
        ok, buf = cv2.imencode(".png", img)
        assert ok
        (folder / f"img_{i:03d}.png").write_bytes(buf.tobytes())
    return images


def write_video(path: Path, n: int, fps: float = 10.0) -> None:
    """MJPG/AVI is encoded by OpenCV itself (no external codec)."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), fps, (W, H))
    assert writer.isOpened(), "OpenCV MJPG writer unavailable"
    for i in range(n):
        writer.write(synth_image(i))
    writer.release()


class SyntheticCamera:
    """Controls shared by every capture object the factory creates (survives reopen)."""

    def __init__(self, *, frames: int | None = None, open_ok: bool = True, backend: str = SYNTH_BACKEND,
                 sizes: tuple[tuple[int, int], ...] = SYNTH_SIZES) -> None:
        self.frames = frames  # None = endless (a live camera); int = disconnect after N reads
        self.open_ok = open_ok
        self.backend = backend
        self.sizes = tuple(sorted(sizes, key=lambda s: s[0] * s[1]))  # first = default size
        self.connected = True
        self.reads = 0
        self.opens = 0
        self._stall = threading.Event()   # set -> read blocks
        self._unstall = threading.Event()
        self.set_gate: threading.Event | None = None  # not None and not set -> cap.set blocks (driver stuck)
        self.open_gate: threading.Event | None = None  # not None and not set -> opening blocks (driver stuck)
        self.caps: list[_Cap] = []  # every capture object created, in order
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

    def factory(self, _target: object, *_args: object) -> _Cap:
        gate = self.open_gate
        if gate is not None:
            gate.wait(10.0)
        cap = _Cap(self)
        with self.lock:
            self.opens += 1
            self.caps.append(cap)
        return cap

    def delivered(self, w: int, h: int) -> tuple[int, int]:
        """Like a real driver: the smallest size that covers the request, else the largest."""
        return next(((sw, sh) for sw, sh in self.sizes if sw >= w and sh >= h), self.sizes[-1])

    def profile(self, device_label: str = "webcam-usb-0") -> CameraProfile:
        """The profile WebcamSource will measure on this camera (computed, camera not opened)."""
        return CameraProfile(self.backend, resolution(*self.sizes[0]),
                             sort_resolutions(resolution(*self.delivered(w, h)) for w, h in PROFILE_CANDIDATES),
                             device_label)


def profile_config(profile: CameraProfile) -> dict[str, Any]:
    """CaptureConfig keyword arguments that store `profile` as the expected one."""
    return {"camera_backend": profile.backend, "camera_default_resolution": profile.default_resolution,
            "camera_resolutions": profile.supported_resolutions, "camera_device_label": profile.device_label}


class _Cap:
    def __init__(self, cam: SyntheticCamera) -> None:
        self.cam = cam
        with cam.lock:
            self._open = cam.open_ok and cam.connected
        self._size = cam.sizes[0]
        self._req = list(self._size)  # requested width/height (set one at a time, like cv2)
        self.set_calls = 0

    def isOpened(self) -> bool:  # noqa: N802 (cv2 API name)
        return self._open

    def getBackendName(self) -> str:  # noqa: N802 (cv2 API name)
        return self.cam.backend

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._size[0])
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._size[1])
        return -1.0

    def set(self, prop: int, value: float) -> bool:
        gate = self.cam.set_gate
        if gate is not None:
            gate.wait(5.0)
        self.set_calls += 1
        if prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT):
            self._req[0 if prop == cv2.CAP_PROP_FRAME_WIDTH else 1] = int(value)
            self._size = self.cam.delivered(self._req[0], self._req[1])
        return True

    def read(self) -> tuple[bool, npt.NDArray[np.uint8] | None]:
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
        return True, synth_image(i, *self._size)

    def release(self) -> None:
        self._open = False
