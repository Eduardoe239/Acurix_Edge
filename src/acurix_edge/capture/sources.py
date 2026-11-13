"""Single frame-source contract with three implementations (CAP-001).

Contract (`FrameSource`):
- `open()` acquires the source or raises `SourceOpenError` (AEX-CAP-001). Never returns a
  half-open source.
- `read()` returns one image (numpy uint8, BGR) or `None` = END OF STREAM. End of stream is
  explicit and final: after `None`, every further `read()` returns `None` (no frame is repeated,
  no false error). A source that can no longer deliver raises `SourceLostError` (AEX-CAP-001).
- `read()` before `open()` / after `close()` raises `SourceError`.
- `close()` is idempotent.
- `live` tells whether frames keep coming whether or not they are consumed (webcam) or are
  produced on demand (folder, video file). It sets the buffer policy, not the contract.

The webcam never ends: a failed read is a loss, not an end of stream. The real camera is only
opened when `open()` is called on a `WebcamSource` — nothing here opens it implicitly.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import cv2
import numpy as np
import numpy.typing as npt

from acurix_edge.contracts.errors import ErrorCode

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
Image = npt.NDArray[np.uint8]  # 8-bit image as delivered by OpenCV (BGR, BGRA or gray)


class SourceError(RuntimeError):
    code = ErrorCode.CAP_001


class SourceOpenError(SourceError):
    """The source could not be opened (missing, busy, empty, unreadable)."""


class SourceLostError(SourceError):
    """An open source stopped delivering valid frames."""


@runtime_checkable
class FrameSource(Protocol):
    source_id: str
    live: bool

    @property
    def is_open(self) -> bool: ...

    def open(self) -> None: ...

    def read(self) -> Image | None: ...

    def close(self) -> None: ...


class ImageFolderSource:
    """Images of one folder (non-recursive), in file-name order. Decoded as 8-bit BGR."""

    live = False

    def __init__(self, folder: Path, *, source_id: str | None = None) -> None:
        self.folder = Path(folder)
        self.source_id = source_id or f"image-folder:{self.folder.name}"
        self._files: list[Path] | None = None
        self._index = 0
        self._ended = False

    @property
    def is_open(self) -> bool:
        return self._files is not None

    def open(self) -> None:
        try:
            if not self.folder.is_dir():
                raise SourceOpenError(f"{self.source_id}: folder not found")
            files = sorted((p for p in self.folder.iterdir()
                            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS), key=lambda p: p.name)
        except OSError as exc:  # permission, network share gone, ...
            raise SourceOpenError(f"{self.source_id}: folder unreadable ({type(exc).__name__})") from exc
        if not files:
            raise SourceOpenError(f"{self.source_id}: no image files")
        self._files, self._index, self._ended = files, 0, False

    def read(self) -> Image | None:
        if self._files is None:
            raise SourceError(f"{self.source_id}: read on a closed source")
        if self._ended or self._index >= len(self._files):
            self._ended = True
            return None
        path = self._files[self._index]
        self._index += 1
        try:
            data = np.fromfile(str(path), dtype=np.uint8)  # works with non-ASCII Windows paths
        except OSError as exc:
            raise SourceLostError(f"{self.source_id}: {path.name} unreadable ({type(exc).__name__})") from exc
        image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        if image is None:
            raise SourceLostError(f"{self.source_id}: {path.name} is not a decodable image")
        return cast(Image, image)

    def close(self) -> None:
        self._files = None


CaptureFactory = Callable[..., Any]  # returns an object with isOpened/read/release (cv2.VideoCapture-like)


def _release(cap: Any) -> None:
    try:
        cap.release()
    except Exception:  # release failure must not hide the original condition
        pass


class _VideoCaptureSource:
    """Shared logic for cv2.VideoCapture-backed sources.

    release() never runs concurrently with read() on the same capture object (a native
    use-after-free risk): close() detaches the object at once; if a read is in flight on it, the
    reading thread releases it when its read returns."""

    live = False

    def __init__(self, target: Any, source_id: str, factory: CaptureFactory) -> None:
        self._target = target
        self.source_id = source_id
        self._factory = factory
        self._cap: Any = None
        self._ended = False
        self._io_lock = threading.Lock()
        self._reading: dict[int, int] = {}   # id(cap) -> reads in flight
        self._deferred: dict[int, Any] = {}  # id(cap) -> cap to release after its read returns

    @property
    def is_open(self) -> bool:
        return self._cap is not None

    def _configure(self, cap: Any) -> None:
        """Hook for source-specific settings after a successful open."""

    def open(self) -> None:
        if self._cap is not None:
            return
        try:
            cap = self._factory(self._target)
        except Exception as exc:  # backend raised instead of reporting "not opened"
            raise SourceOpenError(f"{self.source_id}: open failed ({type(exc).__name__})") from exc
        try:
            opened = cap is not None and bool(cap.isOpened())
            if opened:
                self._configure(cap)
        except Exception as exc:
            _release(cap)
            raise SourceOpenError(f"{self.source_id}: open failed ({type(exc).__name__})") from exc
        if not opened:
            if cap is not None:
                _release(cap)
            raise SourceOpenError(f"{self.source_id}: could not be opened")
        self._cap, self._ended = cap, False

    def _grab(self) -> tuple[bool, Any]:
        with self._io_lock:
            cap = self._cap
            if cap is None:
                raise SourceError(f"{self.source_id}: read on a closed source")
            self._reading[id(cap)] = self._reading.get(id(cap), 0) + 1
        try:
            ok, image = cap.read()
        except Exception as exc:
            raise SourceLostError(f"{self.source_id}: read failed ({type(exc).__name__})") from exc
        finally:
            release_now = None
            with self._io_lock:
                left = self._reading[id(cap)] - 1
                if left:
                    self._reading[id(cap)] = left
                else:
                    del self._reading[id(cap)]
                    release_now = self._deferred.pop(id(cap), None)
            if release_now is not None:
                _release(release_now)
        if self._cap is not cap:
            raise SourceError(f"{self.source_id}: source closed during read")
        return bool(ok) and image is not None, image

    def read(self) -> Image | None:
        raise NotImplementedError

    def close(self) -> None:
        with self._io_lock:
            cap, self._cap = self._cap, None
            if cap is not None and id(cap) in self._reading:
                self._deferred[id(cap)] = cap  # released by the reader when its read returns
                cap = None
        if cap is not None:
            _release(cap)


class VideoFileSource(_VideoCaptureSource):
    """A video file. A failed read after the last frame is END OF STREAM (not an error).
    Limitation: a truncated file also ends early; codecs do not report truncation reliably."""

    def __init__(self, path: Path, *, source_id: str | None = None,
                 capture_factory: CaptureFactory | None = None) -> None:
        self.path = Path(path)
        super().__init__(str(self.path), source_id or f"video-file:{self.path.name}",
                         capture_factory or cv2.VideoCapture)

    def open(self) -> None:
        if self._cap is None and not self.path.is_file():
            raise SourceOpenError(f"{self.source_id}: file not found")
        super().open()

    def read(self) -> Image | None:
        if self._cap is None:
            raise SourceError(f"{self.source_id}: read on a closed source")
        if self._ended:
            return None
        ok, image = self._grab()
        if not ok:
            self._ended = True
            return None
        return cast(Image, image)


def _default_webcam_factory(index: int) -> Any:
    # DirectShow opens quickly on Windows; elsewhere let OpenCV choose.
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    return cv2.VideoCapture(index, backend)


class WebcamSource(_VideoCaptureSource):
    """USB/UVC webcam by device index. Live: a failed read is a LOSS (SourceLostError).
    Opening this source turns the real camera on: only with the owner's authorization."""

    live = True

    def __init__(self, device_index: int, *, width: int | None = None, height: int | None = None,
                 source_id: str | None = None, capture_factory: CaptureFactory | None = None) -> None:
        if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index < 0:
            raise ValueError("device_index must be int >= 0")
        self.device_index = device_index
        self._size = (width, height)
        super().__init__(device_index, source_id or f"webcam-usb-{device_index}",
                         capture_factory or _default_webcam_factory)

    def _configure(self, cap: Any) -> None:
        width, height = self._size
        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

    def read(self) -> Image | None:
        ok, image = self._grab()
        if not ok:
            raise SourceLostError(f"{self.source_id}: no frame from camera")
        return cast(Image, image)
