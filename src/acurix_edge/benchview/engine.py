"""Bench viewer engine: one real session, frames processed by the NORMAL flow
(`Application.process_frame`: quality gate -> shadow specialist -> decision -> event), at most
`fps` frames per second. Frames in between are taken from the capture buffer (so the buffer never
overflows) and discarded. The latest processed frame is kept ONLY in memory as a JPEG for the
MJPEG stream; nothing is ever written to disk by this module.

A camera loss stops the loop (FAULT latched by the application); the camera is never reopened
here (AEX-CAPTURE-007). `stop()` ends the session and releases the camera.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from acurix_edge.app import Application, SessionError
from acurix_edge.benchview.view_model import build_status
from acurix_edge.capture.errors import CaptureError
from acurix_edge.contracts.errors import ErrorCode
from acurix_edge.domain.stages import InspectionStage
from acurix_edge.health.service import host_memory

log = logging.getLogger("acurix_edge.benchview")

FPS_RANGE = (1.0, 15.0)
STREAM_WIDTH_RANGE = (160, 1280)
JPEG_QUALITY = 70
LATENCY_WINDOW = 50


def p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def encode_jpeg(image: Any, max_width: int) -> bytes:
    """In-memory JPEG of `image` downscaled to `max_width` (never written to disk)."""
    h, w = image.shape[:2]
    if w > max_width:
        image = cv2.resize(image, (max_width, max(1, round(h * max_width / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return bytes(np.asarray(buf).tobytes())


class BenchViewEngine:
    def __init__(self, app: Application, stage: InspectionStage | None, *, fps: float = 8.0,
                 stream_width: int = 640, clock: Callable[[], float] = time.monotonic) -> None:
        if not FPS_RANGE[0] <= fps <= FPS_RANGE[1]:
            raise ValueError(f"fps must be in {FPS_RANGE[0]:g}..{FPS_RANGE[1]:g}")
        if not STREAM_WIDTH_RANGE[0] <= stream_width <= STREAM_WIDTH_RANGE[1]:
            raise ValueError(f"stream width must be in {STREAM_WIDTH_RANGE[0]}..{STREAM_WIDTH_RANGE[1]}")
        if app.capture is None:
            raise ValueError("the bench viewer needs an application with a capture source")
        self.app, self.stage = app, stage
        self.interval = 1.0 / fps
        self.stream_width = stream_width
        self._clock = clock
        self._lock = threading.Lock()
        self._frame_ready = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._stop_done = threading.Event()
        self._thread: threading.Thread | None = None
        self._stopped = False
        # last processed frame (memory only)
        self._jpeg: bytes | None = None
        self._jpeg_seq = 0
        self._frame_size: tuple[int, int] | None = None
        self._result: str | None = None
        self._reason: str | None = None
        self._detail: dict[str, Any] | None = None
        self._latencies: deque[float] = deque(maxlen=LATENCY_WINDOW)
        self._processed_at: deque[float] = deque(maxlen=32)
        self.frames_processed = 0
        self.frames_skipped = 0
        self.problem: str | None = None
        self.problem_code: str | None = None

    # -- lifecycle --------------------------------------------------------------------------
    def start(self) -> bool:
        """Start the session (camera verified + profile checked). False if refused (shown on the page)."""
        try:
            self.app.start_session()
        except SessionError as exc:
            self._set_problem(str(exc), exc.code.value if exc.code else None)
            return False
        self._thread = threading.Thread(target=self._run, name="acurix-benchview", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        """Idempotent: end the loop, close the session, release the camera. A second caller
        (button + Ctrl+C) WAITS until the first one finished releasing (review B6)."""
        with self._lock:
            first = not self._stopped
            self._stopped = True
            self._frame_ready.notify_all()
        if not first:
            self._stop_done.wait(15.0)
            return
        self._stop.set()
        try:
            if self._thread is not None:
                self._thread.join(5.0)
            try:
                if self.app.session_id is not None:
                    self.app.close_session()
            except Exception:
                log.exception("bench viewer: close_session failed", extra={"code": "BENCHVIEW_STOP_ERROR"})
            finally:
                self.app.shutdown()  # releases the camera
        finally:
            self._stop_done.set()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def _set_problem(self, message: str, code: str | None) -> None:
        with self._lock:
            self.problem, self.problem_code = message, code

    # -- loop -------------------------------------------------------------------------------
    def _run(self) -> None:
        last = -math.inf
        try:
            while not self._stop.is_set():
                try:
                    frame = self.app.next_frame(0.5)
                except (SessionError, CaptureError) as exc:
                    code = getattr(getattr(exc, "code", None), "value", None)
                    self._set_problem(f"Captura interrompida: {exc}", code or ErrorCode.CAP_001.value)
                    return  # FAULT latched by the application; never reopened here
                if frame is None or frame.pixels is None:
                    continue
                now = self._clock()
                if now - last < self.interval:
                    self.frames_skipped += 1  # taken from the buffer, not processed (rate limit)
                    continue
                last = now
                started = time.perf_counter()
                try:
                    outcome = self.app.process_frame(frame, stage=self.stage)
                except SessionError as exc:
                    self._set_problem(f"Processamento interrompido: {exc}", exc.code.value if exc.code else None)
                    return
                latency_ms = (time.perf_counter() - started) * 1000.0
                jpeg = encode_jpeg(frame.pixels, self.stream_width)
                h, w = frame.pixels.shape[:2]
                with self._lock:
                    self._jpeg, self._jpeg_seq = jpeg, self._jpeg_seq + 1
                    self._frame_size = (int(w), int(h))
                    self._result, self._reason, self._detail = outcome.result, outcome.reason, outcome.detail
                    self._latencies.append(latency_ms)
                    self._processed_at.append(now)
                    self.frames_processed += 1
                    self._frame_ready.notify_all()
        except Exception as exc:  # never silent: shown on the page as a problem
            log.exception("bench viewer loop failed", extra={"code": "BENCHVIEW_LOOP_ERROR"})
            self._set_problem(f"Erro no visualizador ({type(exc).__name__})", ErrorCode.SYS_001.value)
        finally:
            with self._lock:
                self._frame_ready.notify_all()

    # -- outputs ----------------------------------------------------------------------------
    def wait_frame(self, after_seq: int, timeout: float) -> tuple[int, bytes] | None:
        """Next JPEG newer than `after_seq` (memory only), or None on timeout / stop."""
        with self._frame_ready:
            if self._jpeg_seq <= after_seq and not self._stopped:
                self._frame_ready.wait(timeout)
            if self._jpeg is None or self._jpeg_seq <= after_seq:
                return None
            return self._jpeg_seq, self._jpeg

    def snapshot(self) -> dict[str, Any]:
        capture = self.app.capture
        assert capture is not None
        health = capture.health()
        with self._lock:
            latencies = list(self._latencies)
            times = list(self._processed_at)
            fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) >= 2 and times[-1] > times[0] else None
            if times and self._clock() - times[-1] > 2.0:
                fps = 0.0  # nothing processed recently (stopped, refused or lost)
            mem = host_memory()
            return build_status(
                system_state=self.app.state.state.value, camera_status=health[0].value,
                latency_p95_ms=p95(latencies), fps=fps,
                ram_free_gb=mem["available_gb"] if mem else None, quality_config=self.app.config.quality,
                image_stale=self._jpeg is not None and not self.running,
                result=self._result, reason=self._reason, detail=self._detail, frame_size=self._frame_size,
                problem=self.problem, problem_code=self.problem_code, frames_processed=self.frames_processed,
                running=self.running)
