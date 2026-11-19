"""Capture service: verification, producer thread and loss watchdog
(CAP-003, CAP-004, HLT-002, HLT-004).

Lifecycle (`CaptureState`):
    UNVERIFIED --verify()--> VERIFIED --begin(session)--> RUNNING --end()--> STOPPED
    RUNNING --end of finite source--> ENDED (explicit, not an error)
    RUNNING --read error / invalid frame / read blocked > frame_timeout_s--> LOST
    LOST: final. The source is closed and NEVER reopened by itself (owner decision 19/11/2026:
          a reopen by device index may silently pick another physical camera). Only a new
          verify() — reached through reset-fault + a new session — opens it again.

Guarantees:
- the camera is UNKNOWN for health until verify() opened it, matched the expected camera
  profile (capture/profile.py) AND received a valid frame;
- on loss, `on_lost(detail, session_id)` is called once per session; the session binding is
  dropped at that instant, so frames are never delivered to the old session again;
- every frame that entered the buffer is either delivered or counted as dropped with a cause.
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from acurix_edge.capture.buffer import SESSION_ENDED, BufferClosed, BufferCounters, FrameBuffer
from acurix_edge.capture.errors import CaptureError, CaptureLost, CaptureUnavailable, EndOfStream, StaleSession
from acurix_edge.capture.frames import FrameStamper, InvalidImageError, check_image
from acurix_edge.capture.profile import CameraProfile, check_camera, expected_profile
from acurix_edge.capture.sources import FrameSource, Image, SourceError, SourceLostError, SourceOpenError
from acurix_edge.config.settings import CaptureConfig
from acurix_edge.contracts.capture import CapturedFrame
from acurix_edge.contracts.errors import ErrorCode
from acurix_edge.health.service import CheckResult, ComponentStatus

log = logging.getLogger("acurix_edge.capture")

END_OF_STREAM = "END_OF_STREAM"
SOURCE_LOST = "SOURCE_LOST"


class CaptureState(StrEnum):
    UNVERIFIED = "UNVERIFIED"
    VERIFIED = "VERIFIED"
    RUNNING = "RUNNING"
    ENDED = "ENDED"
    LOST = "LOST"
    STOPPED = "STOPPED"


LostHandler = Callable[[str, str | None], None]
AlarmHandler = Callable[[BufferCounters, str | None], None]


class CaptureService:
    def __init__(self, source: FrameSource, config: CaptureConfig, *, on_lost: LostHandler | None = None,
                 on_drop_alarm: AlarmHandler | None = None, clock: Callable[[], float] = time.monotonic,
                 monotonic_ns: Callable[[], int] = time.monotonic_ns) -> None:
        self.source = source
        self.config = config
        self.on_lost = on_lost
        self.on_drop_alarm = on_drop_alarm
        self._clock, self._mono_ns = clock, monotonic_ns
        self._lock = threading.RLock()
        self._state = CaptureState.UNVERIFIED
        self._detail = "not verified (camera not opened yet)"
        self._gen = 0
        self._session_id: str | None = None
        self._retired: dict[str, str] = {}  # session_id -> reason it was unbound
        self._buffer: FrameBuffer | None = None
        self._stop = threading.Event()
        self._shutdown = threading.Event()
        self._reading_since: float | None = None
        self._pending: Image | None = None
        self._threads: list[threading.Thread] = []
        self.lost_at: float | None = None  # clock() when the loss was declared (HLT-004 measurement)
        self.expected_profile: CameraProfile | None = expected_profile(config)
        self.profile: CameraProfile | None = None  # profile measured at the last successful verify()

    # -- introspection ------------------------------------------------------------
    @property
    def state(self) -> CaptureState:
        with self._lock:
            return self._state

    @property
    def detail(self) -> str:
        with self._lock:
            return self._detail

    @property
    def session_id(self) -> str | None:
        with self._lock:
            return self._session_id

    def counters(self, *, active_only: bool = False) -> BufferCounters:
        """Counters of the current/last session buffer. `active_only`: zeros unless a session is
        bound (health must not carry a previous session's drops into a new session)."""
        with self._lock:
            buffer = self._buffer
            if active_only and self._session_id is None:
                buffer = None
        if buffer is None:
            return BufferCounters(0, 0, {}, 0, 0, self.config.queue_max_frames, self.config.drop_alarm_threshold)
        return buffer.counters()

    def health(self) -> CheckResult:
        with self._lock:
            state, detail = self._state, self._detail
        sid = f"{self.source.source_id}: "
        if state in (CaptureState.UNVERIFIED, CaptureState.STOPPED):
            return ComponentStatus.UNKNOWN, sid + detail, ErrorCode.CAP_001
        if state is CaptureState.LOST:
            return ComponentStatus.FAULT, sid + detail, ErrorCode.CAP_001
        return ComponentStatus.OK, sid + detail

    # -- source helpers -------------------------------------------------------------
    def _close_source(self) -> None:
        try:
            self.source.close()
        except Exception:
            log.exception("capture source close failed", extra={"code": "CAPTURE_CLOSE_ERROR"})

    def _call_with_timeout(self, fn: Callable[[], Any], timeout: float, what: str) -> Any:
        """fn() bounded in time (a webcam read can block indefinitely)."""
        box: dict[str, Any] = {}

        def run() -> None:
            try:
                box["value"] = fn()
            except BaseException as exc:  # re-raised in the caller thread
                box["error"] = exc

        t = threading.Thread(target=run, name="acurix-capture-probe", daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            raise SourceLostError(f"{self.source.source_id}: no {what} within {timeout:g} s")
        if "error" in box:
            raise box["error"]
        return box.get("value")

    def _probe_source(self) -> tuple[Image, CameraProfile | None]:
        """Open the source, check the camera profile, get one valid frame.
        Raises SourceError (incl. CameraMismatch) or InvalidImageError."""
        # Always a FRESH open: finite sources are rewound (verification never consumes a frame) and a
        # live camera never reuses a handle left over from a previous session (review 2, B-1).
        self._close_source()
        self.source.open()
        profile: CameraProfile | None = self._call_with_timeout(
            lambda: check_camera(self.source, self.expected_profile), self.config.profile_timeout_s,
            "camera profile")
        image = self._call_with_timeout(self.source.read, self.config.verify_timeout_s, "frame")
        if image is None:
            raise SourceOpenError(f"{self.source.source_id}: source has no frames")
        return check_image(image), profile

    # -- verification (HLT-002) ------------------------------------------------------
    def verify(self) -> None:
        """Open the source and require a valid frame. Raises CaptureUnavailable (AEX-CAP-001)."""
        with self._lock:
            if self._state in (CaptureState.RUNNING, CaptureState.ENDED):
                raise CaptureError(f"capture already bound to session {self._session_id}")
            self._state, self._detail = CaptureState.UNVERIFIED, "verification in progress"
            self.profile = None
        try:
            image, profile = self._probe_source()
        except Exception as exc:  # any failure (incl. OSError, cv2.error, MemoryError) = unavailable
            if not isinstance(exc, (SourceError, InvalidImageError)):  # unexpected: keep the traceback (L2)
                log.exception("camera verification raised", extra={"code": ErrorCode.CAP_001.value})
            self._close_source()
            with self._lock:
                self._state, self._detail = CaptureState.UNVERIFIED, f"verification failed: {exc}"
            log.error("camera verification failed",
                      extra={"code": ErrorCode.CAP_001.value, "ctx": {"source": self.source.source_id,
                                                                      "error": str(exc)}})
            raise CaptureUnavailable(f"{self.source.source_id}: {exc}") from exc
        with self._lock:
            # A finite source keeps the probe frame as the first session frame (nothing skipped).
            self._pending = None if self.source.live else image
            self.profile = profile
            h, w = image.shape[:2]
            matched = f"; camera profile matched ('{profile.device_label}')" if profile else ""
            self._state, self._detail = CaptureState.VERIFIED, f"verified ({w}x{h}){matched}"

    # -- session binding -------------------------------------------------------------
    def begin(self, session_id: str) -> None:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id is mandatory")
        with self._lock:
            if self._state is not CaptureState.VERIFIED:
                raise CaptureError(f"begin requires a verified source (state {self._state})")
            if session_id in self._retired:
                raise StaleSession(f"session {session_id} was already {self._retired[session_id]}")
            self._gen += 1
            gen = self._gen
            self._session_id = session_id
            self._stop = stop = threading.Event()
            self._buffer = buffer = FrameBuffer(
                self.config.queue_max_frames, self.config.drop_alarm_threshold, drop_oldest=self.source.live,
                on_alarm=functools.partial(self._alarm, session_id=session_id))
            stamper = FrameStamper(self.source.source_id, monotonic_ns=self._mono_ns)
            pending, self._pending = self._pending, None
            self._state, self._detail = CaptureState.RUNNING, f"capturing for session {session_id}"
            self._reading_since = None
            worker = threading.Thread(target=self._worker, args=(gen, stop, buffer, stamper, pending),
                                      name="acurix-capture", daemon=True)
            watchdog = threading.Thread(target=self._watchdog, args=(gen, stop), name="acurix-capture-watchdog",
                                        daemon=True)
            self._threads = [worker, watchdog]
        worker.start()
        watchdog.start()

    def _alarm(self, counters: BufferCounters, session_id: str) -> None:
        if self.on_drop_alarm is not None:
            self.on_drop_alarm(counters, session_id)

    def next_frame(self, session_id: str | None, timeout: float | None = None) -> CapturedFrame | None:
        """Next frame for `session_id`; None on timeout. Raises EndOfStream, CaptureLost or
        StaleSession. A lost or ended session never receives a frame again."""
        with self._lock:
            if session_id is None or session_id != self._session_id:
                reason = self._retired.get(session_id or "")
                if reason == SOURCE_LOST:
                    raise CaptureLost(f"session {session_id}: capture source was lost; a new session is required")
                raise StaleSession(f"session {session_id} is not bound to the capture"
                                   + (f" ({reason})" if reason else ""))
            buffer = self._buffer
        assert buffer is not None
        try:
            return buffer.get(timeout)
        except BufferClosed as closed:
            if closed.reason == END_OF_STREAM:
                raise EndOfStream(self.source.source_id) from None
            if closed.reason == SESSION_ENDED:
                raise StaleSession(f"session {session_id} ended") from None
            raise CaptureLost(f"session {session_id}: {closed.reason}") from None

    def end(self, session_id: str | None = None) -> None:
        """Unbind the session and release the source."""
        with self._lock:
            if session_id is not None and session_id != self._session_id:
                return
            state = self._state
            buffer, sid = self._buffer, self._session_id
            if state in (CaptureState.RUNNING, CaptureState.ENDED):
                self._gen += 1
                self._stop.set()
                self._state, self._detail = CaptureState.STOPPED, "session ended; source closed"
                self._reading_since = None
            elif state is CaptureState.VERIFIED:
                self._state, self._detail = CaptureState.STOPPED, "verified but not used; source closed"
                self._pending = None
            if sid is not None:
                self._retired[sid] = SESSION_ENDED
            self._session_id = None
        if buffer is not None and state in (CaptureState.RUNNING, CaptureState.ENDED):
            buffer.close(SESSION_ENDED, discard_pending=True)
        if state in (CaptureState.RUNNING, CaptureState.ENDED, CaptureState.VERIFIED):
            self._close_source()
            self._join(self.config.frame_timeout_s)

    def shutdown(self) -> None:
        self._shutdown.set()
        self.end()
        with self._lock:
            self._gen += 1
            self._stop.set()
            if self._state is not CaptureState.STOPPED:
                self._state, self._detail = CaptureState.STOPPED, "capture shut down"
        self._close_source()
        self._join(self.config.frame_timeout_s)

    def _join(self, timeout: float) -> None:
        with self._lock:
            threads, self._threads = self._threads, []
        current = threading.current_thread()
        for t in threads:
            if t is not current:
                t.join(timeout)  # a thread blocked in a driver read is daemon and gen-fenced

    # -- producer + watchdog -----------------------------------------------------------
    def _worker(self, gen: int, stop: threading.Event, buffer: FrameBuffer, stamper: FrameStamper,
                pending: Image | None) -> None:
        """A producer that dies for ANY reason is a loss (FAULT), never a silent stop (review H1)."""
        try:
            self._produce(gen, stop, buffer, stamper, pending)
        except BaseException as exc:
            log.exception("capture worker failed", extra={"code": ErrorCode.CAP_001.value})
            self._declare_lost(gen, f"{self.source.source_id}: capture worker failed ({type(exc).__name__})")

    def _produce(self, gen: int, stop: threading.Event, buffer: FrameBuffer, stamper: FrameStamper,
                 pending: Image | None) -> None:
        if pending is not None:
            buffer.put(stamper.stamp(pending, 0.0), stop=stop)
        while not stop.is_set():
            with self._lock:
                if self._gen != gen:
                    return
                self._reading_since = self._clock()
            started = time.perf_counter()
            try:
                image = self.source.read()
            except SourceError as exc:
                self._declare_lost(gen, str(exc))
                return
            except Exception as exc:  # unexpected backend failure is still a loss, never silent
                self._declare_lost(gen, f"{self.source.source_id}: read raised {type(exc).__name__}")
                return
            finally:
                with self._lock:
                    if self._gen == gen:
                        self._reading_since = None
            latency_ms = (time.perf_counter() - started) * 1000.0
            if stop.is_set():
                return
            if image is None:
                self._declare_ended(gen, buffer)
                return
            try:
                frame = stamper.stamp(image, latency_ms)
            except InvalidImageError as exc:
                self._declare_lost(gen, f"{self.source.source_id}: invalid frame ({exc})")
                return
            buffer.put(frame, stop=stop)

    def _watchdog(self, gen: int, stop: threading.Event) -> None:
        """A watchdog that dies would stop detecting stalls: that is itself a loss (review L3)."""
        try:
            self._watch(gen, stop)
        except BaseException as exc:
            log.exception("capture watchdog failed", extra={"code": ErrorCode.CAP_001.value})
            self._declare_lost(gen, f"{self.source.source_id}: capture watchdog failed ({type(exc).__name__})")

    def _watch(self, gen: int, stop: threading.Event) -> None:
        timeout = self.config.frame_timeout_s
        period = min(0.05, timeout / 4)
        while not stop.wait(period):
            with self._lock:
                if self._gen != gen:
                    return
                since = self._reading_since
            if since is not None and self._clock() - since > timeout:
                self._declare_lost(gen, f"{self.source.source_id}: no frame for more than {timeout:g} s")
                return

    def _declare_ended(self, gen: int, buffer: FrameBuffer) -> None:
        with self._lock:
            if self._gen != gen or self._state is not CaptureState.RUNNING:
                return
            self._state, self._detail = CaptureState.ENDED, "end of stream (finite source exhausted)"
            self._stop.set()
        buffer.close(END_OF_STREAM, discard_pending=False)  # queued frames are still delivered
        log.info("capture source ended", extra={"code": "CAPTURE_END_OF_STREAM",
                                                "ctx": {"source": self.source.source_id}})

    def _declare_lost(self, gen: int, detail: str) -> None:
        with self._lock:
            if self._gen != gen or self._state is not CaptureState.RUNNING:
                return  # already ended/lost/stopped: the first cause wins
            self._gen += 1
            self._stop.set()
            self.lost_at = self._clock()
            lost_session, self._session_id = self._session_id, None
            if lost_session is not None:
                self._retired[lost_session] = SOURCE_LOST
            buffer = self._buffer
            self._state = CaptureState.LOST
            self._detail = (f"camera lost: {detail}; not reopened automatically — reset-fault and a new "
                            "session are required")
            self._reading_since = None
        if buffer is not None:
            buffer.close(SOURCE_LOST, discard_pending=True)
        log.error("camera lost", extra={"code": ErrorCode.CAP_001.value,
                                        "ctx": {"source": self.source.source_id, "detail": detail,
                                                "session_id": lost_session}})
        if self.on_lost is not None:
            try:
                self.on_lost(detail, lost_session)
            except Exception:
                log.exception("camera loss handler failed", extra={"code": "CAPTURE_LOST_HANDLER_ERROR"})
        # No automatic reconnection (owner decision 19/11/2026): the source is closed and stays closed.
        self._close_source()
