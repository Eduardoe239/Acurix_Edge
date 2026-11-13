"""Capture service: verification, producer thread, loss watchdog and reconnection
(CAP-003, CAP-004, HLT-002, HLT-004).

Lifecycle (`CaptureState`):
    UNVERIFIED --verify()--> VERIFIED --begin(session)--> RUNNING --end()--> STOPPED
    RUNNING --end of finite source--> ENDED (explicit, not an error)
    RUNNING --read error / invalid frame / read blocked > frame_timeout_s--> LOST
    LOST --reconnect attempt ok--> RECONNECTED   (never re-bound to the lost session)
    LOST --all attempts failed--> LOST (final until a new verify())

Guarantees:
- the camera is UNKNOWN for health until verify() opened it AND received a valid frame;
- on loss, `on_lost(detail, session_id)` is called once per session; the session binding is
  dropped at that instant, so frames are never delivered to the old session again — a
  reconnection only makes the source usable for a NEW session (begin() with a new id);
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
    RECONNECTING = "RECONNECTING"
    RECONNECTED = "RECONNECTED"
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
        if state in (CaptureState.LOST, CaptureState.RECONNECTING):
            return ComponentStatus.FAULT, sid + detail, ErrorCode.CAP_001
        return ComponentStatus.OK, sid + detail

    # -- source helpers -------------------------------------------------------------
    def _close_source(self) -> None:
        try:
            self.source.close()
        except Exception:
            log.exception("capture source close failed", extra={"code": "CAPTURE_CLOSE_ERROR"})

    def _read_with_timeout(self, timeout: float) -> Image | None:
        """source.read() bounded in time (a webcam read can block indefinitely)."""
        box: dict[str, Any] = {}

        def run() -> None:
            try:
                box["image"] = self.source.read()
            except BaseException as exc:  # re-raised in the caller thread
                box["error"] = exc

        t = threading.Thread(target=run, name="acurix-capture-probe", daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            raise SourceLostError(f"{self.source.source_id}: no frame within {timeout:g} s")
        if "error" in box:
            raise box["error"]
        return box.get("image")

    def _probe_source(self) -> Image:
        """Open the source and get one valid frame, or raise SourceError/InvalidImageError."""
        if not self.source.live:
            self._close_source()  # finite sources are rewound: verification never consumes a frame
        if not self.source.is_open:
            self.source.open()
        image = self._read_with_timeout(self.config.verify_timeout_s)
        if image is None:
            raise SourceOpenError(f"{self.source.source_id}: source has no frames")
        return check_image(image)

    # -- verification (HLT-002) ------------------------------------------------------
    def verify(self) -> None:
        """Open the source and require a valid frame. Raises CaptureUnavailable (AEX-CAP-001)."""
        with self._lock:
            if self._state in (CaptureState.RUNNING, CaptureState.ENDED):
                raise CaptureError(f"capture already bound to session {self._session_id}")
            if self._state is CaptureState.RECONNECTING:
                raise CaptureUnavailable(f"{self.source.source_id}: reconnection in progress ({self._detail})")
            self._state, self._detail = CaptureState.UNVERIFIED, "verification in progress"
        try:
            image = self._probe_source()
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
            h, w = image.shape[:2]
            self._state, self._detail = CaptureState.VERIFIED, f"verified ({w}x{h})"

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
        """Unbind the session and release the source. Never cancels an ongoing reconnection."""
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
            new_gen = self._gen
            self._stop.set()
            self.lost_at = self._clock()
            lost_session, self._session_id = self._session_id, None
            if lost_session is not None:
                self._retired[lost_session] = SOURCE_LOST
            buffer = self._buffer
            reconnect = self.config.reconnect_max_attempts > 0 and not self._shutdown.is_set()
            # RECONNECTING is set here, under the lock, so verify() can never slip in between (review B1).
            self._state = CaptureState.RECONNECTING if reconnect else CaptureState.LOST
            self._detail = f"camera lost: {detail}"
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
        if reconnect:
            t = threading.Thread(target=self._reconnect, args=(new_gen,), name="acurix-capture-reconnect",
                                 daemon=True)
            with self._lock:
                self._threads.append(t)
            t.start()
        else:
            self._close_source()

    # -- reconnection (CAP-004) ------------------------------------------------------
    def _reconnect(self, gen: int) -> None:
        """A reconnection thread that fails unexpectedly ends in a final LOST, never stuck in
        RECONNECTING (review M2)."""
        try:
            self._try_reconnect(gen)
        except BaseException as exc:
            log.exception("camera reconnection crashed", extra={"code": ErrorCode.CAP_001.value})
            self._close_source()
            with self._lock:
                if self._gen == gen:
                    self._state = CaptureState.LOST
                    self._detail = f"camera lost; reconnection aborted ({type(exc).__name__})"

    def _try_reconnect(self, gen: int) -> None:
        self._close_source()
        attempts = self.config.reconnect_max_attempts
        last_error = ""
        for attempt in range(1, attempts + 1):
            if self._shutdown.wait(self.config.reconnect_interval_s):
                return
            with self._lock:
                if self._gen != gen:
                    return
                self._detail = f"reconnecting (attempt {attempt}/{attempts})"
            try:
                self._probe_source()
            except Exception as exc:  # an unexpected error is one more failed attempt
                last_error = f"{type(exc).__name__}: {exc}"
                self._close_source()
                continue
            with self._lock:
                superseded = self._gen != gen
                if not superseded:
                    self._state = CaptureState.RECONNECTED
                    self._detail = (f"reconnected after {attempt} attempt(s); previous session discarded, "
                                    "a new session is required")
            if superseded:
                self._close_source()  # outside the lock: a driver stuck in release never blocks health (L4)
                return
            log.warning("camera reconnected; previous session is NOT resumed",
                        extra={"code": "CAPTURE_RECONNECTED", "ctx": {"source": self.source.source_id,
                                                                      "attempts": attempt}})
            return
        with self._lock:
            if self._gen == gen:
                self._state = CaptureState.LOST
                self._detail = f"camera lost; reconnection failed after {attempts} attempt(s): {last_error}"
        log.error("camera reconnection failed", extra={"code": ErrorCode.CAP_001.value,
                                                       "ctx": {"source": self.source.source_id,
                                                               "attempts": attempts}})
