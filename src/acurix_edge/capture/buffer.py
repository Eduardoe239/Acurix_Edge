"""Bounded capture queue with counted drops (CAP-003). Never a silent drop.

Policy:
- live source (webcam): when full, the OLDEST frame is dropped (cause QUEUE_FULL_DROP_OLDEST),
  so the consumer always sees the most recent scene;
- non-live source (folder, video file): the producer waits for space (back-pressure), no drop.
Frames still queued when the buffer is closed are counted as dropped with the close cause
(e.g. SOURCE_LOST, SESSION_ENDED). Invariant (tested):
    captured == delivered + sum(dropped_by_cause) + depth
When the frames dropped in this buffer (excluding the normal SESSION_ENDED) go above
`drop_alarm_threshold`, `on_alarm` is called once (AEX-CAP-002) and the health `queue` entry
turns DEGRADED.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from acurix_edge.contracts.capture import CapturedFrame
from acurix_edge.contracts.errors import ErrorCode

log = logging.getLogger("acurix_edge.capture.buffer")

DROP_QUEUE_FULL = "QUEUE_FULL_DROP_OLDEST"
SESSION_ENDED = "SESSION_ENDED"
# Counted like every drop, but a normal end of session is not a capture problem: it does not
# count toward the AEX-CAP-002 limit (review B2).
NON_ALARM_CAUSES = frozenset({SESSION_ENDED})


@dataclass(frozen=True, slots=True)
class BufferCounters:
    captured: int
    delivered: int
    dropped_by_cause: dict[str, int] = field(default_factory=dict)
    depth: int = 0
    max_depth: int = 0
    capacity: int = 0
    drop_alarm_threshold: int = 0

    @property
    def dropped(self) -> int:
        return sum(self.dropped_by_cause.values())

    @property
    def alarm_dropped(self) -> int:
        return sum(n for cause, n in self.dropped_by_cause.items() if cause not in NON_ALARM_CAUSES)

    @property
    def over_limit(self) -> bool:
        return self.alarm_dropped > self.drop_alarm_threshold

    def to_dict(self) -> dict[str, Any]:
        return {"captured": self.captured, "delivered": self.delivered, "dropped": self.dropped,
                "dropped_by_cause": dict(sorted(self.dropped_by_cause.items())), "depth": self.depth,
                "max_depth": self.max_depth, "capacity": self.capacity, "alarm_dropped": self.alarm_dropped,
                "drop_alarm_threshold": self.drop_alarm_threshold, "over_limit": self.over_limit}


class BufferClosed(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class FrameBuffer:
    def __init__(self, capacity: int, drop_alarm_threshold: int, *, drop_oldest: bool,
                 on_alarm: Callable[[BufferCounters], None] | None = None) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be int >= 1")
        if isinstance(drop_alarm_threshold, bool) or not isinstance(drop_alarm_threshold, int) \
                or drop_alarm_threshold < 0:
            raise ValueError("drop_alarm_threshold must be int >= 0")
        self.capacity, self.threshold, self.drop_oldest = capacity, drop_alarm_threshold, drop_oldest
        self._on_alarm = on_alarm
        self._q: deque[CapturedFrame] = deque()
        self._cond = threading.Condition()
        self._captured = self._delivered = self._max_depth = 0
        self._dropped: Counter[str] = Counter()
        self._closed: str | None = None
        self._alarmed = False

    # caller holds self._cond
    def _drop(self, cause: str, n: int = 1) -> bool:
        self._dropped[cause] += n
        total = sum(self._dropped.values())
        log.debug("capture frame dropped", extra={"code": ErrorCode.CAP_002.value,
                                                  "ctx": {"cause": cause, "dropped_total": total}})
        alarm_total = sum(v for k, v in self._dropped.items() if k not in NON_ALARM_CAUSES)
        if alarm_total > self.threshold and not self._alarmed:
            self._alarmed = True
            return True
        return False

    def _snapshot(self) -> BufferCounters:
        return BufferCounters(self._captured, self._delivered, dict(self._dropped), len(self._q), self._max_depth,
                              self.capacity, self.threshold)

    def _fire_alarm(self, counters: BufferCounters) -> None:
        log.warning("capture drops above configured limit",
                    extra={"code": ErrorCode.CAP_002.value, "ctx": counters.to_dict()})
        if self._on_alarm is not None:
            try:
                self._on_alarm(counters)
            except Exception:
                log.exception("capture drop alarm handler failed", extra={"code": "CAPTURE_ALARM_HANDLER_ERROR"})

    def put(self, frame: CapturedFrame, *, stop: threading.Event | None = None) -> bool:
        """Queue a frame. Returns False if the buffer is closed (the frame is then counted as
        dropped with the close cause)."""
        alarm = None
        with self._cond:
            if not self.drop_oldest:  # back-pressure: wait for space, a close or a stop request
                while (self._closed is None and len(self._q) >= self.capacity
                       and not (stop is not None and stop.is_set())):
                    self._cond.wait(0.05)
            self._captured += 1  # counted after the wait: the invariant holds at every snapshot
            cause: str | None = None
            if self._closed is not None:
                cause = self._closed
            elif not self.drop_oldest and len(self._q) >= self.capacity:
                cause = "PRODUCER_STOPPED"  # the wait above ended by a stop request
            if cause is not None:
                if self._drop(cause):
                    alarm = self._snapshot()
                accepted = False
            else:
                if len(self._q) >= self.capacity:
                    self._q.popleft()
                    if self._drop(DROP_QUEUE_FULL):
                        alarm = self._snapshot()
                self._q.append(frame)
                self._max_depth = max(self._max_depth, len(self._q))
                accepted = True
            self._cond.notify_all()
        if alarm is not None:
            self._fire_alarm(alarm)
        return accepted

    def get(self, timeout: float | None = None) -> CapturedFrame | None:
        """Next frame; None on timeout. Raises BufferClosed once closed AND empty."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while not self._q:
                if self._closed is not None:
                    raise BufferClosed(self._closed)
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._cond.wait(remaining)
            frame = self._q.popleft()
            self._delivered += 1
            self._cond.notify_all()
            return frame

    def close(self, reason: str, *, discard_pending: bool) -> None:
        """Close for producers. With discard_pending, queued frames are counted as dropped
        (cause = reason) and consumers get BufferClosed immediately."""
        alarm = None
        with self._cond:
            if self._closed is None:
                self._closed = reason
            if discard_pending and self._q:
                n = len(self._q)
                self._q.clear()
                if self._drop(reason, n):
                    alarm = self._snapshot()
            self._cond.notify_all()
        if alarm is not None:
            self._fire_alarm(alarm)

    @property
    def closed_reason(self) -> str | None:
        with self._cond:
            return self._closed

    def counters(self) -> BufferCounters:
        with self._cond:
            return self._snapshot()
