"""Concrete health checks (AEX-HEALTH-005). Standard library only; nothing is installed.

Thresholds used here and their source:
- RAM: available < 1536 MB -> DEGRADED + AEX-CAS-004 (ADR-014 item 11: below 1.5 GB only
  light work). Not a guarantee of execution, a conservative guardrail.
- Disk: free < storage.min_free_disk_mb (configuration) -> DEGRADED + AEX-DB-002. An actual
  write failure for lack of space is handled by persistence (AEX-DB-002 + FAULT).
- Clock: earlier than CLOCK_FLOOR_UTC (Foundation release month) -> FAULT (timestamps cannot be
  trusted); a backward jump > CLOCK_BACKWARD_TOLERANCE_S between checks -> DEGRADED.
  The AEX catalog has no clock code: `code` stays None (reported as a catalog gap).
- CPU: informational (no approved load threshold); unreadable -> UNKNOWN.
- GPU: optional. No `nvidia-smi` on PATH -> DISABLED (CPU fallback is mandatory); probe
  error/timeout -> DEGRADED. Result cached CACHE_S seconds (subprocess is not run per frame).
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from acurix_edge.contracts.errors import ErrorCode
from acurix_edge.health.service import CheckResult, ComponentStatus, host_memory

ADR014_MIN_FREE_RAM_MB = 1536
CLOCK_FLOOR_UTC = datetime(2026, 11, 1, tzinfo=timezone.utc)
CLOCK_BACKWARD_TOLERANCE_S = 2.0
GPU_CACHE_S = 30.0
GPU_TIMEOUT_S = 3.0


def storage_check(path: Path, min_free_mb: int) -> Callable[[], CheckResult]:
    def _check() -> CheckResult:
        target = path if path.exists() else path.parent
        free_mb = shutil.disk_usage(target).free // (1024 * 1024)
        if free_mb < min_free_mb:
            return ComponentStatus.DEGRADED, f"free {free_mb} MB < minimum {min_free_mb} MB", ErrorCode.DB_002
        return ComponentStatus.OK, f"free {free_mb} MB"

    return _check


def memory_check(probe: Callable[[], dict[str, float] | None] = host_memory,
                 min_free_mb: int = ADR014_MIN_FREE_RAM_MB) -> Callable[[], CheckResult]:
    def _check() -> CheckResult:
        mem = probe()
        if mem is None:
            return ComponentStatus.UNKNOWN, "host memory not measurable on this platform"
        free_mb = int(mem["available_gb"] * 1024)
        detail = f"available {free_mb} MB of {mem['total_gb']} GB (minimum {min_free_mb} MB, ADR-014)"
        if free_mb < min_free_mb:
            return ComponentStatus.DEGRADED, detail, ErrorCode.CAS_004
        return ComponentStatus.OK, detail

    return _check


class ClockCheck:
    def __init__(self, now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                 monotonic: Callable[[], float] = time.monotonic) -> None:
        self._now, self._mono = now, monotonic
        self._last: tuple[datetime, float] | None = None
        self._lock = threading.Lock()

    def __call__(self) -> CheckResult:
        wall, mono = self._now(), self._mono()
        with self._lock:
            last, self._last = self._last, (wall, mono)
        if wall < CLOCK_FLOOR_UTC:
            return ComponentStatus.FAULT, f"system clock {wall.isoformat()} earlier than {CLOCK_FLOOR_UTC.date()}"
        if last is not None:
            drift = (wall - last[0]).total_seconds() - (mono - last[1])
            if drift < -CLOCK_BACKWARD_TOLERANCE_S:
                return ComponentStatus.DEGRADED, f"wall clock jumped back {-drift:.1f} s since last check"
        return ComponentStatus.OK, "UTC clock plausible"


class CpuCheck:
    """Load between consecutive checks via GetSystemTimes (Windows). Informational."""

    def __init__(self) -> None:
        self._prev: tuple[int, int, int] | None = None
        self._lock = threading.Lock()

    @staticmethod
    def _times() -> tuple[int, int, int] | None:
        if sys.platform != "win32":
            return None
        idle, kernel, user = ctypes.c_ulonglong(), ctypes.c_ulonglong(), ctypes.c_ulonglong()
        if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel),  # type: ignore[attr-defined]
                                                     ctypes.byref(user)):
            return None
        return idle.value, kernel.value, user.value

    def __call__(self) -> CheckResult:
        import os

        cur = self._times()
        if cur is None:
            return ComponentStatus.UNKNOWN, "CPU load not measurable on this platform"
        with self._lock:
            prev, self._prev = self._prev, cur
        cpus = os.cpu_count() or 0
        if prev is None:
            return ComponentStatus.OK, f"{cpus} logical CPUs; load sampled from next check"
        d_idle, d_total = cur[0] - prev[0], (cur[1] - prev[1]) + (cur[2] - prev[2])
        load = 0.0 if d_total <= 0 else max(0.0, min(100.0, 100.0 * (1 - d_idle / d_total)))
        return ComponentStatus.OK, f"{cpus} logical CPUs; load {load:.0f}%"


class GpuCheck:
    def __init__(self, which: Callable[[str], str | None] = shutil.which,
                 run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._which, self._run, self._clock = which, run, clock
        self._cache: tuple[float, CheckResult] | None = None
        self._lock = threading.Lock()

    def __call__(self) -> CheckResult:
        with self._lock:
            if self._cache is not None and self._clock() - self._cache[0] < GPU_CACHE_S:
                return self._cache[1]
            result = self._probe()
            self._cache = (self._clock(), result)
            return result

    def _probe(self) -> CheckResult:
        exe = self._which("nvidia-smi")
        if exe is None:
            return ComponentStatus.DISABLED, "no NVIDIA tool on PATH; CPU fallback"
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            out = self._run([exe, "--query-gpu=name,memory.total,memory.used,temperature.gpu",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True,
                            timeout=GPU_TIMEOUT_S, check=True, creationflags=flags)
            first = out.stdout.strip().splitlines()[0]
            name, total, used, temp = (p.strip() for p in first.split(","))
            return ComponentStatus.OK, f"{name}; VRAM {used}/{total} MiB; {temp} C"
        except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
            return ComponentStatus.DEGRADED, f"GPU probe failed: {type(exc).__name__}"


class QueueGauge:
    """Inspections waiting or in flight. 0.1.0 processes synchronously; no frame is dropped
    by design, so `dropped` stays 0 unless a future capture queue reports drops."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.depth = 0
        self.max_depth = 0
        self.dropped = 0
        self.received = 0

    def enter(self) -> None:
        with self._lock:
            self.received += 1
            self.depth += 1
            self.max_depth = max(self.max_depth, self.depth)

    def leave(self) -> None:
        with self._lock:
            self.depth -= 1

    def drop(self) -> None:
        with self._lock:
            self.dropped += 1

    def check(self) -> CheckResult:
        with self._lock:
            depth, max_depth, dropped = self.depth, self.max_depth, self.dropped
        detail = f"depth {depth}, max {max_depth}, dropped {dropped} (synchronous processing in 0.1.0)"
        if dropped:
            return ComponentStatus.DEGRADED, detail, ErrorCode.CAP_002
        return ComponentStatus.OK, detail
