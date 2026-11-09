"""Health aggregation that never reports false normality.

Aggregation:
- any critical component FAULT or UNKNOWN  -> FAULT (cannot validate pre-conditions)
- otherwise any component not OK           -> DEGRADED
- OK only when every registered component is OK
A check that raises is reported as FAULT for that component.
"""

from __future__ import annotations

import ctypes
import logging
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from acurix_edge.contracts.events import utc_now_iso

log = logging.getLogger("acurix_edge.health")


class ComponentStatus(StrEnum):
    OK = "OK"
    DEGRADED = "DEGRADED"
    FAULT = "FAULT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class ComponentHealth:
    name: str
    status: ComponentStatus
    detail: str
    critical: bool

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.status.value, "detail": self.detail, "critical": self.critical}


@dataclass(frozen=True, slots=True)
class HealthReport:
    checked_at_utc: str
    overall: ComponentStatus
    components: tuple[ComponentHealth, ...]

    @property
    def ok(self) -> bool:
        return self.overall is ComponentStatus.OK

    def component(self, name: str) -> ComponentHealth | None:
        return next((c for c in self.components if c.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked_at_utc": self.checked_at_utc,
            "overall": self.overall.value,
            "components": [c.to_dict() for c in self.components],
        }


Check = Callable[[], tuple[ComponentStatus, str]]


class HealthService:
    def __init__(self) -> None:
        self._checks: dict[str, tuple[Check, bool]] = {}

    def register(self, name: str, check: Check, *, critical: bool) -> None:
        if name in self._checks:
            raise ValueError(f"health check {name!r} already registered")
        self._checks[name] = (check, critical)

    def evaluate(self) -> HealthReport:
        results: list[ComponentHealth] = []
        for name, (check, critical) in self._checks.items():
            try:
                status, detail = check()
                if not isinstance(status, ComponentStatus):
                    status, detail = ComponentStatus.FAULT, "check returned invalid status"
            except Exception as exc:
                log.exception("health check failed", extra={"code": "HEALTH_CHECK_ERROR", "ctx": {"component": name}})
                status, detail = ComponentStatus.FAULT, f"check error: {type(exc).__name__}"
            results.append(ComponentHealth(name, status, detail, critical))
        return HealthReport(utc_now_iso(), self._aggregate(results), tuple(results))

    @staticmethod
    def _aggregate(results: list[ComponentHealth]) -> ComponentStatus:
        if not results:
            return ComponentStatus.UNKNOWN
        if any(r.critical and r.status in (ComponentStatus.FAULT, ComponentStatus.UNKNOWN) for r in results):
            return ComponentStatus.FAULT
        if all(r.status is ComponentStatus.OK for r in results):
            return ComponentStatus.OK
        return ComponentStatus.DEGRADED


def storage_check(path: Path, min_free_mb: int) -> Check:
    def _check() -> tuple[ComponentStatus, str]:
        target = path if path.exists() else path.parent
        free_mb = shutil.disk_usage(target).free // (1024 * 1024)
        if free_mb < min_free_mb:
            return ComponentStatus.DEGRADED, f"free {free_mb} MB < minimum {min_free_mb} MB"
        return ComponentStatus.OK, f"free {free_mb} MB"

    return _check


def host_memory() -> dict[str, float] | None:
    """Physical memory snapshot in GB (Windows only; None elsewhere). Informational."""
    if sys.platform != "win32":
        return None

    class _MemStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = _MemStatus()
    stat.dwLength = ctypes.sizeof(_MemStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
        return None
    gb = 1024 ** 3
    return {"total_gb": round(stat.ullTotalPhys / gb, 2), "available_gb": round(stat.ullAvailPhys / gb, 2)}
