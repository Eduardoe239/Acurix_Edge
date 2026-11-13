"""Health aggregation that never reports false normality (AEX-HEALTH-005, HLT-006).

Entries are split in two groups (owner decision 10/11/2026):
- `components`: only the OpenAPI ComponentHealth names (camera, models, database,
  storage, clock, peripheral);
- `resources`: host CPU/RAM/GPU, frame queue and decision rule — outside the
  OpenAPI enum, exposed as an extra Status block. They feed `overall` too.

Statuses follow the OpenAPI enum: OK, DEGRADED, FAULT, UNKNOWN, DISABLED.

Aggregation ("worst relevant entry"):
- any critical entry FAULT or UNKNOWN          -> FAULT (pre-conditions cannot be validated)
- otherwise any entry not OK/DISABLED           -> DEGRADED
- OK only when every entry is OK or DISABLED
DISABLED means "deliberately not in use" (e.g. no GPU: CPU fallback) and is only valid
for non-critical entries; a critical DISABLED is reported as FAULT. A check that raises
or returns an invalid result is FAULT for that entry. No entries -> UNKNOWN.
"""

from __future__ import annotations

import ctypes
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from acurix_edge.contracts.errors import ErrorCode
from acurix_edge.contracts.events import utc_now_iso

log = logging.getLogger("acurix_edge.health")

# OpenAPI ComponentHealth.component enum (docs/02-architecture/api/openapi-v1.yaml).
OPENAPI_COMPONENTS = frozenset({"camera", "models", "database", "storage", "clock", "peripheral"})
COMPONENTS = "components"
RESOURCES = "resources"


class ComponentStatus(StrEnum):
    OK = "OK"
    DEGRADED = "DEGRADED"
    FAULT = "FAULT"
    UNKNOWN = "UNKNOWN"
    DISABLED = "DISABLED"


@dataclass(frozen=True, slots=True)
class ComponentHealth:
    name: str
    status: ComponentStatus
    detail: str
    critical: bool
    code: ErrorCode | None = None
    checked_at_utc: str = ""
    group: str = COMPONENTS

    def to_dict(self) -> dict[str, Any]:
        # OpenAPI ComponentHealth fields + `code` and `critical` (extra properties).
        return {"component": self.name, "status": self.status.value, "detail": self.detail,
                "checked_at_utc": self.checked_at_utc, "code": self.code.value if self.code else None,
                "critical": self.critical}


@dataclass(frozen=True, slots=True)
class HealthReport:
    checked_at_utc: str
    overall: ComponentStatus
    components: tuple[ComponentHealth, ...]
    resources: tuple[ComponentHealth, ...] = ()

    @property
    def ok(self) -> bool:
        return self.overall is ComponentStatus.OK

    @property
    def all_entries(self) -> tuple[ComponentHealth, ...]:
        return self.components + self.resources

    def component(self, name: str) -> ComponentHealth | None:
        return next((c for c in self.all_entries if c.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked_at_utc": self.checked_at_utc,
            "overall": self.overall.value,
            "components": [c.to_dict() for c in self.components],
            "resources": [c.to_dict() for c in self.resources],
        }


CheckResult = tuple[ComponentStatus, str] | tuple[ComponentStatus, str, ErrorCode | None]
Check = Callable[[], CheckResult]


class HealthService:
    def __init__(self) -> None:
        self._checks: dict[str, tuple[Check, bool, str]] = {}

    def register(self, name: str, check: Check, *, critical: bool, group: str = COMPONENTS) -> None:
        if name in self._checks:
            raise ValueError(f"health check {name!r} already registered")
        if group not in (COMPONENTS, RESOURCES):
            raise ValueError(f"unknown health group {group!r}")
        self._checks[name] = (check, critical, group)

    def evaluate(self) -> HealthReport:
        groups: dict[str, list[ComponentHealth]] = {COMPONENTS: [], RESOURCES: []}
        for name, (check, critical, group) in self._checks.items():
            code: ErrorCode | None = None
            try:
                # typed as object: a check may violate its signature; validated below at runtime
                result: object = check()
                if not isinstance(result, tuple) or len(result) not in (2, 3):
                    raise TypeError("check must return (status, detail[, code])")
                status, detail = result[0], result[1]
                code = result[2] if len(result) == 3 else None
                if not isinstance(status, ComponentStatus) or not isinstance(detail, str) or (
                        code is not None and not isinstance(code, ErrorCode)):
                    status, detail, code = ComponentStatus.FAULT, "check returned invalid status", ErrorCode.SYS_001
                elif critical and status is ComponentStatus.DISABLED:
                    status, detail = ComponentStatus.FAULT, f"critical component cannot be DISABLED ({detail})"
            except Exception as exc:
                log.exception("health check failed", extra={"code": ErrorCode.SYS_001.value,
                                                            "ctx": {"component": name}})
                status, detail, code = ComponentStatus.FAULT, f"check error: {type(exc).__name__}", ErrorCode.SYS_001
            groups[group].append(ComponentHealth(name, status, detail, critical, code, utc_now_iso(), group))
        entries = groups[COMPONENTS] + groups[RESOURCES]
        return HealthReport(utc_now_iso(), self._aggregate(entries), tuple(groups[COMPONENTS]),
                            tuple(groups[RESOURCES]))

    @staticmethod
    def _aggregate(results: list[ComponentHealth]) -> ComponentStatus:
        if not results:
            return ComponentStatus.UNKNOWN
        if any(r.critical and r.status in (ComponentStatus.FAULT, ComponentStatus.UNKNOWN) for r in results):
            return ComponentStatus.FAULT
        if all(r.status in (ComponentStatus.OK, ComponentStatus.DISABLED) for r in results):
            return ComponentStatus.OK
        return ComponentStatus.DEGRADED


def host_memory() -> dict[str, float] | None:
    """Physical memory snapshot in GB (Windows only; None elsewhere)."""
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
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined,unused-ignore]  # windll: Windows only
        return None
    gb = 1024 ** 3
    return {"total_gb": round(stat.ullTotalPhys / gb, 2), "available_gb": round(stat.ullAvailPhys / gb, 2)}
