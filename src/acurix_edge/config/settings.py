"""Typed, validated configuration with content hash.

Configuration version is independent from software and model versions
(ARC-010). Unknown keys are rejected; all problems are reported together.
Secrets must never be placed in this file (use environment/secret store).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePath
from typing import Any

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1"})  # literal IPs only; 'localhost' depends on hosts file
SUPPORTED_INFERENCE_ADAPTERS = frozenset({"noop"})  # ADR-016: NoOp is the only authorized adapter
LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
_VERSION_RE = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class ConfigError(ValueError):
    def __init__(self, problems: list[str]) -> None:
        super().__init__("invalid configuration: " + "; ".join(problems))
        self.problems = problems


@dataclass(frozen=True, slots=True)
class StorageConfig:
    data_dir: str = ".data"
    database_file: str = "acurix.db"
    min_free_disk_mb: int = 1024


@dataclass(frozen=True, slots=True)
class ApiConfig:
    bind_host: str = "127.0.0.1"
    port: int = 8765


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    level: str = "INFO"


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    adapter: str = "noop"


CLOCK_TOLERANCE_MAX_S = 10.0  # schema bound: a larger tolerance would hide a real clock regression


@dataclass(frozen=True, slots=True)
class ClockConfig:
    """Clock trust (owner decision 10/11/2026, AEX-CLK-001). A clock up to
    `backward_tolerance_s` behind the last persisted event is tolerated and logged; beyond it,
    FAULT. The same tolerance applies to backward jumps against a time already observed."""

    backward_tolerance_s: float = 2.0


@dataclass(frozen=True, slots=True)
class ModelsConfig:
    """Model approval policy (ADR-016): default-deny.

    Foundation 0.1.0 locks this section: `approved_models` must be empty and
    `real_model_loading` must be false. Changing either requires an owner
    decision recorded in a new ADR revision, license/hash audit and a software
    change that lifts this lock — it cannot be enabled by configuration alone.
    """

    approved_models: tuple[Any, ...] = ()
    real_model_loading: bool = False


@dataclass(frozen=True, slots=True)
class AppConfig:
    config_version: str = "0.1.0"
    site_id: str = "local-site"
    station_id: str = "station-01"
    storage: StorageConfig = field(default_factory=StorageConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    clock: ClockConfig = field(default_factory=ClockConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def database_path(self, base_dir: Path) -> Path:
        return (base_dir / self.storage.data_dir / self.storage.database_file).resolve()

    def reference(self) -> dict[str, str]:
        return {"version": self.config_version, "sha256": self.sha256}


_SECTIONS: dict[str, type] = {
    "storage": StorageConfig,
    "api": ApiConfig,
    "logging": LoggingConfig,
    "inference": InferenceConfig,
    "models": ModelsConfig,
    "clock": ClockConfig,
}
_TOP_LEVEL = {"config_version", "site_id", "station_id", *_SECTIONS}


_WIN_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
)


def _is_safe_relative(p: Any) -> bool:
    if not isinstance(p, str) or not p:
        return False
    pure = PurePath(p.replace("\\", "/"))
    return (
        not p.startswith(("/", "\\"))
        and not pure.is_absolute()
        and ":" not in p
        and ".." not in pure.parts
        and not any(part.split(".")[0].upper() in _WIN_RESERVED for part in pure.parts)
    )


def _in(value: Any, allowed: frozenset[str]) -> bool:
    return isinstance(value, str) and value in allowed


def _build(data: dict[str, Any]) -> AppConfig:
    problems: list[str] = []
    unknown = set(data) - _TOP_LEVEL
    if unknown:
        problems.append(f"unknown keys: {sorted(unknown)}")
    sections: dict[str, Any] = {}
    for name, cls in _SECTIONS.items():
        raw = data.get(name, {})
        if not isinstance(raw, dict):
            problems.append(f"[{name}] must be a table")
            continue
        allowed = set(cls.__dataclass_fields__)
        extra = set(raw) - allowed
        if extra:
            problems.append(f"unknown keys in [{name}]: {sorted(extra)}")
        try:
            sections[name] = cls(**{k: v for k, v in raw.items() if k in allowed})
        except TypeError as exc:
            problems.append(f"[{name}]: {exc}")
    if problems:
        raise ConfigError(problems)

    defaults = AppConfig()
    cfg = AppConfig(
        config_version=data.get("config_version", defaults.config_version),
        site_id=data.get("site_id", defaults.site_id),
        station_id=data.get("station_id", defaults.station_id),
        **sections,
    )
    validate(cfg)
    return cfg


def validate(cfg: AppConfig) -> None:
    p: list[str] = []
    if not isinstance(cfg.config_version, str) or not _VERSION_RE.fullmatch(cfg.config_version):
        p.append("config_version must be semantic version (x.y.z)")
    for name in ("site_id", "station_id"):
        v = getattr(cfg, name)
        if not isinstance(v, str) or not _ID_RE.fullmatch(v):
            p.append(f"{name} must match {_ID_RE.pattern}")
    s = cfg.storage
    if not isinstance(s.data_dir, str) or not _is_safe_relative(s.data_dir):
        p.append("storage.data_dir must be a relative path without '..'")
    if not isinstance(s.database_file, str) or not _is_safe_relative(s.database_file) or "/" in s.database_file or "\\" in s.database_file:
        p.append("storage.database_file must be a plain file name")
    if not isinstance(s.min_free_disk_mb, int) or isinstance(s.min_free_disk_mb, bool) or s.min_free_disk_mb < 0:
        p.append("storage.min_free_disk_mb must be int >= 0")
    if not _in(cfg.api.bind_host, LOOPBACK_HOSTS):
        p.append("api.bind_host must be a literal loopback IP (127.0.0.1 or ::1) in 0.1.0 (ARC-007)")
    if not isinstance(cfg.api.port, int) or isinstance(cfg.api.port, bool) or not 1024 <= cfg.api.port <= 65535:
        p.append("api.port must be int in 1024..65535")
    if not _in(cfg.logging.level, LOG_LEVELS):
        p.append(f"logging.level must be one of {sorted(LOG_LEVELS)}")
    m = cfg.models
    if not isinstance(m.approved_models, (list, tuple)) or len(m.approved_models) != 0:
        p.append("models.approved_models must be empty in 0.1.0 (default-deny, ADR-016)")
    if m.real_model_loading is not False:
        p.append("models.real_model_loading must be false in 0.1.0 (ADR-016)")
    if not _in(cfg.inference.adapter, SUPPORTED_INFERENCE_ADAPTERS):
        p.append(f"inference.adapter must be one of {sorted(SUPPORTED_INFERENCE_ADAPTERS)}")
    tol = cfg.clock.backward_tolerance_s
    if (not isinstance(tol, (int, float)) or isinstance(tol, bool) or not math.isfinite(tol)
            or not 0 <= tol <= CLOCK_TOLERANCE_MAX_S):
        p.append(f"clock.backward_tolerance_s must be a number in 0..{CLOCK_TOLERANCE_MAX_S:g} (seconds)")
    if p:
        raise ConfigError(p)


def load_config(path: Path | None = None) -> AppConfig:
    """Load and validate configuration. `None` returns validated defaults."""
    if path is None:
        cfg = AppConfig()
        validate(cfg)
        return cfg
    try:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except FileNotFoundError as exc:
        raise ConfigError([f"configuration file not found: {path.name}"]) from exc
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ConfigError([f"invalid TOML: {exc}"]) from exc
    except OSError as exc:
        raise ConfigError([f"configuration file unreadable: {type(exc).__name__}"]) from exc
    return _build(data)
