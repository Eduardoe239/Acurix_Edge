"""Camera identity profile (owner decision 19/11/2026, option b).

On every open of a camera (session verification, bench capture) the source measures a profile:
backend, default resolution, the set of resolutions it actually delivers for a fixed list of
requests, and the configured device label. It must equal the EXPECTED profile stored in the
configuration (`[capture] camera_*`); otherwise the camera is refused with AEX-CAP-001 and the
system never reaches OBSERVING.

HEURISTIC, not an identity proof: two units of the same webcam model have the same profile, and a
driver/firmware update may change it. Identification by device name/path (option c) needs a new
dependency and is deferred until after the demo, subject to approval.

Fail-closed: a camera source with no expected profile configured is refused as well.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from acurix_edge.capture.sources import SourceOpenError

MISMATCH_MESSAGE = "a câmera aberta não corresponde à câmera configurada"
NOT_CONFIGURED_MESSAGE = "nenhum perfil de câmera configurado"
# Requested sizes; the profile records what the driver NEGOTIATES for each (deduplicated).
# Each request costs ~0.6-0.7 s on DirectShow (measured 19/11/2026): sizes every webcam supports
# (320x240, 640x480) are left out; 3840x2160 reveals the maximum.
PROFILE_CANDIDATES: tuple[tuple[int, int], ...] = (
    (800, 600), (960, 540), (1280, 720), (1920, 1080), (3840, 2160))
RESOLUTION_RE = re.compile(r"[1-9][0-9]{0,4}x[1-9][0-9]{0,4}")
BACKEND_RE = re.compile(r"[A-Za-z0-9_-]{1,32}")
MAX_RESOLUTIONS = 32
MAX_LABEL = 200


class CameraMismatch(SourceOpenError):
    """The opened camera is not the configured one (or no camera is configured). AEX-CAP-001."""


def resolution(w: int, h: int) -> str:
    return f"{int(w)}x{int(h)}"


def _res_key(r: str) -> tuple[int, int]:
    w, h = (int(x) for x in r.split("x"))
    return w * h, w


def sort_resolutions(items: Any) -> tuple[str, ...]:
    return tuple(sorted(set(items), key=_res_key))


@dataclass(frozen=True, slots=True)
class CameraProfile:
    backend: str
    default_resolution: str
    supported_resolutions: tuple[str, ...]
    device_label: str

    def to_dict(self) -> dict[str, Any]:
        return {"backend": self.backend, "default_resolution": self.default_resolution,
                "supported_resolutions": list(self.supported_resolutions), "device_label": self.device_label}

    def to_toml(self) -> str:
        """The `[capture]` keys that store this profile as the expected one."""
        res = ", ".join(f'"{r}"' for r in self.supported_resolutions)
        label = self.device_label.replace("\\", "\\\\").replace('"', '\\"')
        return (f'camera_backend = "{self.backend}"\n'
                f'camera_default_resolution = "{self.default_resolution}"\n'
                f"camera_resolutions = [{res}]\n"
                f'camera_device_label = "{label}"\n')


def label_ok(v: Any) -> bool:
    return isinstance(v, str) and 0 < len(v) <= MAX_LABEL and v.isprintable() and v.strip() == v


def profile_problems(backend: Any, default_resolution: Any, resolutions: Any, device_label: Any) -> list[str]:
    """Schema of an expected profile. All four empty = not configured (no problem here)."""
    if backend == "" and default_resolution == "" and not resolutions and device_label == "":
        return []
    p: list[str] = []
    if not isinstance(backend, str) or not BACKEND_RE.fullmatch(backend):
        p.append(f"capture.camera_backend must match {BACKEND_RE.pattern}")
    if not isinstance(default_resolution, str) or not RESOLUTION_RE.fullmatch(default_resolution):
        p.append("capture.camera_default_resolution must be WIDTHxHEIGHT, e.g. 640x480")
    if (not isinstance(resolutions, (list, tuple)) or not 1 <= len(resolutions) <= MAX_RESOLUTIONS
            or not all(isinstance(r, str) and RESOLUTION_RE.fullmatch(r) for r in resolutions)
            or len(set(resolutions)) != len(resolutions)):
        p.append(f"capture.camera_resolutions must be 1..{MAX_RESOLUTIONS} distinct WIDTHxHEIGHT strings")
    if not label_ok(device_label):
        p.append(f"capture.camera_device_label must be printable text, 1..{MAX_LABEL} chars, no outer spaces")
    if p:
        p.append("capture.camera_*: set all four keys (measure them with `python -m acurix_edge camera-profile`) "
                 "or none")
    return p


def expected_profile(capture_config: Any) -> CameraProfile | None:
    """Expected profile from a validated CaptureConfig; None when not configured."""
    c = capture_config
    if not c.camera_backend:
        return None
    return CameraProfile(c.camera_backend, c.camera_default_resolution, sort_resolutions(c.camera_resolutions),
                         c.camera_device_label)


def profile_differences(expected: CameraProfile, observed: CameraProfile) -> list[str]:
    d: list[str] = []
    if observed.backend != expected.backend:
        d.append(f"backend {observed.backend} (expected {expected.backend})")
    if observed.default_resolution != expected.default_resolution:
        d.append(f"default resolution {observed.default_resolution} (expected {expected.default_resolution})")
    obs, exp = set(observed.supported_resolutions), set(expected.supported_resolutions)
    if obs != exp:
        extra, missing = sort_resolutions(obs - exp), sort_resolutions(exp - obs)
        d.append("supported resolutions differ"
                 + (f"; unexpected {', '.join(extra)}" if extra else "")
                 + (f"; missing {', '.join(missing)}" if missing else ""))
    if observed.device_label != expected.device_label:
        d.append(f"device label '{observed.device_label}' (expected '{expected.device_label}')")
    return d


@runtime_checkable
class ProfiledSource(Protocol):
    """A camera source that can measure its profile once open."""

    def measure_profile(self) -> CameraProfile: ...


def check_camera(source: object, expected: CameraProfile | None) -> CameraProfile | None:
    """Measure and compare the profile of an OPEN camera source. Sources that are not cameras
    (folder, video file) are not profiled: returns None. Raises CameraMismatch (AEX-CAP-001)."""
    if not isinstance(source, ProfiledSource):
        if getattr(source, "live", False):  # a live camera that cannot be profiled is refused (review B2)
            raise CameraMismatch(f"{MISMATCH_MESSAGE}: live source without camera profile support")
        return None
    if expected is None:
        raise CameraMismatch(f"{NOT_CONFIGURED_MESSAGE} ([capture] camera_*); measure the camera with "
                             "`python -m acurix_edge camera-profile` and store it in the configuration")
    observed = source.measure_profile()
    diffs = profile_differences(expected, observed)
    if diffs:
        raise CameraMismatch(f"{MISMATCH_MESSAGE}: " + "; ".join(diffs))
    return observed
