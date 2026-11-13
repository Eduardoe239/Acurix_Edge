"""Session manifest, `manifest_schema` 1.0 — exactly Anexo 04A §3.5 (CAP-005).

    {"manifest_schema": "1.0", "session_id", "captured_by", "captured_at_utc", "optics_version",
     "camera": {"device", "resolution", "exposure", "focus"},
     "lighting": {"setup", "ambient"},
     "frames": [{"file", "sha256", "sku", "unit_id", "provisional_label", "pose", "notes"}]}

The schema below is the single definition: every object has EXACTLY these keys (unknown or
missing keys are errors), every value is a string. `validate_manifest` reports all problems at
once; `verify_session` also re-hashes every file and refuses files not listed in the manifest.
No OpenCV import here: verification works without the capture stack.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA = "1.0"
MANIFEST_FILE = "manifest.json"
# Classes of Anexo 04A §1 (provisional label = class at capture time; reviewed in annotation).
PROVISIONAL_LABELS = ("OK", "LABEL_MISSING", "LABEL_SKEWED", "LABEL_DAMAGED", "UNKNOWN")
NOT_RECORDED = "NOT_RECORDED"

ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
# A folder name: no trailing '.' (Windows silently strips it, review B7).
SESSION_ID_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,62}[A-Za-z0-9_-])?")
WINDOWS_RESERVED = frozenset({"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(0, 10)),
                              *(f"LPT{i}" for i in range(0, 10))})


def is_valid_session_id(value: Any) -> bool:
    return (isinstance(value, str) and SESSION_ID_RE.fullmatch(value) is not None
            and value.split(".")[0].upper() not in WINDOWS_RESERVED)
FILE_RE = re.compile(r"[0-9]{6}\.png")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
UTC_S_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")  # as in Anexo 04A §3.5
RESOLUTION_RE = re.compile(r"[1-9][0-9]{0,4}x[1-9][0-9]{0,4}")
MAX_TEXT = 200


def _text(value: Any) -> bool:
    """Free text: printable single line, at most MAX_TEXT chars (may be empty)."""
    return isinstance(value, str) and len(value) <= MAX_TEXT and value.isprintable()


def _required_text(value: Any) -> bool:
    return _text(value) and bool(value.strip())


def _regex(rx: re.Pattern[str]) -> Callable[[Any], bool]:
    return lambda v: isinstance(v, str) and rx.fullmatch(v) is not None


Rule = Callable[[Any], bool]
TOP: dict[str, Rule] = {
    "manifest_schema": lambda v: v == MANIFEST_SCHEMA,
    "session_id": is_valid_session_id,
    "captured_by": _required_text,
    "captured_at_utc": _regex(UTC_S_RE),
    "optics_version": _required_text,
}
CAMERA: dict[str, Rule] = {"device": _required_text, "resolution": _regex(RESOLUTION_RE),
                           "exposure": _required_text, "focus": _required_text}
LIGHTING: dict[str, Rule] = {"setup": _required_text, "ambient": _required_text}
FRAME: dict[str, Rule] = {"file": _regex(FILE_RE), "sha256": _regex(SHA256_RE), "sku": _regex(ID_RE),
                          "unit_id": _regex(ID_RE), "provisional_label": lambda v: v in PROVISIONAL_LABELS,
                          "pose": _text, "notes": _text}
TOP_KEYS = frozenset(TOP) | {"camera", "lighting", "frames"}


class ManifestError(ValueError):
    def __init__(self, problems: list[str]) -> None:
        super().__init__("invalid manifest: " + "; ".join(problems))
        self.problems = problems


def _check_object(obj: Any, rules: dict[str, Rule], where: str, problems: list[str],
                  extra_keys: frozenset[str] = frozenset()) -> None:
    if not isinstance(obj, dict):
        problems.append(f"{where} must be an object")
        return
    allowed = frozenset(rules) | extra_keys
    missing = sorted(allowed - set(obj))
    unknown = sorted(set(obj) - allowed)
    if missing:
        problems.append(f"{where}: missing keys {missing}")
    if unknown:
        problems.append(f"{where}: unknown keys {unknown}")
    for key, rule in rules.items():
        if key in obj and not rule(obj[key]):
            problems.append(f"{where}.{key}: invalid value {obj[key]!r}")


def validate_manifest(data: Any) -> None:
    """Raise ManifestError listing every problem; return None if the manifest is valid."""
    problems: list[str] = []
    _check_object(data, TOP, "manifest", problems, frozenset({"camera", "lighting", "frames"}))
    if isinstance(data, dict):
        _check_object(data.get("camera"), CAMERA, "camera", problems)
        _check_object(data.get("lighting"), LIGHTING, "lighting", problems)
        frames = data.get("frames")
        if not isinstance(frames, list) or not frames:
            problems.append("frames must be a non-empty list")
        else:
            seen: set[str] = set()
            for i, frame in enumerate(frames):
                _check_object(frame, FRAME, f"frames[{i}]", problems)
                name = frame.get("file") if isinstance(frame, dict) else None
                if isinstance(name, str):
                    if name in seen:
                        problems.append(f"frames[{i}].file duplicated: {name}")
                    seen.add(name)
    if problems:
        raise ManifestError(problems)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_session(session_dir: Path) -> dict[str, Any]:
    """Validate the manifest of a session folder and re-hash every file. Raises ManifestError."""
    path = Path(session_dir) / MANIFEST_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError([f"{MANIFEST_FILE} not found"]) from exc
    except (OSError, ValueError) as exc:
        raise ManifestError([f"{MANIFEST_FILE} unreadable: {type(exc).__name__}"]) from exc
    validate_manifest(data)
    problems: list[str] = []
    if data["session_id"] != Path(session_dir).name:
        problems.append(f"session_id {data['session_id']!r} differs from folder name {Path(session_dir).name!r}")
    listed = set()
    for frame in data["frames"]:
        listed.add(frame["file"])
        f = Path(session_dir) / frame["file"]
        if not f.is_file():
            problems.append(f"{frame['file']}: file missing")
        elif sha256_file(f) != frame["sha256"]:
            problems.append(f"{frame['file']}: sha256 mismatch")
    unlisted = sorted(p.name for p in Path(session_dir).iterdir() if p.name != MANIFEST_FILE and p.name not in listed)
    if unlisted:
        problems.append(f"files not in manifest: {unlisted}")
    if problems:
        raise ManifestError(problems)
    return data
