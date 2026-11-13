"""Bench capture tool (CAP-005): lossless PNG frames + manifest (Anexo 04A §3.5) in
`datasets/local/<session_id>/`.

Rules:
- SKU, unit_id, provisional label and lighting condition are mandatory; nothing runs without them;
- output only under a folder named `datasets/local` (git-ignored); never anywhere else;
- a session folder is created exclusively: an existing session is NEVER overwritten (files are
  also opened in exclusive-create mode);
- the source is opened BEFORE the folder is created: an unavailable source leaves nothing behind;
- the manifest is validated against the schema before it is written; sha256 is computed from the
  exact bytes written;
- the session resolution is the first frame's; a frame with another size stops the capture
  (`RESOLUTION_CHANGED`) instead of mixing resolutions under one manifest;
- a source that ends early (folder/video) stops with END_OF_STREAM (not an error); a lost source
  stops with SOURCE_LOST. Frames already written are kept and listed in the manifest — also when
  the run is interrupted (Ctrl+C, disk error): the manifest is written, then the error re-raised.
- each PNG is written to `.tmp` and renamed: no partial frame under a final name.
Known limit: a webcam read that blocks inside the driver has no timeout in this tool.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from acurix_edge.capture.frames import InvalidImageError, check_image
from acurix_edge.capture.manifest import (
    ID_RE,
    MANIFEST_FILE,
    MANIFEST_SCHEMA,
    NOT_RECORDED,
    PROVISIONAL_LABELS,
    SESSION_ID_RE,
    ManifestError,
    _required_text,
    _text,
    is_valid_session_id,
    validate_manifest,
)
from acurix_edge.capture.sources import FrameSource, SourceError

log = logging.getLogger("acurix_edge.capture.bench")

MAX_FRAMES = 10_000
DATASET_ROOT_PARTS = ("datasets", "local")

COMPLETE = "COMPLETE"
END_OF_STREAM = "END_OF_STREAM"
SOURCE_LOST = "SOURCE_LOST"
RESOLUTION_CHANGED = "RESOLUTION_CHANGED"
INTERRUPTED = "INTERRUPTED"


class BenchError(ValueError):
    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


@dataclass(frozen=True, slots=True)
class BenchParams:
    sku: str
    unit_id: str
    provisional_label: str
    lighting_setup: str
    lighting_ambient: str = NOT_RECORDED
    captured_by: str = "owner"
    optics_version: str = NOT_RECORDED
    exposure: str = NOT_RECORDED
    focus: str = NOT_RECORDED
    pose: str = ""
    notes: str = ""

    def problems(self) -> list[str]:
        p: list[str] = []
        for name in ("sku", "unit_id"):
            v = getattr(self, name)
            if not isinstance(v, str) or not ID_RE.fullmatch(v):
                p.append(f"{name} is mandatory and must match {ID_RE.pattern}")
        if self.provisional_label not in PROVISIONAL_LABELS:
            p.append(f"provisional_label is mandatory, one of {list(PROVISIONAL_LABELS)}")
        if not _required_text(self.lighting_setup):
            p.append("lighting (setup) is mandatory: printable text, at most 200 chars")
        for name in ("lighting_ambient", "captured_by", "optics_version", "exposure", "focus"):
            if not _required_text(getattr(self, name)):
                p.append(f"{name} must be non-empty printable text, at most 200 chars")
        for name in ("pose", "notes"):
            if not _text(getattr(self, name)):
                p.append(f"{name} must be printable text, at most 200 chars")
        return p


@dataclass(frozen=True, slots=True)
class BenchResult:
    session_dir: Path
    frames_written: int
    frames_requested: int
    end_reason: str
    detail: str
    manifest: dict[str, Any]

    @property
    def complete(self) -> bool:
        return self.end_reason == COMPLETE


def check_output_root(output_root: Path) -> Path:
    root = Path(output_root).resolve()
    if tuple(p.lower() for p in root.parts[-2:]) != DATASET_ROOT_PARTS:
        raise BenchError([f"output root must be a folder named {'/'.join(DATASET_ROOT_PARTS)} (git-ignored)"])
    return root


def _create_session_dir(root: Path, session_id: str | None, now: datetime) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if session_id is not None:
        if not is_valid_session_id(session_id):
            raise BenchError([f"session_id must match {SESSION_ID_RE.pattern} and not be a reserved name"])
        target = root / session_id
        try:
            target.mkdir()  # exclusive: fails if it exists
        except FileExistsError:
            raise BenchError([f"session {session_id} already exists; sessions are never overwritten"]) from None
        return target
    prefix = f"cap-{now:%Y-%m-%d}-"
    for n in range(1, 100):
        target = root / f"{prefix}{n:02d}"
        try:
            target.mkdir()
            return target
        except FileExistsError:
            continue
    raise BenchError([f"no free session id for {now:%Y-%m-%d} (01..99 used)"])


def _write_exclusive(path: Path, data: bytes) -> None:
    """Write to `<name>.tmp` then rename: a crash never leaves a partial file under the final
    name. Both steps refuse to replace an existing file (exclusive create / rename on Windows)."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "xb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    if path.exists():
        raise FileExistsError(path.name)
    os.rename(tmp, path)


def capture_session(source: FrameSource, output_root: Path, params: BenchParams, *, frames: int,
                    interval_s: float = 0.0, session_id: str | None = None,
                    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                    device_label: str | None = None) -> BenchResult:
    problems = params.problems()
    if isinstance(frames, bool) or not isinstance(frames, int) or not 1 <= frames <= MAX_FRAMES:
        problems.append(f"frames must be int in 1..{MAX_FRAMES}")
    if isinstance(interval_s, bool) or not isinstance(interval_s, (int, float)) or not 0 <= interval_s <= 3600:
        problems.append("interval_s must be a number in 0..3600")
    device = device_label or source.source_id
    if not _required_text(device):
        problems.append("camera device label must be non-empty printable text, at most 200 chars")
    if problems:
        raise BenchError(problems)
    root = check_output_root(output_root)
    started = now()
    try:
        source.open()
    except SourceError as exc:
        raise BenchError([f"source unavailable (AEX-CAP-001): {exc}"]) from exc
    session_dir: Path | None = None
    entries: list[dict[str, str]] = []
    resolution: str | None = None
    end_reason, detail = COMPLETE, ""
    interrupted: BaseException | None = None
    try:
        session_dir = _create_session_dir(root, session_id, started)
    except BaseException:
        source.close()
        raise
    try:
        while len(entries) < frames:
            try:
                image = source.read()
            except SourceError as exc:
                end_reason, detail = SOURCE_LOST, str(exc)
                break
            if image is None:
                end_reason, detail = END_OF_STREAM, "source ended before the requested number of frames"
                break
            try:
                img = check_image(image)
            except InvalidImageError as exc:
                end_reason, detail = SOURCE_LOST, f"invalid frame: {exc}"
                break
            h, w = img.shape[:2]
            if resolution is None:
                resolution = f"{w}x{h}"
            elif resolution != f"{w}x{h}":
                end_reason, detail = RESOLUTION_CHANGED, f"frame {w}x{h} differs from session {resolution}"
                break
            ok, encoded = cv2.imencode(".png", img)
            if not ok:
                end_reason, detail = SOURCE_LOST, "PNG encoding failed"
                break
            data = encoded.tobytes()
            name = f"{len(entries) + 1:06d}.png"
            _write_exclusive(session_dir / name, data)
            entries.append({"file": name, "sha256": hashlib.sha256(data).hexdigest(), "sku": params.sku,
                            "unit_id": params.unit_id, "provisional_label": params.provisional_label,
                            "pose": params.pose, "notes": params.notes})
            if interval_s and len(entries) < frames:
                time.sleep(interval_s)
    except BaseException as exc:  # Ctrl+C, disk full, ...: list what was written, then re-raise (M5)
        interrupted = exc
        end_reason, detail = INTERRUPTED, type(exc).__name__
        listed = {e["file"] for e in entries}
        for stray in session_dir.iterdir():
            # a half-written .tmp, or a PNG renamed just before the interruption but not yet listed
            # (review L1): nothing unlisted stays in the session
            if stray.name not in listed:
                try:
                    stray.unlink()
                except OSError:
                    pass
    finally:
        source.close()
    if not entries:
        try:
            session_dir.rmdir()  # empty: no frame was written
        except OSError:
            pass
        if interrupted is not None:
            raise interrupted
        raise BenchError([f"no frame captured ({end_reason}: {detail})"])
    manifest: dict[str, Any] = {
        "manifest_schema": MANIFEST_SCHEMA,
        "session_id": session_dir.name,
        "captured_by": params.captured_by,
        "captured_at_utc": started.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "optics_version": params.optics_version,
        "camera": {"device": device, "resolution": resolution,
                   "exposure": params.exposure, "focus": params.focus},
        "lighting": {"setup": params.lighting_setup, "ambient": params.lighting_ambient},
        "frames": entries,
    }
    try:
        validate_manifest(manifest)
    except ManifestError as exc:  # defensive: inputs were validated; never write an invalid manifest
        raise BenchError([f"manifest would be invalid: {p}" for p in exc.problems]) from exc
    payload = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if interrupted is None:
        _write_exclusive(session_dir / MANIFEST_FILE, payload)
    else:
        try:
            _write_exclusive(session_dir / MANIFEST_FILE, payload)
        except OSError:  # e.g. disk full: keep the ORIGINAL error, drop a partial .tmp (review L6)
            log.error("manifest not written after interruption", extra={"code": "BENCH_MANIFEST_WRITE_FAILED"})
            try:
                (session_dir / (MANIFEST_FILE + ".tmp")).unlink(missing_ok=True)
            except OSError:
                pass
        raise interrupted
    return BenchResult(session_dir, len(entries), frames, end_reason, detail, manifest)
