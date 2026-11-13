"""CLI.

  python -m acurix_edge init   [--config FILE] [--base-dir DIR]   create/migrate local database
  python -m acurix_edge status [--config FILE] [--base-dir DIR]   read-only health/status JSON
  python -m acurix_edge reset-fault --reason TEXT [--actor ID] [--config FILE] [--base-dir DIR]

Output is the OpenAPI `Status` document (plus extra fields: overall, resources, ...).
`status` opens the database read-only: it never creates, migrates or writes.
A FAULT is latched across restarts (R4-01); only `reset-fault` (audited) clears it.
Exit codes (status): 0 overall OK, 1 DEGRADED/UNKNOWN, 2 FAULT or configuration error.
Exit codes (init):   0 database OK, 2 database FAULT or configuration error.
Exit codes (reset-fault): 0 reset done (state IDLE), 2 refused/failed or configuration error.

Bench capture (CAP-005; writes only under <base-dir>/datasets/local/<session_id>/):
  python -m acurix_edge capture --source folder|video|webcam (--path P | --device N)
         --sku S --unit-id U --label L --lighting TEXT [--frames N] [...]
  python -m acurix_edge capture-verify --session-dir DIR
`--source webcam` turns the real camera on: only with the owner's authorization.
Exit codes (capture): 0 requested frames written (or finite source ended), 1 stopped early
(source lost / resolution changed; manifest still written), 2 refused (missing/invalid
parameters, existing session, source unavailable).
Exit codes (capture-verify): 0 manifest valid and every sha256 matches, 2 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from acurix_edge.app import Application, SessionError
from acurix_edge.config.settings import ConfigError, load_config
from acurix_edge.observability.logging import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="acurix-edge")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("status", "print read-only health/status snapshot as JSON"),
                            ("init", "create or migrate the local database"),
                            ("reset-fault", "audited release of a latched FAULT")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", type=Path, default=None)
        p.add_argument("--base-dir", type=Path, default=Path.cwd())
        if name == "reset-fault":
            p.add_argument("--reason", required=True)
            p.add_argument("--actor", default="operator")
    _add_capture_parsers(sub)
    args = parser.parse_args(argv)
    if args.command == "capture":
        return _capture(args)
    if args.command == "capture-verify":
        return _capture_verify(args)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(json.dumps({"error": "CONFIG_INVALID", "problems": exc.problems}), file=sys.stderr)
        return 2
    configure_logging(cfg.logging.level)
    app = Application(cfg, args.base_dir.resolve(), read_only=(args.command == "status"))
    app.boot()
    reset_error: str | None = None
    try:
        if args.command == "reset-fault":
            try:
                app.reset_fault(args.reason, actor_ref=args.actor)
            except (SessionError, ValueError) as exc:
                reset_error = str(exc)
        snapshot = app.status()
    finally:
        app.shutdown()
    print(json.dumps(snapshot, indent=2, ensure_ascii=False))
    if args.command == "reset-fault":
        if reset_error is not None:
            print(json.dumps({"error": "RESET_REFUSED", "detail": reset_error}), file=sys.stderr)
            return 2
        return 0 if snapshot["state"] == "IDLE" else 2
    components = {c["component"]: c["status"] for c in snapshot.get("components", [])}
    if args.command == "init":
        return 0 if components.get("database") == "OK" else 2
    overall = snapshot.get("overall")
    if snapshot["state"] == "FAULT" or overall == "FAULT":
        return 2
    return 0 if overall == "OK" else 1


def _add_capture_parsers(sub: Any) -> None:
    from acurix_edge.capture.manifest import PROVISIONAL_LABELS

    c = sub.add_parser("capture", help="bench capture: PNG frames + manifest in datasets/local/<session_id>/")
    c.add_argument("--source", required=True, choices=("folder", "video", "webcam"))
    c.add_argument("--path", type=Path, help="image folder or video file (folder/video sources)")
    c.add_argument("--device", type=int, help="webcam index (webcam source)")
    c.add_argument("--width", type=int, help="requested webcam width")
    c.add_argument("--height", type=int, help="requested webcam height")
    c.add_argument("--sku", required=True)
    c.add_argument("--unit-id", required=True)
    c.add_argument("--label", required=True, choices=PROVISIONAL_LABELS, help="provisional label (Anexo 04A §1)")
    c.add_argument("--lighting", required=True, help="lighting setup, e.g. luz-A-difusa-45")
    c.add_argument("--ambient", default="NOT_RECORDED", help="ambient light condition")
    c.add_argument("--frames", type=int, default=1)
    c.add_argument("--interval-s", type=float, default=0.0)
    c.add_argument("--session-id", default=None, help="default: cap-YYYY-MM-DD-NN (next free)")
    c.add_argument("--captured-by", default="owner")
    c.add_argument("--optics-version", default="NOT_RECORDED")
    c.add_argument("--exposure", default="NOT_RECORDED", help="recorded only; the tool does not set the camera")
    c.add_argument("--focus", default="NOT_RECORDED", help="recorded only; the tool does not set the camera")
    c.add_argument("--pose", default="")
    c.add_argument("--notes", default="")
    c.add_argument("--device-label", default=None,
                   help="camera.device in the manifest (default: source id, e.g. webcam-usb-0)")
    c.add_argument("--base-dir", type=Path, default=Path.cwd())
    v = sub.add_parser("capture-verify", help="validate a session manifest and re-hash its files")
    v.add_argument("--session-dir", type=Path, required=True)


def _capture(args: argparse.Namespace) -> int:
    from acurix_edge.capture.bench import BenchError, BenchParams, capture_session
    from acurix_edge.capture.sources import FrameSource, ImageFolderSource, VideoFileSource, WebcamSource

    source: FrameSource
    if args.source == "webcam":
        if args.device is None or args.path is not None:
            print(json.dumps({"error": "CAPTURE_REFUSED", "problems": ["webcam requires --device and no --path"]}),
                  file=sys.stderr)
            return 2
        source = WebcamSource(args.device, width=args.width, height=args.height)
    else:
        if args.path is None or args.device is not None:
            print(json.dumps({"error": "CAPTURE_REFUSED",
                              "problems": [f"{args.source} requires --path and no --device"]}), file=sys.stderr)
            return 2
        source = ImageFolderSource(args.path) if args.source == "folder" else VideoFileSource(args.path)
    params = BenchParams(sku=args.sku, unit_id=args.unit_id, provisional_label=args.label,
                         lighting_setup=args.lighting, lighting_ambient=args.ambient, captured_by=args.captured_by,
                         optics_version=args.optics_version, exposure=args.exposure, focus=args.focus,
                         pose=args.pose, notes=args.notes)
    configure_logging("WARNING")
    try:
        result = capture_session(source, args.base_dir.resolve() / "datasets" / "local", params,
                                 frames=args.frames, interval_s=args.interval_s, session_id=args.session_id,
                                 device_label=args.device_label)
    except BenchError as exc:
        print(json.dumps({"error": "CAPTURE_REFUSED", "problems": exc.problems}, ensure_ascii=False),
              file=sys.stderr)
        return 2
    print(json.dumps({"session_id": result.session_dir.name, "frames_written": result.frames_written,
                      "frames_requested": result.frames_requested, "end_reason": result.end_reason,
                      "detail": result.detail, "resolution": result.manifest["camera"]["resolution"]},
                     indent=2, ensure_ascii=False))
    return 0 if result.end_reason in ("COMPLETE", "END_OF_STREAM") else 1


def _capture_verify(args: argparse.Namespace) -> int:
    from acurix_edge.capture.manifest import ManifestError, verify_session

    try:
        data = verify_session(args.session_dir)
    except ManifestError as exc:
        print(json.dumps({"valid": False, "problems": exc.problems}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({"valid": True, "session_id": data["session_id"], "frames": len(data["frames"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
