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
  python -m acurix_edge camera-profile --device N --device-label L
`--source webcam` and `camera-profile` turn the real camera on: only with the owner's authorization.
`--source webcam` requires `--config FILE` holding the expected camera profile ([capture] camera_*,
printed by `camera-profile`); a different camera is refused (AEX-CAP-001) before anything is written.
Exit codes (capture): 0 requested frames written (or finite source ended), 1 stopped early
(source lost / resolution changed; manifest still written), 2 refused (missing/invalid
parameters, existing session, source unavailable, camera profile mismatch).
Exit codes (capture-verify): 0 manifest valid and every sha256 matches, 2 otherwise.
Exit codes (camera-profile): 0 profile measured, 2 camera unavailable or invalid parameters.

Bench viewer (AEX-BENCH-VIEW-001, DEVELOPMENT TOOL, not the 0.5.0 console):
  python -m acurix_edge bench-view --config FILE [--device N] [--port P] [--fps F] [--stream-width W]
  python -m acurix_edge bench-view --demo-sintetico [--fps F] [--port P] [--config FILE] [--base-dir DIR]
--demo-sintetico: SIMULATED source (image generated in code), never opens a camera, needs no camera
profile, page shows a fixed "FONTE SIMULADA" banner, events record input_source.simulated=true;
data in a temporary folder deleted at exit unless --base-dir is given.
Listens ONLY on 127.0.0.1; never writes images; never shows PASS/DEFECT. Stop: page button or Ctrl+C.
Exit codes (bench-view): 0 stopped cleanly, 2 refused (configuration, camera profile missing, port busy).
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
    if args.command == "camera-profile":
        return _camera_profile(args)
    if args.command == "bench-view":
        return _bench_view(args)

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
                   help="camera.device in the manifest. Webcam: also part of the camera profile, so it must "
                        "equal camera_device_label in --config (default: that configured label)")
    c.add_argument("--base-dir", type=Path, default=Path.cwd())
    c.add_argument("--config", type=Path, default=None,
                   help="configuration with the expected camera profile (required for --source webcam)")
    v = sub.add_parser("capture-verify", help="validate a session manifest and re-hash its files")
    v.add_argument("--session-dir", type=Path, required=True)
    p = sub.add_parser("camera-profile", help="measure the camera profile to store as the expected one")
    p.add_argument("--device", type=int, required=True, help="webcam index")
    p.add_argument("--device-label", required=True, help="real camera name, stored in the profile")
    b = sub.add_parser("bench-view", help="DEVELOPMENT TOOL: local live page (127.0.0.1) of the real engine")
    b.add_argument("--config", type=Path, default=None,
                   help="configuration with the expected camera profile (optional with --demo-sintetico)")
    b.add_argument("--demo-sintetico", action="store_true",
                   help="SIMULATED source: image generated in code; never opens a camera; no camera profile; "
                        "data in a temporary folder deleted at exit (unless --base-dir)")
    b.add_argument("--device", type=int, default=None, help="webcam index (default 0; not with --demo-sintetico)")
    b.add_argument("--port", type=int, default=None, help="local port (default: api.port of the config, 8765)")
    b.add_argument("--fps", type=float, default=8.0, help="processed/displayed frames per second (1..15)")
    b.add_argument("--stream-width", type=int, default=640, help="width of the live image (160..1280 px)")
    b.add_argument("--base-dir", type=Path, default=None,
                   help="data folder (default: current folder; with --demo-sintetico: a temporary folder)")


def _capture(args: argparse.Namespace) -> int:
    from acurix_edge.capture.bench import BenchError, BenchParams, capture_session
    from acurix_edge.capture.profile import CameraProfile, expected_profile
    from acurix_edge.capture.sources import FrameSource, ImageFolderSource, VideoFileSource, WebcamSource
    from acurix_edge.config.settings import CaptureConfig

    source: FrameSource
    expected: CameraProfile | None = None
    device_label: str | None = args.device_label
    open_timeout_s = CaptureConfig().open_timeout_s
    if args.source == "webcam":
        if args.device is None or args.path is not None:
            print(json.dumps({"error": "CAPTURE_REFUSED", "problems": ["webcam requires --device and no --path"]}),
                  file=sys.stderr)
            return 2
        if args.config is None:
            print(json.dumps({"error": "CAPTURE_REFUSED", "problems": [
                "webcam requires --config with the expected camera profile ([capture] camera_*); "
                "measure it with `python -m acurix_edge camera-profile`"]}, ensure_ascii=False), file=sys.stderr)
            return 2
        try:
            capture_cfg = load_config(args.config).capture
            expected, open_timeout_s = expected_profile(capture_cfg), capture_cfg.open_timeout_s
        except ConfigError as exc:
            print(json.dumps({"error": "CONFIG_INVALID", "problems": exc.problems}, ensure_ascii=False),
                  file=sys.stderr)
            return 2
        if device_label is None and expected is not None:
            device_label = expected.device_label  # the manifest names the configured camera
        source = WebcamSource(args.device, width=args.width, height=args.height, device_label=device_label)
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
                                 device_label=device_label, expected_profile=expected,
                                 open_timeout_s=open_timeout_s)
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


def _camera_profile(args: argparse.Namespace) -> int:
    """Open the camera, measure its profile, close it. Frames are only measured (never written)."""
    from acurix_edge.capture.profile import label_ok
    from acurix_edge.capture.sources import SourceError, WebcamSource, open_with_timeout
    from acurix_edge.config.settings import CaptureConfig

    problems = []
    if args.device < 0:
        problems.append("--device must be >= 0")
    if not label_ok(args.device_label):
        problems.append("--device-label must be printable text, 1..200 chars, no outer spaces")
    if problems:
        print(json.dumps({"error": "PROFILE_REFUSED", "problems": problems}, ensure_ascii=False), file=sys.stderr)
        return 2
    source = WebcamSource(args.device, device_label=args.device_label)
    try:
        open_with_timeout(source, CaptureConfig().open_timeout_s)
        profile = source.measure_profile()
    except Exception as exc:  # SourceError or a backend error: always a structured AEX-CAP-001 (B4)
        detail = str(exc) if isinstance(exc, SourceError) else f"profile failed ({type(exc).__name__})"
        print(json.dumps({"error": "CAMERA_UNAVAILABLE", "code": "AEX-CAP-001", "detail": detail},
                         ensure_ascii=False), file=sys.stderr)
        return 2
    finally:
        source.close()
    print(json.dumps({"profile": profile.to_dict(), "config_toml": profile.to_toml()}, indent=2,
                     ensure_ascii=False))
    return 0


def _bench_view(args: argparse.Namespace) -> int:
    """Bench viewer (AEX-BENCH-VIEW-001). Turns the real camera on: only with the owner's authorization."""
    import shutil
    import tempfile

    from acurix_edge.benchview.app_factory import BenchViewError, build_bench_view

    configure_logging("WARNING")
    demo = bool(args.demo_sintetico)
    if demo and args.device is not None:
        print(json.dumps({"error": "BENCH_VIEW_REFUSED", "problems": [
            "--device não se aplica ao --demo-sintetico (nenhuma câmera é aberta)"]}, ensure_ascii=False),
            file=sys.stderr)
        return 2
    temp_dir: Path | None = None
    if args.base_dir is not None:
        base_dir = args.base_dir.resolve()
    elif demo:  # disposable by default (owner, 19/11/2026): never mixed with a real station's data
        base_dir = temp_dir = Path(tempfile.mkdtemp(prefix="acurix-demo-sintetico-"))
    else:
        base_dir = Path.cwd().resolve()
    try:
        server = build_bench_view(config_path=args.config, base_dir=base_dir,
                                  device=0 if args.device is None else args.device, port=args.port, fps=args.fps,
                                  stream_width=args.stream_width, synthetic=demo)
    except BenchViewError as exc:
        print(json.dumps({"error": "BENCH_VIEW_REFUSED", "problems": exc.problems}, ensure_ascii=False),
              file=sys.stderr)
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return 2
    except BaseException:  # unexpected failure or Ctrl+C while building: the disposable folder goes too
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    try:  # start() opens the camera: Ctrl+C during it must still release everything (review B1)
        started = server.engine.start()
        info: dict[str, object] = {"url": server.url, "session_started": started, "problem": server.engine.problem,
                                   "stop": "botão Encerrar na página ou Ctrl+C", "data_dir": str(base_dir)}
        if demo:
            info["source"] = "FONTE SIMULADA — imagem gerada por código, não é câmera real"
            info["data_dir_disposable"] = temp_dir is not None
        print(json.dumps(info, ensure_ascii=False), flush=True)
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.engine.stop()  # ends the session and releases the camera (idempotent)
        server.server_close()
    removed = None
    if temp_dir is not None:
        shutil.rmtree(temp_dir, ignore_errors=True)
        removed = not temp_dir.exists()
    print(json.dumps({"stopped": True, "frames_processed": server.engine.frames_processed,
                      "disposable_data_removed": removed}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
