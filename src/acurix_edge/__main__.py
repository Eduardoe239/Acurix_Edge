"""CLI.

  python -m acurix_edge init   [--config FILE] [--base-dir DIR]   create/migrate local database
  python -m acurix_edge status [--config FILE] [--base-dir DIR]   read-only health/status JSON
  python -m acurix_edge reset-fault --reason TEXT [--actor ID] [--config FILE] [--base-dir DIR]

`status` opens the database read-only: it never creates, migrates or writes.
A FAULT is latched across restarts (R4-01); only `reset-fault` (audited) clears it.
Exit codes (status): 0 overall OK, 1 DEGRADED/UNKNOWN, 2 FAULT or configuration error.
Exit codes (init):   0 database OK, 2 database FAULT or configuration error.
Exit codes (reset-fault): 0 reset done (state IDLE), 2 refused/failed or configuration error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    args = parser.parse_args(argv)

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
    components = {c["name"]: c["status"] for c in (snapshot.get("health") or {}).get("components", [])}
    if args.command == "init":
        return 0 if components.get("database") == "OK" else 2
    overall = (snapshot.get("health") or {}).get("overall")
    if snapshot["state"] == "FAULT" or overall == "FAULT":
        return 2
    return 0 if overall == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
