"""Local HTTP server of the bench viewer (standard library only: http.server).

Security posture (no authentication, so LOCAL ONLY):
- bound to 127.0.0.1 — a constant, not configurable; exclusive port binding on Windows;
- every request must carry Host 127.0.0.1:<port> or localhost:<port> (DNS-rebinding defence);
- closed route list (no file-system paths are ever built from the URL);
- GET only, except POST /api/shutdown, which also requires the header `X-Acurix-Action: shutdown`
  (a cross-site page cannot send it without a CORS preflight, which is never granted);
- a browser request from ANOTHER site (Sec-Fetch-Site cross-site/same-site) is refused, so no
  page can embed the live image (<img src=...>) nor take the stream slots; CORP/COOP same-origin;
- strict Content-Security-Policy, nosniff, no-referrer, no-store, frame-ancestors 'none';
- at most MAX_STREAMS concurrent MJPEG streams and a per-connection timeout (bounded CPU/RAM);
- methods other than GET/POST: 405 with the same security headers.
"""

from __future__ import annotations

import html
import json
import logging
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any

from acurix_edge.benchview.engine import BenchViewEngine
from acurix_edge.benchview.view_model import PRIVACY_NOTICE, PRIVACY_NOTICE_SIMULATED, SIMULATED_NOTICE

log = logging.getLogger("acurix_edge.benchview")

HOST = "127.0.0.1"  # never 0.0.0.0 (Ordem 5)
MAX_STREAMS = 2
CONNECTION_TIMEOUT_S = 10.0
ALLOWED_FETCH_SITES = frozenset({"same-origin", "none"})  # Sec-Fetch-Site: this page, or a typed URL
BOUNDARY = "acurixframe"
SHUTDOWN_HEADER = ("X-Acurix-Action", "shutdown")
CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; font-src 'self'; "
       "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
STATIC: dict[str, tuple[str, str]] = {  # URL path -> (file in static/, content type)
    "/": ("index.html", "text/html; charset=utf-8"),
    "/static/tokens.css": ("tokens.css", "text/css; charset=utf-8"),
    "/static/bundle.css": ("bundle.css", "text/css; charset=utf-8"),
    "/static/viewer.css": ("viewer.css", "text/css; charset=utf-8"),
    "/static/viewer.js": ("viewer.js", "text/javascript; charset=utf-8"),
    "/static/Inter-Variable-latin.woff2": ("Inter-Variable-latin.woff2", "font/woff2"),
}


BANNER_MARK = b"<!--SIMULATED-SOURCE-BANNER-->"
TAG_MARK = b"<!--SIMULATED-SOURCE-TAG-->"
PRIVACY_MARK = b"<!--PRIVACY-NOTICE-->"


def load_static(simulated: bool = False) -> dict[str, tuple[bytes, str]]:
    """Static files. With a SIMULATED source the page itself carries a fixed banner at the top and a
    tag on the image, written here into the HTML (never toggled by JavaScript, cannot be hidden)."""
    root = resources.files("acurix_edge.benchview").joinpath("static")
    files = {path: (root.joinpath(name).read_bytes(), ctype) for path, (name, ctype) in STATIC.items()}
    page, ctype = files["/"]
    if BANNER_MARK not in page or TAG_MARK not in page or PRIVACY_MARK not in page:
        raise RuntimeError("index.html lost the simulated-source / privacy markers")
    privacy = html.escape(PRIVACY_NOTICE_SIMULATED if simulated else PRIVACY_NOTICE).encode("utf-8")
    page = page.replace(PRIVACY_MARK, b'<p class="ax-card bv-privacy" role="note">' + privacy + b"</p>")
    notice = html.escape(SIMULATED_NOTICE).encode("utf-8")
    if simulated:
        page = page.replace(BANNER_MARK, b'<div class="ax-banner ax-banner--p2 bv-simulated" role="note">'
                            + notice + b"</div>")
        page = page.replace(TAG_MARK, b'<div class="bv-sim-tag" aria-hidden="true">FONTE SIMULADA</div>')
    else:
        page = page.replace(BANNER_MARK, b"").replace(TAG_MARK, b"")
    files["/"] = (page, ctype)
    return files


class BenchViewServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False  # never share the port with another process

    def __init__(self, engine: BenchViewEngine, port: int) -> None:
        self.engine = engine
        self.static = load_static(simulated=engine.simulated)
        self.streams = threading.BoundedSemaphore(MAX_STREAMS)
        self._shutting_down = threading.Event()
        super().__init__((HOST, port), _Handler)

    def server_bind(self) -> None:
        exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)  # Windows: no port hijacking
        if exclusive is not None:
            self.socket.setsockopt(socket.SOL_SOCKET, exclusive, 1)
        super().server_bind()

    @property
    def port(self) -> int:
        return int(self.server_address[1])

    @property
    def url(self) -> str:
        return f"http://{HOST}:{self.port}/"

    def allowed_hosts(self) -> frozenset[str]:
        return frozenset({f"{HOST}:{self.port}", f"localhost:{self.port}"})

    def request_shutdown(self) -> None:
        """Stop the engine (camera released) then the server. Safe from any thread, idempotent."""
        if self._shutting_down.is_set():
            return
        self._shutting_down.set()

        def run() -> None:
            self.engine.stop()
            self.shutdown()

        threading.Thread(target=run, name="acurix-benchview-shutdown", daemon=True).start()


class _Handler(BaseHTTPRequestHandler):
    server: BenchViewServer
    server_version = "AcurixBenchView"
    sys_version = ""
    timeout = CONNECTION_TIMEOUT_S  # a stalled client never holds a thread/stream slot forever (B5)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 (stdlib signature)
        log.debug("bench viewer http: " + format, *args)

    # -- helpers ----------------------------------------------------------------------------
    def _common_headers(self) -> None:
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self._common_headers()
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, payload: Any) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _host_ok(self) -> bool:
        """Host must be this server (DNS rebinding) and, when the browser says where the request
        comes from, it must be this page or a typed URL — never another site (<img> embedding)."""
        if self.headers.get("Host", "") not in self.server.allowed_hosts():
            self._json(403, {"error": "forbidden host"})
            return False
        site = self.headers.get("Sec-Fetch-Site")
        if site is not None and site not in ALLOWED_FETCH_SITES:
            self._json(403, {"error": "cross-site request refused"})
            return False
        return True

    def _not_allowed(self) -> None:
        self.send_response(405)
        self.send_header("Allow", "GET, POST")
        self.send_header("Content-Length", "0")
        self._common_headers()
        self.end_headers()

    do_HEAD = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _not_allowed  # noqa: N815 (stdlib names)

    # -- routes -----------------------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 (stdlib name)
        if not self._host_ok():
            return
        path = self.path.split("?", 1)[0]
        if path in self.server.static:
            body, ctype = self.server.static[path]
            self._send(200, body, ctype)
        elif path == "/api/status":
            self._json(200, self.server.engine.snapshot())
        elif path == "/stream.mjpg":
            self._stream()
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 (stdlib name)
        if not self._host_ok():
            return
        name, value = SHUTDOWN_HEADER
        if self.path != "/api/shutdown":
            self._json(404, {"error": "not found"})
        elif self.headers.get(name) != value:
            self._json(403, {"error": f"missing {name} header"})
        else:
            self._json(202, {"stopping": True})
            self.server.request_shutdown()

    def _stream(self) -> None:
        if not self.server.streams.acquire(blocking=False):
            self._json(503, {"error": f"at most {MAX_STREAMS} live streams"})
            return
        try:
            self.send_response(200)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
            self._common_headers()
            self.end_headers()
            seq = 0
            engine = self.server.engine
            while not engine.stopped:
                got = engine.wait_frame(seq, timeout=1.0)
                if got is None:
                    continue
                seq, jpeg = got
                self.wfile.write(f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n"
                                 .encode("ascii") + jpeg + b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):
            pass  # the browser closed (or stopped reading) the stream
        finally:
            self.server.streams.release()
