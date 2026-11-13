"""JSON structured logs without secrets.

Each line: ts (UTC), level, component, msg, code, session_id, correlation_id,
ctx. Context values whose keys look sensitive are redacted recursively; bearer
tokens inside strings are masked. Images/payloads must never be logged.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import re
import sys
import traceback
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any, TextIO

REDACTED = "***REDACTED***"
# Key-name match. Note: bare 'pass' is NOT sensitive (PASS counters must stay visible).
_SENSITIVE_KEY = re.compile(
    r"(?i)(password|passwd|passphrase|(^|[_-])pwd($|[_-])|secret|token|api[_-]?key|authorization|"
    r"(^|[_-])auth($|[_-])|credential|private[_-]?key|cookie|session[_-]?key|(^|[_-])dsn($|[_-])|^key$)"
)
# R5-05: rules with a variable-length prefix may only START at the beginning of a token (negative
# lookbehind over the prefix's own character class). Otherwise every position inside a long token
# is a new start that rescans to the token end -> O(n^2) (measured 4-6 s per 16 KiB line). Anchored,
# each token is scanned from one start only -> linear. The prefix group still covers the same keys.
_STRING_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]+"), r"\1 " + "***REDACTED***"),
    # Scheme = the whole maximal run before '://' (superset of the old \b[a-z]... start: over-redacts,
    # never under-redacts).
    (re.compile(r"(?i)(?<![a-z0-9+.-])([a-z0-9+.-]+://)[^/\s:@]+:[^/\s@]+@"), r"\1***REDACTED***@"),
    # key=value / key: value / "key": "value" / 'key': 'v' — key may carry a prefix (client_secret,
    # access_token, db_password); quoted values may contain escaped quotes.
    (re.compile(
        r"""(?i)(?<![A-Za-z0-9_-])(["']?[A-Za-z0-9_-]*(?:password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key)["']?)"""
        r"""(\s*[=:]\s*)("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|[^\s&,;"'}\]]+)"""),
     r"\1\2***REDACTED***"),
    # JWT (R5-05/F1): equivalent to \beyJ[..]{5,}\.[..]{5,}\.[..]{5,}\b (same spans, fuzz-checked) but
    # one start per run: the lookahead rejects runs not followed by '.', the atomic group commits to
    # the leftmost 'eyJ' (later ones in the same run share the same tail, so they cannot succeed).
    (re.compile(r"(?<![A-Za-z0-9_-])(?=[A-Za-z0-9_-]*\.)(?>([A-Za-z0-9_-]*?)\beyJ[A-Za-z0-9_-]{5,}\.)"
                r"[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\b"), r"\1***REDACTED_JWT***"),
)
_MAX_DEPTH = 6
# R4-10: strings longer than this are OMITTED, never truncated. Truncation could cut a secret so
# that no redaction rule matches it any more (partial JWT, URL credentials without '@', ...) and
# would leak the remaining prefix. Omission also bounds regex cost per log line.
MAX_REDACT_CHARS = 16_384
MAX_ITEMS = 256  # containers: items beyond this are dropped and counted


class RedactionStats:
    """Collects structured evidence of omissions for one log entry."""

    __slots__ = ("omitted_strings", "omitted_chars", "dropped_items", "depth_truncations")

    def __init__(self) -> None:
        self.omitted_strings = 0
        self.omitted_chars = 0
        self.dropped_items = 0
        self.depth_truncations = 0

    def to_dict(self) -> dict[str, int] | None:
        if not (self.omitted_strings or self.dropped_items or self.depth_truncations):
            return None
        return {"omitted_strings": self.omitted_strings, "omitted_chars": self.omitted_chars,
                "dropped_items": self.dropped_items, "depth_truncations": self.depth_truncations,
                "max_chars": MAX_REDACT_CHARS, "max_items": MAX_ITEMS}


_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("acurix_session_id", default=None)
_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("acurix_correlation_id", default=None)


def redact(value: Any, _depth: int = 0, stats: RedactionStats | None = None) -> Any:
    if _depth > _MAX_DEPTH:
        if stats is not None:
            stats.depth_truncations += 1
        return "<truncated>"
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for i, (k, v) in enumerate(value.items()):
            if i >= MAX_ITEMS:
                if stats is not None:
                    stats.dropped_items += len(value) - MAX_ITEMS
                out["<omitted>"] = f"{len(value) - MAX_ITEMS} more entries"
                break
            key = redact(str(k), _depth + 1, stats)
            out[key] = REDACTED if _SENSITIVE_KEY.search(str(k)) else redact(v, _depth + 1, stats)
        return out
    if isinstance(value, (list, tuple)):
        items = [redact(v, _depth + 1, stats) for v in value[:MAX_ITEMS]]
        if len(value) > MAX_ITEMS:
            if stats is not None:
                stats.dropped_items += len(value) - MAX_ITEMS
            items.append(f"<omitted {len(value) - MAX_ITEMS} more items>")
        return items
    if isinstance(value, str):
        if len(value) > MAX_REDACT_CHARS:
            if stats is not None:
                stats.omitted_strings += 1
                stats.omitted_chars += len(value)
            return f"<omitted: {len(value)} chars exceed redaction limit {MAX_REDACT_CHARS}>"
        for pattern, repl in _STRING_RULES:
            value = pattern.sub(repl, value)
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<binary {len(value)} bytes omitted>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact(str(value), _depth + 1, stats)


@contextlib.contextmanager
def bind_context(session_id: str | None = None, correlation_id: str | None = None) -> Iterator[None]:
    tokens = []
    if session_id is not None:
        tokens.append((_session_id, _session_id.set(session_id)))
    if correlation_id is not None:
        tokens.append((_correlation_id, _correlation_id.set(correlation_id)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        stats = RedactionStats()
        try:
            message = record.getMessage()
        except Exception as exc:  # malformed %-args must not drop the log line
            message = f"<unformattable log message: {type(exc).__name__}>"
        entry: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "level": record.levelname,
            "component": record.name,
            "msg": redact(message, stats=stats),
            "code": getattr(record, "code", None),
            "session_id": redact(_session_id.get(), stats=stats),
            "correlation_id": redact(_correlation_id.get(), stats=stats),
        }
        ctx = getattr(record, "ctx", None)
        if ctx is not None:
            entry["ctx"] = redact(ctx, stats=stats)
        if record.exc_info:
            exc_type, exc_value, _tb = record.exc_info
            entry["exc"] = {
                "type": exc_type.__name__ if exc_type else None,
                "message": redact(str(exc_value), stats=stats),
                "trace": redact("".join(traceback.format_exception(*record.exc_info)), stats=stats),
            }
        omitted = stats.to_dict()
        if omitted is not None:
            entry["redaction"] = omitted  # structured evidence that content was omitted (R4-10)
        return json.dumps(entry, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO", stream: TextIO | None = None) -> logging.Logger:
    root = logging.getLogger("acurix_edge")
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.propagate = False
    return root
