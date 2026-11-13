"""Regression tests for the fifth independent review (R5-05, I-01) and its listed test gaps."""

import contextlib
import io
import json
import logging
import random
import re
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from acurix_edge.__main__ import main
from acurix_edge.app import SessionError
from acurix_edge.domain.state_machine import SystemState
from acurix_edge.health.service import ComponentStatus
from acurix_edge.observability import logging as obs
from acurix_edge.observability.logging import MAX_REDACT_CHARS, redact
from tests.unit.test_r4_regressions import AppCase

# -- R5-05: redaction must be linear in the input length ---------------------------------
# Pre-fix patterns (verbatim), used only as a coverage oracle: the fix must never redact less.
_OLD_URL = re.compile(r"(?i)(\b[a-z][a-z0-9+.-]*://)[^/\s:@]+:[^/\s@]+@")
_OLD_KV = re.compile(
    r"""(?i)(?<![A-Za-z0-9])(["']?[A-Za-z0-9_-]*(?:password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key)["']?)"""
    r"""(\s*[=:]\s*)("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|[^\s&,;"'}\]]+)""")
_OLD_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\b")

# Adversarial inputs measured at 0.06-10.8 s per 16 KiB string before the fix.
_ADVERSARIAL = ("a-", "_a", "a.", "a+", "x-pwd", "a-token", "x_http", "eyJ-", "eyJa-", "eyJ_", "eyJ-a.")


def _masked(pattern: re.Pattern[str], s: str, secret_span) -> set[int]:
    """Character indexes of s that a rule masks (the secret part of each match)."""
    out: set[int] = set()
    for m in pattern.finditer(s):
        start, end = secret_span(m)
        out.update(range(start, end))
    return out


_URL_SECRET = lambda m: (m.end(1), m.end() - 1)  # noqa: E731  credentials between '://' and '@'
_KV_SECRET = lambda m: m.span(3)  # noqa: E731  the value
_OLD_JWT_SECRET = lambda m: m.span()  # noqa: E731
_NEW_JWT_SECRET = lambda m: (m.end(1), m.end())  # noqa: E731  group 1 is the preserved run prefix


class RedactionComplexityTests(unittest.TestCase):
    def test_worst_case_inputs_are_fast(self):
        budget_s = 0.25  # pre-fix worst case ~10 s; post-fix measured ~2-5 ms on the dev host
        for unit in _ADVERSARIAL:
            s = (unit * MAX_REDACT_CHARS)[:MAX_REDACT_CHARS]
            with self.subTest(unit=unit):
                t0 = time.perf_counter()
                redact(s)
                self.assertLess(time.perf_counter() - t0, budget_s)

    def test_line_with_four_max_fields_is_fast(self):
        # One log line can carry msg, ctx, exc message and trace at the limit.
        s = ("_a" * MAX_REDACT_CHARS)[:MAX_REDACT_CHARS]
        t0 = time.perf_counter()
        redact({"msg": s, "ctx": s, "exc": s, "trace": s})
        self.assertLess(time.perf_counter() - t0, 1.0)

    def test_never_masks_fewer_characters_than_previous_rules(self):
        # Span-based oracle: every character the old rule masked must still be masked by the new one.
        rules = obs._STRING_RULES
        pairs = ((_OLD_URL, _URL_SECRET, rules[1][0], _URL_SECRET),
                 (_OLD_KV, _KV_SECRET, rules[2][0], _KV_SECRET),
                 (_OLD_JWT, _OLD_JWT_SECRET, rules[3][0], _NEW_JWT_SECRET))
        atoms = ["a", "x", "_", "-", ".", "+", ":", "=", "/", "@", " ", '"', "'", "\\", "pwd", "token", "secret",
                 "api_key", "http", "://", "u:p@", "Z", "9", "é", "ſ", "K", "eyJ", "eyJaaaaa",
                 ".aaaaa"]
        rnd = random.Random(20261109)
        for _ in range(20_000):
            s = "".join(rnd.choice(atoms) for _ in range(rnd.randint(1, 14)))
            for old, old_span, new, new_span in pairs:
                missing = _masked(old, s, old_span) - _masked(new, s, new_span)
                self.assertFalse(missing, f"{new.pattern[:30]}... unmasks {sorted(missing)} in {s!r}")

    def test_jwt_rule_matches_previous_spans_exactly(self):
        new = obs._STRING_RULES[3][0]
        atoms = ["eyJ", "a", "_", "-", ".", " ", "9", "é", "eyJaaaaa", "aaaaa", ".aaaaa", "-eyJ", "_eyJ"]
        rnd = random.Random(7)
        for _ in range(20_000):
            s = "".join(rnd.choice(atoms) for _ in range(rnd.randint(1, 16)))
            self.assertEqual(_masked(_OLD_JWT, s, _OLD_JWT_SECRET), _masked(new, s, _NEW_JWT_SECRET), repr(s))

    def test_known_secrets_still_redacted(self):
        cases = {
            "db-password=hunter2 ok": "hunter2",
            "my_api_key: k3y": "k3y",
            'x-client_secret="s3cr3t"': "s3cr3t",
            '{"access_token": "v4lue"}': "v4lue",
            "conn postgres+psycopg://user:pw1@host/db": "pw1",
            "see:https://u:pw2@h": "pw2",
            "x.y://u:pw3@h": "pw3",
            "Authorization: Bearer abc.def": "abc.def",
        }
        for text, secret in cases.items():
            with self.subTest(text=text):
                out = redact(text)
                self.assertNotIn(secret, out)
                self.assertIn("REDACTED", out)

    def test_plain_text_unchanged(self):
        for text in ("PASS=12 DEFECT=3", "http://host/path", "a-b_c.d", "session closed"):
            self.assertEqual(redact(text), text)


# -- I-01 + latch test gaps --------------------------------------------------------------
class LatchSessionIdTests(AppCase):
    def latch_data(self) -> dict[str, Any]:
        data: dict[str, Any] = json.loads(self.latch.read_text(encoding="utf-8"))
        return data

    def test_fault_during_start_latch_carries_attempted_session_id(self):
        a = self.app()
        a.health.register("forced", lambda: (ComponentStatus.FAULT, "forced"), critical=True)
        with self.assertRaises(SessionError):
            a.start_session()
        self.assertIsNone(a.session_id)
        sid = self.latch_data()["session_id"]
        self.assertIsNotNone(sid)
        fault = next(r for r in a.store.list_audit() if r["target"] == "INITIALIZING->FAULT")
        self.assertEqual(sid, fault["correlation_id"])

    def test_fault_during_session_latch_carries_session_id(self):
        a = self.observing()
        sid = a.session_id
        self.fault_it(a)
        self.assertEqual(self.latch_data()["session_id"], sid)

    def test_fault_outside_session_latch_has_no_session_id(self):
        a = self.app()
        a.events._state.enter_fault("TEST_FAULT")
        self.assertIsNone(self.latch_data()["session_id"])

    def test_latch_write_oserror_keeps_fault_and_audit_latches(self):
        a = self.observing()
        with mock.patch("acurix_edge.app.os.replace", side_effect=PermissionError("denied")), \
                self.assertLogs("acurix_edge.app", level="CRITICAL") as logs:
            a.events._state.enter_fault("TEST_FAULT")
        self.assertIs(a.state.state, SystemState.FAULT)
        self.assertFalse(self.latch.exists())
        self.assertFalse(self.latch.with_name(self.latch.name + ".tmp").exists())  # F3: no stray .tmp
        self.assertTrue(any(getattr(r, "code", None) == "FAULT_LATCH_WRITE_FAILED" for r in logs.records))
        a.shutdown()
        b = self.app()
        self.assertIs(b.state.state, SystemState.FAULT)  # audit trail is the second latch source
        self.assertEqual(b.status()["fault_latch"]["source"], "audit")

    def test_latch_unlink_failure_aborts_reset(self):
        a = self.observing()
        self.fault_it(a)
        latch = self.latch
        real_unlink = Path.unlink

        def unlink(path, *args, **kwargs):
            if Path(path) == latch:
                raise PermissionError("locked")
            return real_unlink(path, *args, **kwargs)

        with mock.patch.object(Path, "unlink", unlink), self.assertRaises(SessionError):
            a.reset_fault("try")
        self.assertIs(a.state.state, SystemState.FAULT)
        self.assertTrue(latch.is_file())

    def test_latch_path_is_directory_latches_fault(self):
        self.app().shutdown()
        self.latch.mkdir()
        b = self.app()
        self.assertIs(b.state.state, SystemState.FAULT)
        self.assertEqual(b.status()["fault_latch"]["reason"], "FAULT_LATCH_UNREADABLE")


class CliResetWithoutDatabaseTests(AppCase):
    def run_cli(self, *args):
        root = logging.getLogger("acurix_edge")
        saved = (list(root.handlers), root.propagate, root.level)
        out, err = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = main(list(args))
        finally:
            root.handlers.clear()
            root.handlers.extend(saved[0])
            root.propagate = saved[1]
            root.setLevel(saved[2])
        return code, out.getvalue(), err.getvalue()

    def test_reset_fault_without_database_and_without_latch_is_refused(self):
        code, _out, err = self.run_cli("reset-fault", "--reason", "x", "--base-dir", str(self.tmp))
        self.assertEqual(code, 2)
        self.assertIn("RESET_REFUSED", err)

    def test_reset_fault_without_database_with_latch_is_audited(self):
        self.latch.parent.mkdir(parents=True, exist_ok=True)
        self.latch.write_text(json.dumps({"reason": "PRIOR_FAULT", "at_utc": "x"}), encoding="utf-8")
        code, out, _err = self.run_cli("reset-fault", "--reason", "verified", "--base-dir", str(self.tmp))
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["state"], "IDLE")
        self.assertFalse(self.latch.exists())
        a = self.app()
        reset = [r for r in a.store.list_audit() if r["action"] == "FAULT_RESET"]
        self.assertEqual(len(reset), 1)
        self.assertEqual(reset[0]["detail"]["latch"]["reason"], "PRIOR_FAULT")


if __name__ == "__main__":
    unittest.main()
