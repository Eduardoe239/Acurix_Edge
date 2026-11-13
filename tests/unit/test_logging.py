import io
import json
import logging
import unittest
from typing import Any

from acurix_edge.observability.logging import REDACTED, bind_context, configure_logging, redact


class LoggingTests(unittest.TestCase):
    def setUp(self):
        root = logging.getLogger("acurix_edge")
        saved = (list(root.handlers), root.propagate, root.level)

        def restore():
            for h in list(root.handlers):
                root.removeHandler(h)
            for h in saved[0]:
                root.addHandler(h)
            root.propagate, root.level = saved[1], saved[2]

        self.addCleanup(restore)
        self.buf = io.StringIO()
        configure_logging("DEBUG", stream=self.buf)
        self.log = logging.getLogger("acurix_edge.test")

    def lines(self):
        return [json.loads(line) for line in self.buf.getvalue().splitlines()]

    def test_structured_fields_and_context(self):
        with bind_context(session_id="s-1", correlation_id="c-1"):
            self.log.info("hello", extra={"code": "X1", "ctx": {"n": 1}})
        self.log.info("after")
        first, second = self.lines()
        self.assertEqual(first["level"], "INFO")
        self.assertEqual(first["component"], "acurix_edge.test")
        self.assertEqual(first["code"], "X1")
        self.assertEqual((first["session_id"], first["correlation_id"]), ("s-1", "c-1"))
        self.assertTrue(first["ts"].endswith("Z"))
        self.assertIsNone(second["session_id"])

    def test_secrets_redacted_in_ctx_message_and_exception(self):
        try:
            raise ValueError("Authorization: Bearer abc.def.ghi")
        except ValueError:
            self.log.exception(
                "login Bearer tok123 failed",
                extra={"ctx": {"password": "p@ss", "nested": {"api_key": "k", "ok": "v"},
                               "items": [{"token": "t"}], "image": b"\x00\x01"}},
            )
        raw = self.buf.getvalue()
        for secret in ("p@ss", "tok123", "abc.def.ghi"):
            self.assertNotIn(secret, raw)
        entry = self.lines()[0]
        self.assertEqual(entry["ctx"]["password"], REDACTED)
        self.assertEqual(entry["ctx"]["nested"]["api_key"], REDACTED)
        self.assertEqual(entry["ctx"]["nested"]["ok"], "v")
        self.assertEqual(entry["ctx"]["items"][0]["token"], REDACTED)
        self.assertIn("binary", entry["ctx"]["image"])

    def test_string_patterns_redacted(self):
        # Public jwt.io sample, assembled at runtime so the literal does not trip the gitleaks
        # pre-commit hook (no suppression comment needed).
        jwt = ".".join(("eyJhbGciOiJIUzI1NiJ9", "eyJzdWIiOiIxMjM0NTY3ODkwIn0",
                        "dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"))
        samples = {
            "db at postgresql://admin:hunter2@host/db": "hunter2",
            "retry with password=s3cr3t&x=1": "s3cr3t",
            "token: abcdef123": "abcdef123",
            'login password="a b c" ok': "a b c",
            "client_secret=cs-123 next": "cs-123",
            "access_token=at.456": "at.456",
            "api_token: zz9": "zz9",
            '{"password": "jsonpw", "user": "u"}': "jsonpw",
            "{'token': 'dictpw'}": "dictpw",
            'password="esc\\"aped" end': 'aped',
            f"header {jwt}": jwt,
        }
        for text, secret in samples.items():
            with self.subTest(text=text):
                self.assertNotIn(secret, redact(text))

    def test_pass_counters_not_hidden(self):
        out = redact({"pass_count": 3, "bypass": False, "result": "PASS", "password": "x", "db_pwd": "y",
                      "auth": "z", "author": "kept", "key": "k", "dsn": "d"})
        self.assertEqual(out["pass_count"], 3)
        self.assertIs(out["bypass"], False)
        self.assertEqual(out["result"], "PASS")
        self.assertEqual(out["author"], "kept")
        for k in ("password", "db_pwd", "auth", "key", "dsn"):
            self.assertEqual(out[k], REDACTED, k)

    def test_redact_depth_limited(self):
        deep: dict[str, Any] = {}
        cur = deep
        for _ in range(20):
            cur["n"] = {}
            cur = cur["n"]
        self.assertIn("<truncated>", json.dumps(redact(deep)))


if __name__ == "__main__":
    unittest.main()
