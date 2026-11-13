"""Ordem 2.1 (owner decisions 10/11/2026):
- migration 3: trigger aborts UPDATE on inspection_event; DELETE stays allowed;
- backup/restore review findings B1-B6 (one test or more per finding);
- clock tolerance from schema-validated configuration (app level; boundaries in test_health_005).
"""

import hashlib
import os
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast
from unittest import mock

from acurix_edge.app import Application
from acurix_edge.config.settings import AppConfig, ConfigError, _build
from acurix_edge.contracts.errors import ErrorCode
from acurix_edge.domain.state_machine import SystemState
from acurix_edge.health.checks import ClockCheck
from acurix_edge.health.service import ComponentStatus as C
from acurix_edge.persistence import sqlite_store
from acurix_edge.persistence.sqlite_store import (
    _MIGRATIONS,
    IMMUTABILITY_TRIGGER,
    INVALID_BACKUP_SUFFIX,
    SCHEMA_VERSION,
    PersistenceError,
    SqliteEventStore,
)
from tests.support import (
    FIXTURE_POLICY,
    HEALTHY_HOST,
    FixtureApprovedRule,
    FixtureOkAdapter,
    TempDirTestCase,
    request,
)
from tests.unit.test_migration_002 import event_rows, user_version
from tests.unit.test_persistence import env


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class V2Store(SqliteEventStore):
    MIGRATIONS = {1: _MIGRATIONS[1], 2: _MIGRATIONS[2]}


# -- migration 3 ---------------------------------------------------------------------------------
class Migration3Tests(TempDirTestCase):
    def store(self, name="m.db"):
        s = SqliteEventStore(self.tmp / name)
        s.open()
        self.addCleanup(s.close)
        return s

    def test_schema_3_and_trigger_present(self):
        self.assertEqual(SCHEMA_VERSION, 3)
        s = self.store()
        self.assertEqual(s.check(), (True, "schema 3"))
        sql = s._c().execute("SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?",
                             (IMMUTABILITY_TRIGGER,)).fetchone()[0]
        self.assertIn("BEFORE UPDATE ON inspection_event", sql)
        triggers = s._c().execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall()
        self.assertEqual(triggers, [(IMMUTABILITY_TRIGGER,)])  # no DELETE trigger (owner decision)

    def test_update_aborted_with_clear_message_delete_allowed(self):
        s = self.store()
        e1, e2 = env(seq=1), env(seq=2)
        s.append(e1)
        s.append(e2)
        before = event_rows(self.tmp / "m.db")
        for sql in ("UPDATE inspection_event SET reason = 'X' WHERE event_id = ?",
                    "UPDATE inspection_event SET review_status = 'CONFIRMED' WHERE event_id = ?",
                    "UPDATE inspection_event SET payload_json = payload_json WHERE event_id = ?"):  # even a no-op
            with self.subTest(sql=sql), self.assertRaises(sqlite3.IntegrityError) as cm:
                s._c().execute(sql, (e1.event_id,))
            self.assertIn("inspection_event is immutable", str(cm.exception))
        with self.assertRaises(sqlite3.IntegrityError):  # also inside an explicit transaction
            s._c().execute("BEGIN")
            try:
                s._c().execute("UPDATE inspection_event SET reason = 'X'")
            finally:
                s._c().execute("ROLLBACK")
        self.assertEqual(event_rows(self.tmp / "m.db"), before)
        s._c().execute("DELETE FROM inspection_event WHERE event_id = ?", (e2.event_id,))  # retention: allowed
        self.assertEqual(s.count_events(), 1)
        s.append(env(seq=3))  # inserts unaffected
        self.assertEqual(s.count_events(), 2)

    def test_r3_m1_upsert_and_update_or_replace_are_blocked(self):
        s = self.store()
        e = env()
        s.append(e)
        before = event_rows(self.tmp / "m.db")
        cols = "event_id, schema_version, site_id, station_id, session_id, inspection_id, captured_at_utc, " \
               "capture_sequence, result, reason, inference_status, config_version, config_sha256, " \
               "review_status, software_version, payload_json, payload_sha256, persisted_at_utc"
        row = s._c().execute(f"SELECT {cols} FROM inspection_event").fetchone()
        marks = ",".join("?" * len(row))
        for sql in (f"INSERT INTO inspection_event ({cols}) VALUES ({marks}) "
                    "ON CONFLICT(event_id) DO UPDATE SET reason = 'UPSERTED'",
                    "UPDATE OR REPLACE inspection_event SET reason = 'X'"):
            with self.subTest(sql=sql[:40]), self.assertRaises(sqlite3.IntegrityError) as cm:
                s._c().execute(sql, row if "VALUES" in sql else ())
            self.assertIn("immutable", str(cm.exception))
        self.assertEqual(event_rows(self.tmp / "m.db"), before)

    def test_r3_m1_replace_known_limit_documented(self):
        # REPLACE = DELETE + INSERT for SQLite: the BEFORE UPDATE trigger does not fire. This test
        # pins the documented limit (the owner left DELETE open for the retention purge); the
        # static test below keeps REPLACE out of product code.
        s = self.store()
        e = env()
        s.append(e)
        cols = [r[1] for r in s._c().execute("PRAGMA table_info(inspection_event)")]
        select = ", ".join("'REWRITTEN'" if c == "reason" else c for c in cols)
        s._c().execute(f"INSERT OR REPLACE INTO inspection_event SELECT {select} FROM inspection_event")
        self.assertEqual(s.count_events(), 1)
        reason = s._c().execute("SELECT reason FROM inspection_event").fetchone()[0]
        self.assertEqual(reason, "REWRITTEN")  # the known limit is real: REPLACE bypasses the trigger

    # Review barrier, not a guarantee (review R4-01): SQL built from non-literal values is not seen.
    _FORBIDDEN_SQL = __import__("re").compile(
        r"(?is)(\bREPLACE\s+INTO\b|\bOR\s+REPLACE\b|\bINSERT\s+OR\s+\{\}"
        r"|\bUPDATE\s+(OR\s+\w+\s+)?[\"\[`]?(main\.)?(inspection_event\b|\{\})"
        r"|ON\s+CONFLICT[^;]*?DO\s+UPDATE)")

    @staticmethod
    def _string_literals(source: str) -> list[tuple[int, str]]:
        """String literals only (comments never count). Adjacent literals and `+` chains are
        joined; f-string placeholders become `{}`."""
        import ast

        def text(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return node.value
            if isinstance(node, ast.JoinedStr):
                return "".join(cast(str, v.value) if isinstance(v, ast.Constant) else "{}" for v in node.values)
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                left, right = text(node.left), text(node.right)
                return None if left is None and right is None else (left or "{}") + (right or "{}")
            return None

        out = []
        for node in ast.walk(ast.parse(source)):
            t = text(node)
            if t is not None:
                out.append((getattr(node, "lineno", 0), t))
        return out

    def test_r3_m1_no_replace_or_update_of_events_in_product_code(self):
        src = Path(sqlite_store.__file__).resolve().parents[1]
        offenders = sorted({f"{f.name}:{line}" for f in src.rglob("*.py")
                            for line, s in self._string_literals(f.read_text(encoding="utf-8"))
                            if self._FORBIDDEN_SQL.search(s)})
        self.assertEqual(offenders, [])

    def test_r3_m1_scanner_catches_known_evasions(self):
        samples = {
            "replace": 'c.execute("INSERT OR REPLACE INTO inspection_event VALUES (?)")',
            "split": 'c.execute("INSERT OR " + "REPLACE" + " INTO inspection_event")',
            "implicit": 'c.execute("INSERT OR " "REPLACE INTO t")',
            "fstring_table": 'c.execute(f"UPDATE {TABLE} SET reason = 1")',
            "fstring_mode": 'c.execute(f"INSERT OR {mode} INTO inspection_event")',
            "quoted": "c.execute('UPDATE \"inspection_event\" SET x = 1')",
            "schema": 'c.execute("UPDATE main.inspection_event SET x = 1")',
            "upsert": 'c.execute("INSERT INTO inspection_event VALUES (1) ON CONFLICT(event_id) DO UPDATE SET x=1")',
            "after_comment": '# ON CONFLICT DO UPDATE is blocked\nc.execute("UPDATE inspection_event SET x=1")',
        }
        for name, code in samples.items():
            with self.subTest(name):
                self.assertTrue(any(self._FORBIDDEN_SQL.search(s) for _, s in self._string_literals(code)))
        ok = '# INSERT OR REPLACE in a comment\nc.execute("INSERT INTO inspection_event VALUES (?)")'
        self.assertFalse(any(self._FORBIDDEN_SQL.search(s) for _, s in self._string_literals(ok)))

    def test_r3_ba_altered_trigger_with_same_name_detected(self):
        s = self.store()
        s._c().execute(f"DROP TRIGGER {IMMUTABILITY_TRIGGER}")
        s._c().execute(f"CREATE TRIGGER {IMMUTABILITY_TRIGGER} BEFORE UPDATE ON inspection_event WHEN 0 "
                       "BEGIN SELECT RAISE(ABORT, 'x'); END")
        ok, detail = s.check()
        self.assertFalse(ok)
        self.assertIn("altered", detail)

    def test_check_reports_missing_trigger(self):
        s = self.store()
        s._c().execute(f"DROP TRIGGER {IMMUTABILITY_TRIGGER}")
        ok, detail = s.check()
        self.assertFalse(ok)
        self.assertIn("immutability trigger", detail)

    def test_2_to_3_copy_first_then_original_events_preserved(self):
        original = self.tmp / "acurix.db"
        v2 = V2Store(original)
        v2.open()
        for i in range(1, 5):
            v2.append(env(seq=i))
        v2.audit(actor_ref="t", action="A", target="x")
        v2.close()
        self.assertEqual(user_version(original), 2)
        before = event_rows(original)

        copy = self.tmp / "copy" / "acurix.db"
        copy.parent.mkdir()
        src, dst = sqlite3.connect(original), sqlite3.connect(copy)
        src.backup(dst)
        src.close()
        dst.close()
        c = SqliteEventStore(copy)
        c.open()
        self.assertEqual((c.schema_version(), c.check()[0]), (3, True))
        self.assertEqual(event_rows(copy), before)
        c.close()
        self.assertEqual(user_version(original), 2)  # rehearsal did not touch the original

        s = SqliteEventStore(original)
        s.open()
        self.addCleanup(s.close)
        self.assertEqual((s.schema_version(), s.check()), (3, (True, "schema 3")))
        self.assertEqual(event_rows(original), before)
        self.assertTrue(s.verify_audit_chain().ok)
        backups = list(self.tmp.glob("acurix.db.pre-v3-*.bak"))
        self.assertEqual(len(backups), 1)
        self.assertEqual((user_version(backups[0]), event_rows(backups[0])), (2, before))
        # Review R4-05: a legitimate pre-migration backup (user_version 2) is still restorable (B-d).
        restored = SqliteEventStore.restore_backup(backups[0], self.tmp / "restored" / "acurix.db")
        self.assertEqual((user_version(restored), event_rows(restored)), (2, before))

    def test_failed_migration_3_rolls_back_to_v2(self):
        path = self.tmp / "f.db"
        v2 = V2Store(path)
        v2.open()
        v2.append(env())
        v2.close()
        before = event_rows(path)

        class Broken(SqliteEventStore):
            MIGRATIONS = {**_MIGRATIONS, 3: (_MIGRATIONS[3][0], "NOT SQL")}

        with self.assertRaises(PersistenceError):
            Broken(path).open()
        self.assertEqual(user_version(path), 2)
        self.assertEqual(event_rows(path), before)
        c = sqlite3.connect(path)
        self.assertEqual(c.execute("SELECT count(*) FROM sqlite_master WHERE type='trigger'").fetchone()[0], 0)
        c.close()


# -- backup / restore findings B1-B6 ---------------------------------------------------------------
class _FailingBackupConn(sqlite3.Connection):
    exc: BaseException = sqlite3.OperationalError("injected failure during copy")

    def backup(self, *a, **k):
        raise type(self).exc


class _BadVerdictConn(sqlite3.Connection):
    def execute(self, sql, *args):
        if "quick_check" in sql:
            return super().execute("SELECT 'page 3: corrupt'")
        return super().execute(sql, *args)


def patch_connect(src_factory=None, target_factory=None):
    """restore_backup opens the source with uri=True and the target without."""
    real = sqlite3.connect

    def fake(database, *a, **k):
        factory = src_factory if k.get("uri") else target_factory
        return real(database, *a, **({**k, "factory": factory} if factory else k))

    return mock.patch.object(sqlite3, "connect", side_effect=fake)


class RestoreFindingsTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        s = SqliteEventStore(self.tmp / "live.db")
        s.open()
        s.append(env())
        self.bak = s.backup(self.tmp / "good.bak")
        s.close()
        self.live = self.tmp / "live.db"

    def invalids(self, name):
        return sorted(self.tmp.glob(name + ".*" + INVALID_BACKUP_SUFFIX))

    # B1 -------------------------------------------------------------------------------------------
    def test_b1_quarantine_never_overwrites_existing_invalid(self):
        dest = self.tmp / "r.db"
        fixed = "2026-11-10T12:00:00.000Z"
        with mock.patch.object(sqlite_store, "utc_now_iso", return_value=fixed), \
                patch_connect(src_factory=_FailingBackupConn):
            with self.assertRaises(PersistenceError):
                SqliteEventStore.restore_backup(self.bak, dest)
            first = self.invalids("r.db")
            self.assertEqual([p.name for p in first], ["r.db.20261110T120000000Z.invalid"])
            first[0].write_bytes(b"forensic evidence of failure 1")
            with self.assertRaises(PersistenceError):  # same dest, same timestamp: must NOT replace
                SqliteEventStore.restore_backup(self.bak, dest)
        names = [p.name for p in self.invalids("r.db")]
        self.assertEqual(names, ["r.db.20261110T120000000Z-1.invalid", "r.db.20261110T120000000Z.invalid"])
        self.assertEqual(first[0].read_bytes(), b"forensic evidence of failure 1")

    # B2 -------------------------------------------------------------------------------------------
    def test_b2_interruption_after_creation_is_quarantined_and_propagated(self):
        bak_hash = sha(self.bak)
        for exc in (KeyboardInterrupt(), MemoryError()):
            with self.subTest(exc=type(exc).__name__):
                dest = self.tmp / f"int-{type(exc).__name__}.db"
                _FailingBackupConn.exc = exc
                self.addCleanup(setattr, _FailingBackupConn, "exc", sqlite3.OperationalError("injected"))
                with patch_connect(src_factory=_FailingBackupConn), self.assertRaises(type(exc)):
                    SqliteEventStore.restore_backup(self.bak, dest)
                self.assertFalse(dest.exists())  # never left under a normal database name
                self.assertEqual(len(self.invalids(dest.name)), 1)
        self.assertEqual(sha(self.bak), bak_hash)  # source untouched (read-only open)

    def test_b2_interruption_during_backup_is_quarantined(self):
        s = SqliteEventStore(self.tmp / "live.db")
        s.open()
        self.addCleanup(s.close)
        dest = self.tmp / "bk.db"
        with patch_connect(target_factory=_BadVerdictConn), \
                mock.patch.object(_BadVerdictConn, "execute", side_effect=KeyboardInterrupt), \
                self.assertRaises(KeyboardInterrupt):
            s.backup(dest)
        self.assertFalse(dest.exists())
        self.assertEqual(len(self.invalids("bk.db")), 1)

    # B3 -------------------------------------------------------------------------------------------
    def test_b3_quarantine_only_touches_the_file_this_call_created(self):
        f = self.tmp / "someone_else.db"
        f.write_bytes(b"not ours")
        msg = SqliteEventStore._quarantine_backup(f, (0, 0))
        self.assertIn("left untouched", msg)
        self.assertEqual(f.read_bytes(), b"not ours")
        self.assertEqual(self.invalids(f.name), [])
        st = f.stat()
        self.assertIn("marked", SqliteEventStore._quarantine_backup(f, (st.st_dev, st.st_ino)))
        self.assertFalse(f.exists())

    def test_b3_replaced_target_is_not_quarantined(self):
        dest = self.tmp / "swap.db"
        real_create = SqliteEventStore._create_exclusive

        def create_then_report_other_identity(path, what):
            real_create(path, what)
            return (-1, -1)  # as if another process swapped the file right after creation

        with mock.patch.object(SqliteEventStore, "_create_exclusive", side_effect=create_then_report_other_identity), \
                patch_connect(src_factory=_FailingBackupConn), self.assertRaises(PersistenceError) as cm:
            SqliteEventStore.restore_backup(self.bak, dest)
        self.assertIn("left untouched", str(cm.exception))
        self.assertTrue(dest.exists())
        self.assertEqual(self.invalids("swap.db"), [])

    # B4 -------------------------------------------------------------------------------------------
    def test_b4_invalid_suffix_refused_in_any_case(self):
        for name in ("x.bak.INVALID", "x.bak.Invalid", "x.bak.invalid"):
            with self.subTest(name=name):
                p = self.tmp / name
                p.write_bytes(self.bak.read_bytes())  # content is even a valid database
                with self.assertRaises(PersistenceError) as cm:
                    SqliteEventStore.restore_backup(p, self.tmp / ("out-" + name + ".db"))
                self.assertIn("marked invalid", str(cm.exception))
                self.assertFalse((self.tmp / ("out-" + name + ".db")).exists())

    # B5 -------------------------------------------------------------------------------------------
    def test_b5_leftover_side_files_refused_untouched(self):
        for suffix in ("-wal", "-shm", "-journal"):
            with self.subTest(suffix=suffix):
                dest = self.tmp / f"side{suffix}.db"
                side = dest.with_name(dest.name + suffix)
                side.write_bytes(b"stale journal")
                with self.assertRaises(PersistenceError) as cm:
                    SqliteEventStore.restore_backup(self.bak, dest)
                self.assertIn("leftover journal", str(cm.exception))
                self.assertFalse(dest.exists())
                self.assertEqual(side.read_bytes(), b"stale journal")

    def test_b5_quarantine_moves_side_files_with_the_file(self):
        f = self.tmp / "q.db"
        f.write_bytes(b"partial")
        (self.tmp / "q.db-journal").write_bytes(b"j")
        (self.tmp / "q.db-wal").write_bytes(b"w")
        st = f.stat()
        msg = SqliteEventStore._quarantine_backup(f, (st.st_dev, st.st_ino))
        invalid = self.invalids("q.db")[0]
        self.assertIn(invalid.name, msg)
        self.assertEqual(invalid.with_name(invalid.name + "-journal").read_bytes(), b"j")
        self.assertEqual(invalid.with_name(invalid.name + "-wal").read_bytes(), b"w")
        self.assertFalse(any((self.tmp / ("q.db" + s)).exists() for s in ("", "-journal", "-wal")))

    # B6 -------------------------------------------------------------------------------------------
    def test_b6a_valid_header_corrupt_page_onto_live_db_is_untouched(self):
        corrupt = self.tmp / "corrupt.bak"
        c = sqlite3.connect(corrupt)
        c.execute("CREATE TABLE t (x TEXT)")
        c.executemany("INSERT INTO t VALUES (?)", [("v" * 200,)] * 400)
        c.commit()
        page = c.execute("PRAGMA page_size").fetchone()[0]
        c.close()
        data = bytearray(corrupt.read_bytes())
        self.assertEqual(data[:16], b"SQLite format 3\x00")
        data[page * 2:page * 2 + 600] = b"\xff" * 600  # page 3 garbage, header intact
        corrupt.write_bytes(bytes(data))
        live_hash = sha(self.live)
        with self.assertRaises(PersistenceError) as cm:
            SqliteEventStore.restore_backup(corrupt, self.live)
        self.assertIn("target untouched", str(cm.exception))
        self.assertEqual(sha(self.live), live_hash)
        self.assertEqual(self.invalids("live.db"), [])

    def test_b6b_failure_after_creation_quarantines_only_the_new_file(self):
        dest = self.tmp / "new.db"
        live_hash, bak_hash = sha(self.live), sha(self.bak)
        with patch_connect(src_factory=_FailingBackupConn), self.assertRaises(PersistenceError) as cm:
            SqliteEventStore.restore_backup(self.bak, dest)
        self.assertIs(cm.exception.code, ErrorCode.DB_001)
        self.assertIn("marked new.db.", str(cm.exception))
        self.assertFalse(dest.exists())
        self.assertEqual(len(self.invalids("new.db")), 1)
        self.assertEqual((sha(self.live), sha(self.bak)), (live_hash, bak_hash))

    def test_b6c_restored_copy_failing_quick_check_is_quarantined(self):
        dest = self.tmp / "verdict.db"
        with patch_connect(target_factory=_BadVerdictConn), self.assertRaises(PersistenceError) as cm:
            SqliteEventStore.restore_backup(self.bak, dest)
        self.assertIn("restored copy fails quick_check", str(cm.exception))
        self.assertFalse(dest.exists())
        self.assertEqual(len(self.invalids("verdict.db")), 1)

    def test_b6d_dest_equal_to_backup_leaves_backup_intact(self):
        h = sha(self.bak)
        for dest in (self.bak, self.bak.parent / "." / self.bak.name, Path(str(self.bak).upper())):
            with self.subTest(dest=str(dest)), self.assertRaises(PersistenceError) as cm:
                SqliteEventStore.restore_backup(self.bak, dest)
            self.assertIn("already exists", str(cm.exception))
        self.assertEqual(sha(self.bak), h)
        self.assertEqual(self.invalids("good.bak"), [])

    def test_backup_never_overwrites_an_existing_file(self):
        s = SqliteEventStore(self.live)
        s.open()
        self.addCleanup(s.close)
        other = self.tmp / "other.db"
        other.write_bytes(b"precious")
        with self.assertRaises(PersistenceError):
            s.backup(other)
        self.assertEqual(other.read_bytes(), b"precious")

    # Review round 3 (B-c, B-d, B-e, B-f) -----------------------------------------------------------
    def test_r3_bc_unlink_fallback_rechecks_identity(self):
        f = self.tmp / "u.db"
        f.write_bytes(b"ours")
        st = f.stat()
        with mock.patch.object(os, "rename", side_effect=PermissionError("locked by AV")):
            msg = SqliteEventStore._quarantine_backup(f, (st.st_dev, st.st_ino))
        self.assertIn("deleted (rename failed)", msg)
        self.assertFalse(f.exists())
        g = self.tmp / "v.db"
        g.write_bytes(b"not ours")
        real_stat = Path.stat
        calls = []

        def stat_then_swapped(self_, *a, **k):
            st_ = real_stat(self_, *a, **k)
            if self_ != g:
                return st_
            calls.append(1)  # 1st stat of g: identity check passes; later: "another file"
            return st_ if len(calls) == 1 else mock.Mock(st_dev=-1, st_ino=-1)

        gst = g.stat()
        with mock.patch.object(os, "rename", side_effect=PermissionError("locked")), \
                mock.patch.object(Path, "stat", stat_then_swapped):
            msg = SqliteEventStore._quarantine_backup(g, (gst.st_dev, gst.st_ino))
        self.assertIn("left untouched", msg)
        self.assertEqual(g.read_bytes(), b"not ours")

    def test_r3_bd_empty_or_foreign_sqlite_is_not_a_backup(self):
        empty = self.tmp / "empty.bak"
        empty.write_bytes(b"")  # e.g. a 0-byte leftover: quick_check says "ok"
        foreign = self.tmp / "foreign.bak"
        c = sqlite3.connect(foreign)
        c.execute("CREATE TABLE other (x)")
        c.execute("PRAGMA user_version = 7")
        c.commit()
        c.close()
        for b in (empty, foreign):
            with self.subTest(backup=b.name), self.assertRaises(PersistenceError) as cm:
                SqliteEventStore.restore_backup(b, self.tmp / ("out-" + b.name + ".db"))
            self.assertIn("not an Acurix database", str(cm.exception))
            self.assertFalse((self.tmp / ("out-" + b.name + ".db")).exists())

    def test_r3_be_journal_mode_switch_refused_is_an_error(self):
        class StuckInWal(sqlite3.Connection):
            def execute(self, sql, *args):
                if "journal_mode" in sql:
                    return super().execute("SELECT 'wal'")
                return super().execute(sql, *args)

        dest = self.tmp / "stuck.db"
        with patch_connect(target_factory=StuckInWal), self.assertRaises(PersistenceError) as cm:
            SqliteEventStore.restore_backup(self.bak, dest)
        self.assertIn("journal mode", str(cm.exception))
        self.assertFalse(dest.exists())
        self.assertEqual(len(self.invalids("stuck.db")), 1)

    def test_r3_bf_side_file_appearing_after_creation_is_refused(self):
        dest = self.tmp / "late.db"
        real_create = SqliteEventStore._create_exclusive

        def create_then_side_file(path, what):
            ident = real_create(path, what)
            path.with_name(path.name + "-journal").write_bytes(b"alien")
            return ident

        with mock.patch.object(SqliteEventStore, "_create_exclusive", side_effect=create_then_side_file), \
                self.assertRaises(PersistenceError) as cm:
            SqliteEventStore.restore_backup(self.bak, dest)
        self.assertIn("leftover journal", str(cm.exception))
        self.assertFalse(dest.exists())  # our new file quarantined, never left under a normal name
        invalid = self.invalids("late.db")
        self.assertEqual(len(invalid), 1)
        # Review R4-04: the alien side file is renamed (paired with the quarantined file), never deleted.
        self.assertEqual(invalid[0].with_name(invalid[0].name + "-journal").read_bytes(), b"alien")
        self.assertFalse(dest.with_name(dest.name + "-journal").exists())
        self.assertIn(invalid[0].name, str(cm.exception))

    def test_backup_is_self_contained_no_side_files(self):
        # Found while testing B6d: a WAL-mode copy left -wal/-shm next to the backup when read.
        self.assertFalse(any(self.tmp.glob("good.bak-*")))
        c = sqlite3.connect(f"{self.bak.resolve().as_uri()}?mode=ro", uri=True)
        self.assertEqual(c.execute("PRAGMA journal_mode").fetchone()[0], "delete")
        c.close()
        self.assertFalse(any(self.tmp.glob("good.bak-*")))

    def test_round_trip_still_works(self):
        out = SqliteEventStore.restore_backup(self.bak, self.tmp / "ok" / "restored.db")
        r = SqliteEventStore(out)
        r.open()
        self.addCleanup(r.close)
        self.assertEqual((r.check()[0], r.count_events()), (True, 1))


# -- clock tolerance: configuration + application ------------------------------------------------
class ClockConfigTests(TempDirTestCase):
    def test_default_and_schema_validation(self):
        self.assertEqual(AppConfig().clock.backward_tolerance_s, 2.0)
        self.assertEqual(_build({"clock": {"backward_tolerance_s": 1.5}}).clock.backward_tolerance_s, 1.5)
        self.assertEqual(_build({"clock": {"backward_tolerance_s": 0}}).clock.backward_tolerance_s, 0)
        for bad in (-0.1, 10.5, "2", True, float("nan"), float("inf")):
            with self.subTest(bad=bad), self.assertRaises(ConfigError) as cm:
                _build({"clock": {"backward_tolerance_s": bad}})
            self.assertTrue(any("clock.backward_tolerance_s" in p for p in cm.exception.problems))
        with self.assertRaises(ConfigError):
            _build({"clock": {"tolerance": 2}})  # unknown key

    def test_example_toml_declares_the_tolerance(self):
        from acurix_edge.config.settings import load_config

        example = Path(__file__).resolve().parents[2] / "config" / "acurix.example.toml"
        self.assertEqual(load_config(example).clock.backward_tolerance_s, 2.0)

    def app(self, offset, tolerance=None):
        cfg = AppConfig() if tolerance is None else _build({"clock": {"backward_tolerance_s": tolerance}})
        store = SqliteEventStore(cfg.database_path(self.tmp))
        clock = ClockCheck(backward_tolerance_s=cfg.clock.backward_tolerance_s,
                           now=lambda: datetime.now(UTC) + offset[0],
                           last_event_utc=store.last_event_persisted_utc)
        a = Application(cfg, self.tmp, model_policy=FIXTURE_POLICY, inference=FixtureOkAdapter(),
                        rule=FixtureApprovedRule(), capture_probe=lambda: (C.OK, "fixture"), store=store,
                        host_checks={**HEALTHY_HOST, "clock": clock})
        a.boot()
        self.addCleanup(a.shutdown)
        return a

    def test_within_tolerance_logged_state_unchanged(self):
        offset = [timedelta(0)]
        a = self.app(offset)
        a.start_session()
        a.process(request())
        last = datetime.fromisoformat(a.store.last_event_persisted_utc()[:-1] + "+00:00")
        offset[0] = (last - datetime.now(UTC)) - timedelta(seconds=1.0)  # ~1 s behind last event
        with self.assertLogs("acurix_edge.health.clock", "WARNING"):
            st = a.status()
        clock = next(c for c in st["components"] if c["component"] == "clock")
        self.assertEqual((clock["status"], clock["code"]), ("OK", "AEX-CLK-001"))
        self.assertIs(a.state.state, SystemState.OBSERVING)  # state unchanged
        self.assertEqual(a.process(request(inspection_id="i2")).result, "PASS")

    def test_beyond_tolerance_fault_latched(self):
        offset = [timedelta(0)]
        a = self.app(offset)
        a.start_session()
        a.process(request())
        last = datetime.fromisoformat(a.store.last_event_persisted_utc()[:-1] + "+00:00")
        offset[0] = (last - datetime.now(UTC)) - timedelta(seconds=3.0)
        a.refresh_health()
        self.assertIs(a.state.state, SystemState.FAULT)

    def test_configured_tolerance_is_used_by_the_app(self):
        offset = [timedelta(0)]
        a = self.app(offset, tolerance=5)
        a.start_session()
        a.process(request())
        last = datetime.fromisoformat(a.store.last_event_persisted_utc()[:-1] + "+00:00")
        offset[0] = (last - datetime.now(UTC)) - timedelta(seconds=3.0)  # > 2 s but < 5 s
        with self.assertLogs("acurix_edge.health.clock", "WARNING"):
            a.refresh_health()
        self.assertIs(a.state.state, SystemState.OBSERVING)
