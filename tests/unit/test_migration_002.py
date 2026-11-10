"""Migration 2 (Anexo 02A §2, APPROVED 10/11/2026), audit hash chain (AUD-002) and clock rule
AEX-CLK-001 (owner decision 10/11/2026).

Migration acceptance: 1->2 tested on a COPY first, then on the original; automatic verified backup;
no existing event lost or altered (every column compared byte for byte); reviews never alter
inspection_event; audit hash chain verifiable (legacy prefix + chained entries).
"""

import json
import re
import sqlite3
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from acurix_edge.app import Application, SessionError
from acurix_edge.config.settings import AppConfig
from acurix_edge.contracts.errors import ErrorCode
from acurix_edge.domain.state_machine import SystemState
from acurix_edge.health.checks import ClockCheck
from acurix_edge.health.service import ComponentStatus as C
from acurix_edge.persistence.sqlite_store import (
    AUDIT_GENESIS_SHA256,
    EXPECTED_TABLES,
    PersistenceError,
    SqliteEventStore,
    _MIGRATIONS,
    audit_entry_sha256,
)
from tests.support import (
    FIXTURE_POLICY,
    HEALTHY_HOST,
    FixtureApprovedRule,
    FixtureOkAdapter,
    TempDirTestCase,
    request,
)
from tests.unit.test_persistence import env

ANEXO_02A = Path(__file__).resolve().parents[2] / "docs" / "02-architecture" / "02A_ANEXO_API_DADOS_ERROS_v2.0.0.md"
MIGRATION_2_TABLES = {"user_account", "login_session", "inspection_session", "cascade_step", "review",
                      "media_asset", "system_incident", "export_record"}


class V1Store(SqliteEventStore):
    """The 0.1.0 Foundation schema exactly as shipped (migration 1 only)."""

    MIGRATIONS = {1: _MIGRATIONS[1]}


def _normalize(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).replace("( ", "(").replace(" )", ")").strip()


def event_rows(path: Path) -> list[tuple]:
    c = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        return c.execute("SELECT * FROM inspection_event ORDER BY event_id").fetchall()
    finally:
        c.close()


def user_version(path: Path) -> int:
    c = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        return int(c.execute("PRAGMA user_version").fetchone()[0])
    finally:
        c.close()


class MigrationTextTests(unittest.TestCase):
    @unittest.skipUnless(ANEXO_02A.is_file(), "docs/ not present (not versioned)")
    def test_migration_2_is_the_approved_sql_verbatim(self):
        text = ANEXO_02A.read_text(encoding="utf-8")
        section = text.split("## 2. Esquema do banco", 1)[1].split("## 3.", 1)[0]
        block = re.search(r"```sql\n(.*?)```", section, re.S).group(1)
        self.assertIn("APROVADA", block)
        no_comments = "\n".join(line.split("--", 1)[0] for line in block.splitlines())
        doc_statements = [_normalize(s) for s in no_comments.split(";") if s.strip()]
        self.assertEqual([_normalize(s) for s in _MIGRATIONS[2]], doc_statements)

    def test_migration_1_unchanged_and_2_never_touches_events(self):
        self.assertEqual(len(_MIGRATIONS[1]), 5)
        for stmt in _MIGRATIONS[2]:
            norm = _normalize(stmt).upper()
            self.assertFalse(norm.startswith(("DROP", "DELETE", "UPDATE")), stmt)
            self.assertNotRegex(norm, r"^(ALTER TABLE|CREATE TABLE) INSPECTION_EVENT\b")


class Migration1To2Tests(TempDirTestCase):
    def make_v1(self, path: Path) -> tuple[list[tuple], int]:
        """A v1 database with events (incl. a legacy 1.0-era layout) and legacy audit rows."""
        s = V1Store(path)
        s.open()
        for i in range(1, 6):
            s.append(env(seq=i))
        conn = s._c()
        for i in range(3):  # 0.1.0 audit() wrote exactly these columns
            conn.execute("INSERT INTO audit_entry (occurred_at_utc, actor_ref, action, target, detail_json,"
                         " correlation_id) VALUES (?,?,?,?,?,?)",
                         (f"2026-11-0{i + 1}T10:00:00.000Z", "system", "STATE_TRANSITION", "IDLE->INITIALIZING",
                          json.dumps({"seq": i}, sort_keys=True), None if i else "s-1"))
        s.close()
        self.assertEqual(user_version(path), 1)
        return event_rows(path), 3

    def migrate(self, path: Path) -> SqliteEventStore:
        s = SqliteEventStore(path)
        s.open()
        self.addCleanup(s.close)
        return s

    def test_copy_first_then_original_events_identical_backup_verified(self):
        original = self.tmp / "acurix.db"
        before, legacy_audit = self.make_v1(original)

        # 1) COPY first (online backup API = consistent copy), original untouched.
        copy = self.tmp / "copy" / "acurix.db"
        copy.parent.mkdir()
        src, dst = sqlite3.connect(original), sqlite3.connect(copy)
        src.backup(dst)
        src.close(), dst.close()
        sc = self.migrate(copy)
        self.assertEqual(sc.schema_version(), 2)
        self.assertEqual(sc.check(), (True, "schema 2"))
        self.assertEqual(event_rows(copy), before)
        self.assertEqual(user_version(original), 1)  # the original was not touched by the rehearsal
        self.assertEqual(len(list(copy.parent.glob("acurix.db.pre-v2-*.bak"))), 1)
        sc.close()

        # 2) Then the original.
        s = self.migrate(original)
        self.assertEqual(s.schema_version(), 2)
        self.assertEqual(s.check(), (True, "schema 2"))
        after = event_rows(original)
        self.assertEqual(after, before)  # nothing lost, nothing altered (all columns)
        self.assertEqual(s.count_events(), 5)
        for e in s.list_events():  # payloads still decode and match their stored SHA-256
            self.assertIsNotNone(e.event_id)
        tables = {r[0] for r in s._c().execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertTrue(MIGRATION_2_TABLES <= tables and EXPECTED_TABLES <= tables)
        self.assertEqual(s._c().execute("PRAGMA foreign_key_check").fetchall(), [])
        self.assertTrue(s.integrity_check())

        # Automatic backup: exactly one, verified, still v1 with the same events.
        backups = list(self.tmp.glob("acurix.db.pre-v2-*.bak"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(user_version(backups[0]), 1)
        self.assertEqual(event_rows(backups[0]), before)

        # Audit chain: legacy prefix kept as is (NULL hashes) and still verifiable.
        rep = s.verify_audit_chain()
        self.assertTrue(rep.ok, rep.to_dict())
        self.assertEqual((rep.legacy_entries, rep.chained_entries), (legacy_audit, 0))

    def test_reopen_is_idempotent_no_second_backup(self):
        path = self.tmp / "a.db"
        self.make_v1(path)
        self.migrate(path).close()
        self.migrate(path).close()
        self.assertEqual(len(list(self.tmp.glob("a.db.pre-v2-*.bak"))), 1)

    def test_failure_inside_migration_2_rolls_back_to_v1_intact(self):
        path = self.tmp / "a.db"
        before, _ = self.make_v1(path)

        class Broken(SqliteEventStore):
            MIGRATIONS = {**_MIGRATIONS, 2: _MIGRATIONS[2][:-1] + ("NOT SQL",)}

        with self.assertRaises(PersistenceError):
            Broken(path).open()
        self.assertEqual(user_version(path), 1)
        self.assertEqual(event_rows(path), before)
        c = sqlite3.connect(path)
        tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        cols = {r[1] for r in c.execute("PRAGMA table_info(audit_entry)")}
        c.close()
        self.assertFalse(MIGRATION_2_TABLES & tables)
        self.assertNotIn("prev_sha256", cols)


class AuditChainTests(TempDirTestCase):
    def store_with_chain(self, legacy: int = 2, chained: int = 4) -> SqliteEventStore:
        path = self.tmp / "c.db"
        s = V1Store(path)
        s.open()
        for i in range(legacy):
            s._c().execute("INSERT INTO audit_entry (occurred_at_utc, actor_ref, action, target, detail_json,"
                           " correlation_id) VALUES (?,?,?,?,?,NULL)",
                           (f"2026-11-02T00:00:0{i}.000Z", "system", "LEGACY", "x", "{}"))
        s.close()
        s = SqliteEventStore(path)
        s.open()
        self.addCleanup(s.close)
        for i in range(chained):
            s.audit(actor_ref="tester", action="ACT", target=f"t{i}", detail={"i": i}, correlation_id="c")
        return s

    def rows(self, s):
        return s._c().execute("SELECT audit_id, prev_sha256, entry_sha256 FROM audit_entry ORDER BY audit_id").fetchall()

    def test_chain_links_legacy_prefix_and_new_entries(self):
        s = self.store_with_chain()
        rep = s.verify_audit_chain()
        self.assertTrue(rep.ok, rep.to_dict())
        self.assertEqual((rep.entries, rep.legacy_entries, rep.chained_entries), (6, 2, 4))
        rows = self.rows(s)
        self.assertIsNone(rows[0][1])  # legacy rows are never rewritten
        # First chained entry commits to the legacy prefix (recomputed with the same formula).
        p = AUDIT_GENESIS_SHA256
        for audit_id, occurred, actor, action, target, detail, corr in s._c().execute(
                "SELECT audit_id, occurred_at_utc, actor_ref, action, target, detail_json, correlation_id"
                " FROM audit_entry ORDER BY audit_id LIMIT 2"):
            p = audit_entry_sha256(p, occurred, actor, action, target, detail, corr)
        self.assertEqual(rows[2][1], p)
        for prev_row, row in zip(rows[2:], rows[3:]):
            self.assertEqual(row[1], prev_row[2])
        self.assertEqual(rep.tip_sha256, rows[-1][2])

    def test_fresh_database_chain_starts_at_genesis(self):
        s = SqliteEventStore(self.tmp / "f.db")
        s.open()
        self.addCleanup(s.close)
        s.audit(actor_ref="a", action="X", target="y")
        self.assertEqual(self.rows(s)[0][1], AUDIT_GENESIS_SHA256)
        self.assertTrue(s.verify_audit_chain().ok)

    def test_tampering_is_detected(self):
        cases = {
            "altered chained": ("UPDATE audit_entry SET target='evil' WHERE audit_id=4", 4, "altered"),
            "altered legacy": ("UPDATE audit_entry SET actor_ref='evil' WHERE audit_id=1", 3, "previous"),
            "deleted middle": ("DELETE FROM audit_entry WHERE audit_id=4", 5, "previous"),
            "hashes stripped": ("UPDATE audit_entry SET prev_sha256=NULL, entry_sha256=NULL WHERE audit_id=4", 4,
                                "unchained"),
        }
        for name, (sql, bad_id, word) in cases.items():
            with self.subTest(name):
                s = self.store_with_chain()
                s._c().execute(sql)
                rep = s.verify_audit_chain()
                self.assertFalse(rep.ok)
                self.assertEqual(rep.first_bad_audit_id, bad_id)
                self.assertIn(word, rep.problem)
                s.close()
                for f in self.tmp.glob("c.db*"):
                    f.unlink()

    def test_append_refuses_to_extend_a_broken_legacy_prefix(self):
        path = self.tmp / "l.db"
        s = V1Store(path)
        s.open()
        s._c().execute("INSERT INTO audit_entry (occurred_at_utc, actor_ref, action, target, detail_json)"
                       " VALUES (?,?,?,?,?)", ("t", "a", "b", "c", "{}"))
        s.close()
        s = SqliteEventStore(path)
        s.open()
        self.addCleanup(s.close)
        s._c().execute("UPDATE audit_entry SET detail_json = X'00'")  # BLOB: chain cannot be computed
        with self.assertRaises(PersistenceError):
            s.audit(actor_ref="a", action="X", target="y")
        self.assertEqual(s._c().execute("SELECT count(*) FROM audit_entry").fetchone()[0], 1)

    def test_app_audit_trail_is_chained_and_verifiable(self):
        a = Application(AppConfig(), self.tmp, model_policy=FIXTURE_POLICY, inference=FixtureOkAdapter(),
                        rule=FixtureApprovedRule(), capture_probe=lambda: (C.OK, "fixture"),
                        host_checks=HEALTHY_HOST)
        a.boot()
        self.addCleanup(a.shutdown)
        a.start_session()
        a.process(request())
        a.close_session()
        rep = a.store.verify_audit_chain()
        self.assertTrue(rep.ok, rep.to_dict())
        self.assertGreaterEqual(rep.chained_entries, 3)
        self.assertEqual(rep.legacy_entries, 0)


class ReviewImmutabilityTests(TempDirTestCase):
    def test_review_row_never_alters_inspection_event_and_fk_enforced(self):
        s = SqliteEventStore(self.tmp / "r.db")
        s.open()
        self.addCleanup(s.close)
        e = env()
        s.append(e)
        before = event_rows(self.tmp / "r.db")
        c = s._c()
        c.execute("INSERT INTO user_account (user_id, username, role, password_hash, created_at_utc,"
                  " updated_at_utc) VALUES ('u1','rev','reviewer','scrypt$x','2026-11-10T00:00:00.000Z',"
                  "'2026-11-10T00:00:00.000Z')")
        c.execute("INSERT INTO review (review_id, event_id, review_result, justification, reviewer_id,"
                  " reviewed_at_utc, idempotency_key) VALUES ('r1', ?, 'DEFECT', 'visual', 'u1',"
                  " '2026-11-10T00:00:01.000Z', 'k1')", (e.event_id,))
        self.assertEqual(event_rows(self.tmp / "r.db"), before)
        with self.assertRaises(sqlite3.IntegrityError):  # FK: unknown event
            c.execute("INSERT INTO review (review_id, event_id, review_result, justification, reviewer_id,"
                      " reviewed_at_utc, idempotency_key) VALUES ('r2', 'nope', 'PASS', 'j', 'u1', 't', 'k2')")
        with self.assertRaises(sqlite3.IntegrityError):  # same idempotency key
            c.execute("INSERT INTO review (review_id, event_id, review_result, justification, reviewer_id,"
                      " reviewed_at_utc, idempotency_key) VALUES ('r3', ?, 'PASS', 'j', 'u1', 't', 'k1')",
                      (e.event_id,))
        with self.assertRaises(sqlite3.IntegrityError):  # CHECK on review_result
            c.execute("INSERT INTO review (review_id, event_id, review_result, justification, reviewer_id,"
                      " reviewed_at_utc, idempotency_key) VALUES ('r4', ?, 'MAYBE', 'j', 'u1', 't', 'k4')",
                      (e.event_id,))

    def test_store_has_no_event_update_path(self):
        import inspect

        from acurix_edge.persistence import sqlite_store

        src = inspect.getsource(sqlite_store)
        self.assertNotRegex(src, r"(?i)UPDATE\s+inspection_event")
        self.assertNotRegex(src, r"(?i)DELETE\s+FROM\s+inspection_event")


class ClockRuleTests(TempDirTestCase):
    """Owner decision 10/11/2026: before last event -> FAULT; any other clock problem -> DEGRADED
    and no OBSERVING. Code AEX-CLK-001."""

    def app(self, clock_now):
        store = SqliteEventStore(AppConfig().database_path(self.tmp))
        clock = ClockCheck(now=clock_now, last_event_utc=store.last_event_persisted_utc)
        a = Application(AppConfig(), self.tmp, model_policy=FIXTURE_POLICY, inference=FixtureOkAdapter(),
                        rule=FixtureApprovedRule(), capture_probe=lambda: (C.OK, "fixture"), store=store,
                        host_checks={**HEALTHY_HOST, "clock": clock})
        a.boot()
        self.addCleanup(a.shutdown)
        return a

    def clock_entry(self, a):
        return next(c for c in a.status()["components"] if c["component"] == "clock")

    def test_clock_before_last_persisted_event_is_fault_latched(self):
        offset = [timedelta(0)]
        a = self.app(lambda: datetime.now(timezone.utc) + offset[0])
        a.start_session()
        self.assertIs(a.state.state, SystemState.OBSERVING)
        self.assertEqual(a.process(request()).result, "PASS")  # fixture: all pre-conditions met
        offset[0] = -timedelta(hours=1)  # clock goes back in time
        e = self.clock_entry(a)
        self.assertEqual((e["status"], e["code"], e["critical"]), ("FAULT", "AEX-CLK-001", True))
        self.assertIs(a.state.state, SystemState.FAULT)
        self.assertTrue(a.status()["fault_latched"])
        with self.assertRaises(SessionError) as ctx:
            a.process(request(inspection_id="after"))
        self.assertIs(ctx.exception.code, ErrorCode.STATE_002)
        # Reset refused while the clock is still behind the last event.
        a.close_session()
        with self.assertRaises(SessionError):
            a.reset_fault("clock fixed?", actor_ref="admin")
        offset[0] = timedelta(0)
        a.reset_fault("clock corrected", actor_ref="admin")
        self.assertIs(a.state.state, SystemState.IDLE)

    def test_fault_survives_restart_via_persisted_last_event(self):
        a = self.app(lambda: datetime.now(timezone.utc))
        a.start_session()
        a.process(request())
        a.shutdown()
        b = self.app(lambda: datetime.now(timezone.utc) - timedelta(days=1))  # booted with a wrong clock
        with self.assertRaises(SessionError):  # start re-evaluates health: clock FAULT -> refused
            b.start_session()
        self.assertIs(b.state.state, SystemState.FAULT)
        self.assertEqual(self.clock_entry(b)["code"], "AEX-CLK-001")

    def test_other_clock_problems_degrade_and_block_observing(self):
        cases = {
            "before floor": lambda: datetime(2020, 1, 1, tzinfo=timezone.utc),
            "unreadable": lambda: (_ for _ in ()).throw(OSError("rtc")),
        }
        for name, now in cases.items():
            with self.subTest(name):
                a = self.app(now)
                e = self.clock_entry(a)
                self.assertEqual((e["status"], e["code"]), ("DEGRADED", "AEX-CLK-001"))
                a.start_session()
                self.assertIs(a.state.state, SystemState.DEGRADED)  # never OBSERVING
                self.assertEqual(a.process(request()).result, "NO_DECISION")
                self.assertIs(a.state.state, SystemState.DEGRADED)  # refresh keeps it out of OBSERVING
                a.shutdown()
                for f in (self.tmp / ".data").glob("*"):
                    f.unlink()

    def test_backward_jump_without_events_is_degraded_until_caught_up(self):
        t = [datetime.now(timezone.utc)]
        a = self.app(lambda: t[0])
        self.assertEqual(self.clock_entry(a)["status"], "OK")
        t[0] -= timedelta(minutes=5)
        a.start_session()
        self.assertIs(a.state.state, SystemState.DEGRADED)
        t[0] += timedelta(minutes=10)
        a.refresh_health()
        self.assertIs(a.state.state, SystemState.OBSERVING)


if __name__ == "__main__":
    unittest.main()
