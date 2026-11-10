"""AEX-DATA-004 acceptance: EVT-001, EVT-002, NFR-017, AEX-DB-001/002.

Disk full is simulated with SQLite's own limit (PRAGMA max_page_count), which raises the real
SQLITE_FULL error path; nothing fills the host disk.
"""

import io
import json
import logging
import re
import sqlite3
import uuid

from acurix_edge.app import Application, SessionError
from acurix_edge.config.settings import AppConfig, StorageConfig
from acurix_edge.contracts.errors import ErrorCode
from acurix_edge.contracts.inference import InferenceRequest, InputDescriptor
from acurix_edge.domain.state_machine import SystemState
from acurix_edge.observability.logging import configure_logging
from acurix_edge.persistence.sqlite_store import (
    INVALID_BACKUP_SUFFIX,
    AppendOutcome,
    IdempotencyConflict,
    PersistenceError,
    SqliteEventStore,
    _MIGRATIONS,
)
from tests.support import FIXTURE_POLICY, TempDirTestCase, request
from tests.unit.test_persistence import env

UTC_RE = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$")


class StoreCase(TempDirTestCase):
    def store(self, name="a.db", **kw) -> SqliteEventStore:
        s = SqliteEventStore(self.tmp / name, **kw)
        s.open()
        self.addCleanup(s.close)
        return s

    def app(self, **kw) -> Application:
        a = Application(kw.pop("config", AppConfig()), self.tmp, model_policy=FIXTURE_POLICY, **kw)
        a.boot()
        self.addCleanup(a.shutdown)
        return a


class SqliteSettingsTests(StoreCase):
    def test_pragmas_utc_and_uuid_text(self):
        s = self.store()
        conn = s._c()
        self.assertEqual(conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)
        self.assertEqual(conn.execute("PRAGMA journal_mode").fetchone()[0], "wal")
        e = env()
        s.append(e)
        row = conn.execute("SELECT event_id, typeof(event_id), captured_at_utc, persisted_at_utc "
                           "FROM inspection_event").fetchone()
        self.assertEqual(str(uuid.UUID(row[0])), row[0])
        self.assertEqual(row[1], "text")
        self.assertRegex(row[2], UTC_RE)
        self.assertRegex(row[3], UTC_RE)
        s.audit(actor_ref="t", action="X", target="y")
        self.assertRegex(s.list_audit(1)[0]["occurred_at_utc"], UTC_RE)


class RestartAndIdempotencyTests(StoreCase):
    def test_events_persist_after_restart(self):
        a = self.app()
        sid = a.start_session()
        before = [a.process(request(inspection_id=f"r{i}")).record.event_id for i in range(3)]
        a.shutdown()
        b = self.app()
        after = b.store.list_events(session_id=sid)
        self.assertEqual([e.event_id for e in after], before)
        self.assertTrue(all(e.schema_version == "1.1" and e.latency_ms is not None for e in after))

    def test_identical_resubmission_does_not_duplicate(self):
        a = self.app()
        a.start_session()
        first = a.process(request(inspection_id="dup"))
        again = a.process(request(inspection_id="dup"))
        self.assertEqual(again.record.outcome, AppendOutcome.DUPLICATE)
        self.assertEqual(again.record.event_id, first.record.event_id)
        self.assertEqual(a.store.count_events(), 1)

    def test_same_inspection_id_different_content_is_recorded_conflict(self):
        a = self.app()
        a.start_session()
        first = a.process(request(inspection_id="same"))
        other = InferenceRequest(inspection_id="same", input=InputDescriptor(640, 480))
        o = a.process(other)
        self.assertEqual((o.result, o.reason, o.code), ("NO_DECISION", "IDEMPOTENCY_CONFLICT", ErrorCode.API_004))
        self.assertFalse(o.record.persisted)
        self.assertEqual(o.record.event_id, first.record.event_id)
        self.assertEqual(a.store.count_events(), 1)  # nothing overwritten, nothing added
        self.assertEqual(a.events.counters().inspection_conflicts, 1)
        audit = next(r for r in a.store.list_audit() if r["action"] == "IDEMPOTENCY_CONFLICT")
        self.assertEqual(audit["detail"]["code"], "AEX-API-004")
        # different pixels are also a different content
        b1 = InferenceRequest(inspection_id="px", input=InputDescriptor(2, 2), frame=b"\x00\x01")
        b2 = InferenceRequest(inspection_id="px", input=InputDescriptor(2, 2), frame=b"\x00\x02")
        self.assertNotEqual(b1.sha256(), b2.sha256())

    def test_buffer_frames_are_fingerprinted_and_opaque_frames_never_replay(self):  # review R1-M1
        import array

        a1 = InferenceRequest(inspection_id="arr", input=InputDescriptor(2, 2), frame=array.array("B", [0, 1, 2, 3]))
        a2 = InferenceRequest(inspection_id="arr", input=InputDescriptor(2, 2), frame=array.array("B", [0, 1, 2, 4]))
        self.assertNotEqual(a1.sha256(), a2.sha256())
        self.assertEqual(a1.sha256(), InferenceRequest(inspection_id="arr", input=InputDescriptor(2, 2),
                                                       frame=array.array("B", [0, 1, 2, 3])).sha256())
        d = lambda v: InferenceRequest(inspection_id="d", input=InputDescriptor(2, 1),  # noqa: E731
                                       frame=array.array("d", v)).sha256()
        self.assertEqual(d([1.0, 2.0]), d([1.0, 2.0]))  # non-byte format: deterministic, not random
        self.assertNotEqual(d([1.0, 2.0]), d([1.0, 2.5]))
        a = self.app()
        a.start_session()
        opaque = InferenceRequest(inspection_id="opaque", input=InputDescriptor(2, 2), frame=object())
        a.process(opaque)
        again = a.process(opaque)  # content cannot be compared -> conflict, never a replayed result
        self.assertEqual((again.result, again.code), ("NO_DECISION", ErrorCode.API_004))

    def test_latency_excludes_health_evaluation(self):  # review R1-M2
        import time

        from acurix_edge.health.service import ComponentStatus

        def slow_gpu():
            time.sleep(0.3)
            return ComponentStatus.DISABLED, "slow fixture"

        a = self.app(host_checks={"gpu": slow_gpu})
        a.start_session()
        o = a.process(request(inspection_id="lat"))
        self.assertLess(a.store.get(o.record.event_id).latency_ms, 300.0)

    def test_store_level_event_id_conflict_code(self):
        s = self.store()
        e = env()
        s.append(e)
        with self.assertRaises(IdempotencyConflict) as ctx:
            s.append(env(event_id=e.event_id, seq=2))
        self.assertIs(ctx.exception.code, ErrorCode.API_004)


class BackupRestoreTests(StoreCase):
    def test_backup_and_restore_round_trip(self):
        s = self.store()
        events = [env(seq=i) for i in range(1, 4)]
        for e in events:
            s.append(e)
        bak = s.backup(self.tmp / "bk" / "a.db.bak")
        restored = SqliteEventStore.restore_backup(bak, self.tmp / "restored" / "a.db")
        r = SqliteEventStore(restored)
        r.open()
        self.addCleanup(r.close)
        self.assertEqual(r.check()[0], True)
        self.assertEqual([x.event_id for x in r.list_events()], [e.event_id for e in events])
        self.assertEqual(r.list_events()[0].payload_sha256(), events[0].payload_sha256())

    def test_restore_never_overwrites_and_refuses_invalid(self):
        s = self.store()
        s.append(env())
        bak = s.backup(self.tmp / "a.db.bak")
        live = self.tmp / "a.db"
        size = live.stat().st_size
        with self.assertRaises(PersistenceError):
            SqliteEventStore.restore_backup(bak, live)
        self.assertEqual(live.stat().st_size, size)
        invalid = self.tmp / ("x.bak" + INVALID_BACKUP_SUFFIX)
        invalid.write_bytes(bak.read_bytes())
        with self.assertRaises(PersistenceError):
            SqliteEventStore.restore_backup(invalid, self.tmp / "new.db")
        garbage = self.tmp / "garbage.bak"
        garbage.write_bytes(b"not a database" * 100)
        with self.assertRaises(PersistenceError):
            SqliteEventStore.restore_backup(garbage, self.tmp / "new2.db")
        # review R2-H1: corrupt backup + EXISTING live database -> live file untouched, never quarantined.
        # The store is closed first: on Windows an open file cannot be renamed, which would hide the bug.
        s.close()
        with self.assertRaises(PersistenceError):
            SqliteEventStore.restore_backup(garbage, live)
        self.assertTrue(live.is_file())
        self.assertFalse(live.with_name(live.name + INVALID_BACKUP_SUFFIX).exists())
        self.assertEqual(self.store("a.db").count_events(), 1)


class MigrationMechanismTests(StoreCase):
    """Proves the mechanism is ready for migration 2 using a TEST-ONLY migration (not Anexo 02A)."""

    def v1_with_events(self) -> list[str]:
        s = SqliteEventStore(self.tmp / "m.db")
        s.open()
        ids = []
        for i in range(1, 3):
            e = env(seq=i)
            s.append(e)
            ids.append(e.event_id)
        s.close()
        return ids

    def test_v2_applied_in_transaction_after_backup_and_events_intact(self):
        ids = self.v1_with_events()

        class V2(SqliteEventStore):
            MIGRATIONS = {**_MIGRATIONS, 2: ("CREATE TABLE test_only_m2 (x INTEGER)",)}

        s = V2(self.tmp / "m.db")
        s.open()
        self.addCleanup(s.close)
        self.assertEqual(s.schema_version(), 2)
        self.assertEqual(s.check(), (True, "schema 2"))
        self.assertEqual([e.event_id for e in s.list_events()], ids)
        self.assertEqual(len(list(self.tmp.glob("m.db.pre-v2-*.bak"))), 1)

    def test_failed_migration_rolls_back_and_keeps_data(self):
        ids = self.v1_with_events()

        class Broken(SqliteEventStore):
            MIGRATIONS = {**_MIGRATIONS, 2: ("CREATE TABLE test_only_m2 (x INTEGER)", "THIS IS NOT SQL")}

        with self.assertRaises(PersistenceError) as ctx:
            Broken(self.tmp / "m.db").open()
        self.assertIs(ctx.exception.code, ErrorCode.DB_001)
        s = self.store("m.db")  # shipped registry (v1) opens the untouched database
        self.assertEqual(s.schema_version(), 1)
        tables = {r[0] for r in s._c().execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertNotIn("test_only_m2", tables)
        self.assertEqual([e.event_id for e in s.list_events()], ids)
        self.assertEqual(len(list(self.tmp.glob("m.db.pre-v2-*.bak"))), 1)

    def test_non_contiguous_registry_rejected(self):
        class Gap(SqliteEventStore):
            MIGRATIONS = {1: _MIGRATIONS[1], 3: ("SELECT 1",)}

        with self.assertRaises(PersistenceError):
            Gap(self.tmp / "g.db").open()


class StorageFailureTests(StoreCase):
    def test_locked_database_is_db_001_and_no_decision(self):
        a = self.app()
        a.start_session()
        a.store._timeout_s = 0.1
        a.store._c().execute("PRAGMA busy_timeout = 100")
        blocker = sqlite3.connect(self.tmp / ".data" / "acurix.db", isolation_level=None)
        self.addCleanup(blocker.close)
        blocker.execute("BEGIN EXCLUSIVE")
        try:
            o = a.process(request(inspection_id="locked"))
        finally:
            blocker.execute("ROLLBACK")
        self.assertEqual((o.result, o.code), ("NO_DECISION", ErrorCode.DB_001))
        self.assertFalse(o.record.persisted)
        self.assertIs(a.state.state, SystemState.FAULT)

    def test_missing_database_is_fault(self):
        (self.tmp / "blocked").write_text("not a dir")
        a = self.app(config=AppConfig(storage=StorageConfig(data_dir="blocked")))
        self.assertIs(a.state.state, SystemState.FAULT)
        with self.assertRaises(SessionError) as ctx:
            a.start_session()
        self.assertIs(ctx.exception.code, ErrorCode.STATE_002)
        ro = SqliteEventStore(self.tmp / "nope.db", read_only=True)
        with self.assertRaises(PersistenceError) as ctx2:
            ro.open()
        self.assertIs(ctx2.exception.code, ErrorCode.DB_001)

    def fill_to_limit(self, store: SqliteEventStore) -> None:
        conn = store._c()
        pages = conn.execute("PRAGMA page_count").fetchone()[0]
        conn.execute(f"PRAGMA max_page_count = {pages + 1}")

    def test_disk_full_store_level_is_db_002(self):
        s = self.store()
        self.fill_to_limit(s)
        with self.assertRaises(PersistenceError) as ctx:
            for i in range(1, 200):
                s.append(env(seq=i))
        self.assertIs(ctx.exception.code, ErrorCode.DB_002)
        self.assertIn("AEX-DB-002", str(ctx.exception))

    def test_disk_full_app_level_no_decision_and_logs_with_correlation(self):
        stream = io.StringIO()
        configure_logging("INFO", stream)
        self.addCleanup(lambda: configure_logging("WARNING", io.StringIO()))
        a = self.app()
        sid = a.start_session()
        self.fill_to_limit(a.store)
        outcomes = [a.process(request(inspection_id=f"full-{i}")) for i in range(50)
                    if a.state.state is not SystemState.FAULT]
        last = outcomes[-1]
        self.assertEqual((last.result, last.code), ("NO_DECISION", ErrorCode.DB_002))
        self.assertIs(a.state.state, SystemState.FAULT)
        lines = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
        failure = next(rec for rec in lines if rec.get("code") == "AEX-DB-002")
        self.assertEqual(failure["session_id"], sid)
        self.assertEqual(failure["correlation_id"], f"full-{len(outcomes) - 1}")  # the failing inspection
        text = stream.getvalue()
        self.assertNotIn(str(self.tmp), text)  # no absolute local paths
        self.assertNotIn(self.tmp.as_posix(), text)


if __name__ == "__main__":
    import unittest

    unittest.main()
