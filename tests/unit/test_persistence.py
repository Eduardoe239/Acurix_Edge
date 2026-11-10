import sqlite3

from acurix_edge.contracts.decision import InspectionResult
from acurix_edge.contracts.events import EventEnvelope, ReviewStatus
from acurix_edge.persistence.sqlite_store import (
    SCHEMA_VERSION,
    AppendOutcome,
    IdempotencyConflict,
    PersistenceError,
    SqliteEventStore,
)
from tests.support import TempDirTestCase

CFG = {"version": "0.1.0", "sha256": "a" * 64}


def env(seq: int = 1, event_id: str | None = None, result=InspectionResult.NO_DECISION, session="s-1",
        inspection_id: str | None = None) -> EventEnvelope:
    return EventEnvelope(
        event_id=event_id or EventEnvelope.new_id(), site_id="site", station_id="st", session_id=session,
        inspection_id=inspection_id or ("insp-" + EventEnvelope.new_id()),
        captured_at_utc=f"2026-11-03T12:00:{seq:02d}.000Z", capture_sequence=seq, result=result,
        reason="NO_APPROVED_RULE", inference_status="NO_MODEL", model=None, configuration=CFG, software_version="0.1.0",
        review_status=ReviewStatus.NOT_REQUIRED, latency_ms=0.5, request_sha256="d" * 64,
    )


class PersistenceTests(TempDirTestCase):
    def store(self, name="db.sqlite") -> SqliteEventStore:
        s = SqliteEventStore(self.tmp / "data" / name)
        s.open()
        self.addCleanup(s.close)
        return s

    def test_fresh_migration(self):
        s = self.store()
        self.assertEqual(s.schema_version(), SCHEMA_VERSION)
        self.assertEqual(s.check(), (True, f"schema {SCHEMA_VERSION}"))
        self.assertTrue(s.integrity_check())

    def test_append_get_roundtrip(self):
        s = self.store()
        e = env()
        self.assertIs(s.append(e), AppendOutcome.INSERTED)
        self.assertEqual(s.get(e.event_id), e)
        self.assertIsNone(s.get(EventEnvelope.new_id()))

    def test_idempotent_duplicate_and_conflict(self):
        s = self.store()
        e = env()
        s.append(e)
        self.assertIs(s.append(e), AppendOutcome.DUPLICATE)
        self.assertEqual(s.count_events(), 1)
        with self.assertRaises(IdempotencyConflict):
            s.append(env(seq=2, event_id=e.event_id))
        self.assertEqual(s.count_events(), 1)
        self.assertEqual(s.get(e.event_id), e)  # original not overwritten

    def test_persists_across_restart(self):
        s = self.store()
        e = env()
        s.append(e)
        s.close()
        s2 = self.store()
        self.assertEqual(s2.get(e.event_id), e)
        self.assertEqual(s2.schema_version(), SCHEMA_VERSION)

    def test_listing_order_filter_and_limits(self):
        s = self.store()
        for i in (3, 1, 2):
            s.append(env(seq=i))
        s.append(env(seq=4, session="other"))
        seqs = [e.capture_sequence for e in s.list_events(session_id="s-1")]
        self.assertEqual(seqs, [1, 2, 3])
        self.assertEqual(len(s.list_events(limit=2)), 2)
        for bad in ({"limit": 0}, {"limit": 1001}, {"offset": -1}):
            with self.assertRaises(ValueError):
                s.list_events(**bad)

    def test_db_unavailable_raises_persistence_error(self):
        blocker = self.tmp / "data"
        blocker.mkdir()
        (blocker / "db.sqlite").mkdir()  # a directory where the db file should be
        with self.assertRaises(PersistenceError):
            SqliteEventStore(blocker / "db.sqlite").open()

    def test_operations_after_close_raise(self):
        s = self.store()
        s.close()
        with self.assertRaises(PersistenceError):
            s.append(env())
        self.assertFalse(s.check()[0])

    def test_newer_schema_rejected_and_handle_released(self):
        path = self.tmp / "data" / "new.sqlite"
        path.parent.mkdir(parents=True)
        c = sqlite3.connect(path)
        c.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
        c.close()
        s = SqliteEventStore(path)
        with self.assertRaises(PersistenceError):
            s.open()
        self.assertFalse(s.is_open)
        path.unlink()  # would fail on Windows if the handle leaked
        self.assertFalse(path.exists())

    def test_double_open_rejected(self):
        s = self.store()
        with self.assertRaises(PersistenceError):
            s.open()
        self.assertTrue(s.check()[0])

    def test_failing_migration_rolls_back(self):
        from acurix_edge.persistence import sqlite_store as mod

        original = mod._MIGRATIONS[1]
        mod._MIGRATIONS[1] = original + ("CREATE TABLE broken (",)
        self.addCleanup(mod._MIGRATIONS.__setitem__, 1, original)
        path = self.tmp / "data" / "fail.sqlite"
        s = SqliteEventStore(path)
        with self.assertRaises(PersistenceError):
            s.open()
        self.assertFalse(s.is_open)
        c = sqlite3.connect(path)
        self.assertEqual(c.execute("PRAGMA user_version").fetchone()[0], 0)
        self.assertEqual(c.execute("SELECT count(*) FROM sqlite_master WHERE type='table'").fetchone()[0], 0)
        c.close()

    def test_locked_database_append_fails_cleanly(self):
        s = SqliteEventStore(self.tmp / "data" / "lock.sqlite", timeout_s=0.1)
        s.open()
        self.addCleanup(s.close)
        other = sqlite3.connect(self.tmp / "data" / "lock.sqlite", isolation_level=None)
        other.execute("BEGIN EXCLUSIVE")
        try:
            with self.assertRaises(PersistenceError):
                s.append(env())
            self.assertFalse(s._c().in_transaction)
        finally:
            other.execute("ROLLBACK")
            other.close()
        self.assertEqual(s.count_events(), 0)
        self.assertIs(s.append(env()), AppendOutcome.INSERTED)

    def test_check_detects_missing_table(self):
        s = self.store()
        s._c().execute("DROP TABLE audit_entry")
        ok, detail = s.check()
        self.assertFalse(ok)
        self.assertIn("audit_entry", detail)

    def test_read_only_mode(self):
        path = self.tmp / "data" / "ro.sqlite"
        missing = SqliteEventStore(path, read_only=True)
        with self.assertRaises(PersistenceError):
            missing.open()
        self.assertFalse(path.exists())  # read-only never creates
        rw = SqliteEventStore(path)
        rw.open()
        rw.append(env())
        rw.close()
        ro = SqliteEventStore(path, read_only=True)
        ro.open()
        self.addCleanup(ro.close)
        self.assertTrue(ro.check()[0])
        self.assertEqual(ro.count_events(), 1)
        with self.assertRaises(PersistenceError):
            ro.append(env())
        with self.assertRaises(PersistenceError):
            ro.audit(actor_ref="a", action="b", target="c")

    def test_error_messages_hide_absolute_path(self):
        s = SqliteEventStore(self.tmp / "data" / "x.sqlite", read_only=True)
        with self.assertRaises(PersistenceError) as ctx:
            s.open()
        self.assertNotIn(str(self.tmp), str(ctx.exception))

    def _raw_pass_insert(self, s, **override):
        row = dict(event_id="e", schema_version="1.0", site_id="s", station_id="st", session_id="se",
                   inspection_id="i", captured_at_utc="t", capture_sequence=1, result="PASS", reason="r",
                   inference_status="OK", rule_id="r1", rule_version="1", model_id="m", model_version="1",
                   model_sha256="0" * 64, config_version="v", config_sha256="h", review_status="NOT_REQUIRED",
                   software_version="0.1.0", payload_json="{}", payload_sha256="h", persisted_at_utc="t")
        row.update(override)
        cols = ",".join(row)
        s._c().execute(f"INSERT INTO inspection_event ({cols}) VALUES ({','.join('?' * len(row))})", list(row.values()))

    def test_conclusive_result_db_constraint_each_clause_isolated(self):
        s = self.store()
        self._raw_pass_insert(s)  # fully traceable row is accepted
        for i, field in enumerate(("model_id", "model_version", "model_sha256", "rule_id", "rule_version")):
            with self.subTest(field=field):
                with self.assertRaises(sqlite3.IntegrityError):
                    self._raw_pass_insert(s, event_id=f"e{i}", inspection_id=f"i{i}", capture_sequence=10 + i,
                                          **{field: None})
        with self.assertRaises(sqlite3.IntegrityError):
            self._raw_pass_insert(s, event_id="ex", inspection_id="ix", capture_sequence=99, inference_status="NO_MODEL")

    def test_unique_inspection_and_sequence_per_session(self):
        s = self.store()
        s.append(env(seq=1, inspection_id="dup"))
        with self.assertRaises(PersistenceError):
            s.append(env(seq=2, inspection_id="dup"))  # same inspection, new event_id
        with self.assertRaises(PersistenceError):
            s.append(env(seq=1, inspection_id="other"))  # same sequence
        s.append(env(seq=1, inspection_id="dup", session="s-2"))  # other session is fine
        self.assertEqual(s.count_events(), 2)
        self.assertIsNotNone(s.get_by_inspection("s-1", "dup"))
        self.assertIsNone(s.get_by_inspection("s-1", "nope"))

    def test_check_detects_outdated_v1_schema(self):
        path = self.tmp / "data" / "old.sqlite"
        path.parent.mkdir(parents=True)
        c = sqlite3.connect(path)
        c.execute("CREATE TABLE inspection_event (event_id TEXT PRIMARY KEY)")
        c.execute("CREATE TABLE audit_entry (audit_id INTEGER)")
        c.execute("PRAGMA user_version = 1")
        c.commit()
        c.close()
        s = SqliteEventStore(path)
        s.open()
        self.addCleanup(s.close)
        ok, detail = s.check()
        self.assertFalse(ok)
        self.assertIn("schema mismatch", detail)

    def test_concurrent_appends(self):
        import threading

        s = self.store()
        errors = []

        def worker(n):
            try:
                for i in range(10):
                    s.append(env(seq=(n * 10 + i) % 60))
            except Exception as exc:  # pragma: no cover - reported below
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(s.count_events(), 40)

    def test_migration_of_existing_db_creates_backup(self):
        path = self.tmp / "data" / "legacy.sqlite"
        path.parent.mkdir(parents=True)
        c = sqlite3.connect(path)
        c.execute("CREATE TABLE legacy (x INTEGER)")
        c.execute("INSERT INTO legacy VALUES (42)")
        c.commit()
        c.close()
        s = SqliteEventStore(path)
        s.open()
        self.addCleanup(s.close)
        backups = list(path.parent.glob("legacy.sqlite.pre-v1-*.bak"))
        self.assertEqual(len(backups), 1)
        b = sqlite3.connect(backups[0])
        self.assertEqual(b.execute("SELECT x FROM legacy").fetchone()[0], 42)
        b.close()

    def test_backup_and_restore(self):
        s = self.store()
        e = env()
        s.append(e)
        dest = s.backup(self.tmp / "bk" / "copy.sqlite")
        restored = SqliteEventStore(dest)
        restored.open()
        self.addCleanup(restored.close)
        self.assertEqual(restored.get(e.event_id), e)

    def test_result_check_constraint(self):
        s = self.store()
        with self.assertRaises(sqlite3.IntegrityError):
            s._c().execute(
                "INSERT INTO inspection_event (event_id, schema_version, site_id, station_id, session_id,"
                " captured_at_utc, capture_sequence, result, reason, inference_status, config_version,"
                " config_sha256, review_status, software_version, payload_json, payload_sha256, persisted_at_utc)"
                " VALUES ('e','1.0','s','st','se','t',1,'MAYBE','r','OK','v','h','NOT_REQUIRED','0.1.0','{}','h','t')"
            )

    def test_audit_redacts_and_lists(self):
        s = self.store()
        s.audit(actor_ref="system", action="X", target="t", detail={"token": "secret-value", "k": 1})
        entry = s.list_audit()[0]
        self.assertNotIn("secret-value", str(entry))
        self.assertEqual(entry["detail"]["k"], 1)
