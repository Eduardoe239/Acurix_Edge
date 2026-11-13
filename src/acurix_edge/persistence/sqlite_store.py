"""SQLite event store: versioned migrations, idempotent append, audit trail.

- Idempotency: re-appending the same event_id with identical payload returns
  DUPLICATE; a different payload under the same id raises IdempotencyConflict
  (event_id is immutable, Vol.02 §6.2).
- Any storage failure is surfaced as PersistenceError; callers must never
  report an event as persisted when append() did not return.
- Before migrating an existing database, a consistent backup is written.
- Conclusive results (PASS/DEFECT) are rejected by a CHECK constraint unless
  they carry model sha256, decision rule id/version and inference status OK.
- `read_only=True` opens without creating, migrating or writing (status probe).
- Error messages never include absolute local paths.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import os
import sqlite3
import threading
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

from acurix_edge.contracts.errors import ContractError, ErrorCode
from acurix_edge.contracts.events import EventEnvelope, utc_now_iso

SCHEMA_VERSION = 3  # 2 = Anexo 02A §2 (APPROVED 10/11/2026); 3 = immutability trigger (owner, Ordem 2.1)
MAX_PAGE = 1000
EXPECTED_TABLES = frozenset({
    "inspection_event", "audit_entry",
    # migration 2
    "user_account", "login_session", "inspection_session", "cascade_step", "review", "media_asset",
    "system_incident", "export_record",
})
EXPECTED_EVENT_COLUMNS = frozenset({
    "event_id", "session_id", "inspection_id", "capture_sequence", "result", "inference_status",
    "rule_id", "rule_version", "model_id", "model_version", "model_sha256", "payload_sha256",
})
EXPECTED_AUDIT_COLUMNS = frozenset({"audit_id", "detail_json", "prev_sha256", "entry_sha256"})

# Audit hash chain (AUD-002). entry_sha256 = SHA-256 of the canonical JSON array
# [AUDIT_CHAIN_SCHEME, prev_sha256, occurred_at_utc, actor_ref, action, target, detail_json, correlation_id].
# The first entry's prev is AUDIT_GENESIS_SHA256. Entries written before migration 2 keep NULL hash
# columns (never rewritten); their hash is computed on the fly with the same formula, so the first
# chained entry's prev_sha256 commits to the whole legacy prefix.
AUDIT_CHAIN_SCHEME = "aex-audit-chain-v1"
AUDIT_GENESIS_SHA256 = "0" * 64

_MIGRATIONS: dict[int, tuple[str, ...]] = {
    1: (
        """CREATE TABLE inspection_event (
            event_id          TEXT PRIMARY KEY,
            schema_version    TEXT NOT NULL,
            site_id           TEXT NOT NULL,
            station_id        TEXT NOT NULL,
            session_id        TEXT NOT NULL,
            inspection_id     TEXT NOT NULL,
            captured_at_utc   TEXT NOT NULL,
            capture_sequence  INTEGER NOT NULL CHECK (capture_sequence >= 0),
            result            TEXT NOT NULL CHECK (result IN ('PASS','DEFECT','REVIEW','NO_DECISION')),
            reason            TEXT NOT NULL,
            inference_status  TEXT NOT NULL,
            rule_id           TEXT,
            rule_version      TEXT,
            class_id          TEXT,
            confidence        REAL,
            model_id          TEXT,
            model_version     TEXT,
            model_sha256      TEXT,
            config_version    TEXT NOT NULL,
            config_sha256     TEXT NOT NULL,
            review_status     TEXT NOT NULL,
            software_version  TEXT NOT NULL,
            payload_json      TEXT NOT NULL,
            payload_sha256    TEXT NOT NULL,
            persisted_at_utc  TEXT NOT NULL,
            UNIQUE (session_id, inspection_id),
            UNIQUE (session_id, capture_sequence),
            CHECK (result NOT IN ('PASS','DEFECT') OR (
                model_id IS NOT NULL AND model_version IS NOT NULL AND model_sha256 IS NOT NULL
                AND rule_id IS NOT NULL AND rule_version IS NOT NULL
                AND inference_status = 'OK'))
        )""",
        "CREATE INDEX ix_event_session_time ON inspection_event (session_id, captured_at_utc)",
        "CREATE INDEX ix_event_result ON inspection_event (result)",
        """CREATE TABLE audit_entry (
            audit_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            occurred_at_utc  TEXT NOT NULL,
            actor_ref        TEXT NOT NULL,
            action           TEXT NOT NULL,
            target           TEXT NOT NULL,
            detail_json      TEXT NOT NULL,
            correlation_id   TEXT
        )""",
        "CREATE INDEX ix_audit_time ON audit_entry (occurred_at_utc)",
    ),
    # Migration 2 — Anexo 02A §2, APPROVED 10/11/2026. Transcribed verbatim (statements and
    # columns unchanged; only SQL comments dropped). Never edit: a change is migration 3.
    2: (
        """CREATE TABLE user_account (
  user_id         TEXT PRIMARY KEY,
  username        TEXT NOT NULL UNIQUE,
  display_name    TEXT,
  role            TEXT NOT NULL CHECK (role IN ('operator','reviewer','admin','support')),
  password_hash   TEXT NOT NULL,
  active          INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0,1)),
  failed_logins   INTEGER NOT NULL DEFAULT 0,
  locked_until_utc TEXT,
  created_at_utc  TEXT NOT NULL,
  updated_at_utc  TEXT NOT NULL
)""",
        """CREATE TABLE login_session (
  token_sha256    TEXT PRIMARY KEY,
  user_id         TEXT NOT NULL REFERENCES user_account(user_id),
  created_at_utc  TEXT NOT NULL,
  last_seen_utc   TEXT NOT NULL,
  reauth_until_utc TEXT,
  revoked         INTEGER NOT NULL DEFAULT 0
)""",
        """CREATE TABLE inspection_session (
  session_id      TEXT PRIMARY KEY,
  state           TEXT NOT NULL CHECK (state IN ('ACTIVE','CLOSED','INTERRUPTED')),
  started_at_utc  TEXT NOT NULL,
  ended_at_utc    TEXT,
  started_by      TEXT NOT NULL REFERENCES user_account(user_id),
  config_version  TEXT NOT NULL,
  config_sha256   TEXT NOT NULL,
  report_json     TEXT
)""",
        """CREATE TABLE cascade_step (
  event_id        TEXT NOT NULL REFERENCES inspection_event(event_id),
  step_index      INTEGER NOT NULL CHECK (step_index >= 0),
  stage           TEXT NOT NULL CHECK (stage IN ('quality_gate','anomaly','router','specialist')),
  status          TEXT NOT NULL,
  model_id        TEXT,
  model_version   TEXT,
  model_sha256    TEXT,
  result          TEXT,
  confidence      REAL CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
  duration_ms     REAL NOT NULL CHECK (duration_ms >= 0),
  PRIMARY KEY (event_id, step_index)
)""",
        """CREATE TABLE review (
  review_id       TEXT PRIMARY KEY,
  event_id        TEXT NOT NULL REFERENCES inspection_event(event_id),
  review_result   TEXT NOT NULL CHECK (review_result IN ('CONFIRM','PASS','DEFECT','UNCLEAR')),
  class_id        TEXT,
  justification   TEXT NOT NULL,
  reviewer_id     TEXT NOT NULL REFERENCES user_account(user_id),
  reviewed_at_utc TEXT NOT NULL,
  idempotency_key TEXT NOT NULL UNIQUE
)""",
        """CREATE TABLE media_asset (
  asset_id        TEXT PRIMARY KEY,
  event_id        TEXT NOT NULL REFERENCES inspection_event(event_id),
  storage_ref     TEXT NOT NULL,
  sha256          TEXT NOT NULL,
  bytes           INTEGER NOT NULL,
  retention_until_utc TEXT NOT NULL,
  deleted_at_utc  TEXT
)""",
        """CREATE TABLE system_incident (
  incident_id     TEXT PRIMARY KEY,
  code            TEXT NOT NULL,
  severity        TEXT NOT NULL CHECK (severity IN ('S1','S2','S3','S4')),
  opened_at_utc   TEXT NOT NULL,
  closed_at_utc   TEXT,
  detail_json     TEXT NOT NULL
)""",
        """CREATE TABLE export_record (
  export_id       TEXT PRIMARY KEY,
  requested_by    TEXT NOT NULL REFERENCES user_account(user_id),
  filters_json    TEXT NOT NULL,
  include_media   INTEGER NOT NULL CHECK (include_media IN (0,1)),
  rows            INTEGER NOT NULL,
  sha256          TEXT NOT NULL,
  created_at_utc  TEXT NOT NULL
)""",
        "CREATE INDEX ix_review_event   ON review (event_id)",
        "CREATE INDEX ix_media_retention ON media_asset (retention_until_utc) WHERE deleted_at_utc IS NULL",
        "CREATE INDEX ix_incident_open  ON system_incident (opened_at_utc) WHERE closed_at_utc IS NULL",
        "ALTER TABLE audit_entry ADD COLUMN prev_sha256 TEXT",
        "ALTER TABLE audit_entry ADD COLUMN entry_sha256 TEXT",
    ),
    # Migration 3 — owner decision 10/11/2026 (Ordem 2.1): inspection_event rows are immutable.
    # UPDATE is aborted by the database itself; DELETE is deliberately NOT blocked (future
    # retention purge stays controlled and audited by code). To be documented in Anexo 02A.
    3: (
        """CREATE TRIGGER trg_inspection_event_no_update BEFORE UPDATE ON inspection_event
BEGIN
  SELECT RAISE(ABORT, 'AEX-DB: inspection_event is immutable; UPDATE forbidden (migration 3). Record reviews in table review.');
END""",  # noqa: E501 (trigger DDL text must stay byte-identical)
    ),
}
IMMUTABILITY_TRIGGER = "trg_inspection_event_no_update"
# Known limit (review R3 M1): `INSERT OR REPLACE` / `REPLACE INTO` is DELETE+INSERT for SQLite and
# does not fire a BEFORE UPDATE trigger. Product code never uses it on inspection_event; a static
# test (tests/unit/test_migration_003.py) forbids it in src/. UPSERT `ON CONFLICT DO UPDATE` and
# `UPDATE OR REPLACE` DO fire the trigger. Closing REPLACE at database level requires a DELETE
# guard, which the owner deliberately left open for the future retention purge.


def _norm_sql(sql: str | None) -> str:
    return " ".join((sql or "").split())


def audit_entry_sha256(prev_sha256: str, occurred_at_utc: str, actor_ref: str, action: str, target: str,
                       detail_json: str, correlation_id: str | None) -> str:
    canonical = json.dumps([AUDIT_CHAIN_SCHEME, prev_sha256, occurred_at_utc, actor_ref, action, target,
                            detail_json, correlation_id], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AuditChainReport:
    """Result of verify_audit_chain(). `ok` False => `first_bad_audit_id` and `problem` set."""

    __slots__ = ("ok", "entries", "legacy_entries", "chained_entries", "tip_sha256", "first_bad_audit_id",
                 "problem")

    def __init__(self) -> None:
        self.ok = True
        self.entries = self.legacy_entries = self.chained_entries = 0
        self.tip_sha256 = AUDIT_GENESIS_SHA256
        self.first_bad_audit_id: int | None = None
        self.problem: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


class PersistenceError(RuntimeError):
    """Storage unavailable, locked, full, corrupted or otherwise failing.

    `code`: AEX-DB-001 (unavailable/locked/missing/corrupt) or AEX-DB-002 (disk full)."""

    default_code = ErrorCode.DB_001

    def __init__(self, message: str, code: ErrorCode | None = None) -> None:
        super().__init__(message)
        self.code = code or self.default_code


class IdempotencyConflict(PersistenceError):
    """Same event_id re-submitted with a different payload."""

    default_code = ErrorCode.API_004


class CorruptRecordError(PersistenceError):
    """A stored event payload is undecodable, violates the contract or fails its SHA-256 (R4-03)."""


_SQLITE_FULL = 13


def classify_storage_error(exc: BaseException) -> ErrorCode:
    """AEX-DB-002 when the failure is lack of space; AEX-DB-001 otherwise."""
    if isinstance(exc, sqlite3.Error):
        if (getattr(exc, "sqlite_errorcode", 0) or 0) & 0xFF == _SQLITE_FULL or "disk is full" in str(exc):
            return ErrorCode.DB_002
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        return ErrorCode.DB_002
    return ErrorCode.DB_001


INVALID_BACKUP_SUFFIX = ".invalid"


class AppendOutcome(StrEnum):
    INSERTED = "INSERTED"
    DUPLICATE = "DUPLICATE"


def _validated_target(migrations: dict[int, tuple[str, ...]]) -> int:
    versions = sorted(migrations)
    if not versions or versions != list(range(1, len(versions) + 1)):
        raise PersistenceError(f"migration registry must be contiguous from 1, got {versions}")
    return versions[-1]


class SqliteEventStore:
    # Versioned migration registry. Migration N runs once, inside one transaction, after an
    # automatic verified backup of any existing database. Shipped migrations are never edited:
    # a schema change is a NEW entry (Anexo 02A migration 2 will be key 2 once approved).
    MIGRATIONS: ClassVar[dict[int, tuple[str, ...]]] = _MIGRATIONS

    def __init__(self, db_path: Path, *, timeout_s: float = 5.0, read_only: bool = False) -> None:
        self._path = Path(db_path)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._timeout_s = timeout_s
        self.read_only = read_only
        # Latest persisted_at_utc of inspection_event (clock-regression check, AEX-CLK-001).
        # Loaded once at open, then advanced by append(): no table scan per health check.
        self._last_event_persisted_utc: str | None = None

    def _err(self, prefix: str, exc: BaseException) -> PersistenceError:
        code = classify_storage_error(exc)
        msg = f"{code}: {prefix}: {type(exc).__name__}: {exc}"
        for p in (str(self._path), str(self._path.parent), self._path.as_posix(), self._path.parent.as_posix()):
            msg = msg.replace(p, "<data_dir>")
        return PersistenceError(msg, code)

    # -- lifecycle -----------------------------------------------------------
    def open(self) -> None:
        with self._lock:
            if self._conn is not None:
                raise PersistenceError("database already open")
            try:
                if self.read_only:
                    if not self._path.is_file():
                        raise PersistenceError("database not initialized (run 'init')")
                    conn = sqlite3.connect(f"{self._path.resolve().as_uri()}?mode=ro", uri=True,
                                           timeout=self._timeout_s, isolation_level=None, check_same_thread=False)
                    self._conn = conn
                    conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
                    self._load_last_event_time()
                    return
                self._path.parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(self._path, timeout=self._timeout_s, isolation_level=None,
                                       check_same_thread=False)
                self._conn = conn
                conn.execute("PRAGMA foreign_keys = ON")
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = FULL")
                self._migrate()
                self._load_last_event_time()
            except PersistenceError:
                self.close()
                raise
            except Exception as exc:
                self.close()
                raise self._err("cannot open database", exc) from exc

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None

    @property
    def is_open(self) -> bool:
        return self._conn is not None

    def _c(self) -> sqlite3.Connection:
        if self._conn is None:
            raise PersistenceError("database is not open")
        return self._conn

    def _require_writable(self) -> None:
        if self.read_only:
            raise PersistenceError("database opened read-only")

    def _load_last_event_time(self) -> None:
        # A missing/malformed inspection_event leaves None: check() reports that schema as FAULT.
        try:
            row = self._c().execute("SELECT max(persisted_at_utc) FROM inspection_event").fetchone()
            self._last_event_persisted_utc = row[0] if row and isinstance(row[0], str) else None
        except sqlite3.OperationalError:
            self._last_event_persisted_utc = None

    def last_event_persisted_utc(self) -> str | None:
        """Latest `persisted_at_utc` ever written to inspection_event (None: no event)."""
        with self._lock:
            return self._last_event_persisted_utc

    # -- migrations ----------------------------------------------------------
    def schema_version(self) -> int:
        with self._lock:
            try:
                return int(self._c().execute("PRAGMA user_version").fetchone()[0])
            except sqlite3.Error as exc:
                raise self._err("schema version unavailable", exc) from exc

    @property
    def target_version(self) -> int:
        return _validated_target(self.MIGRATIONS)

    def _migrate(self) -> None:
        with self._lock:
            conn = self._c()
            target = self.target_version
            current = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if current > target:
                raise PersistenceError(f"database schema {current} is newer than supported {target}")
            if current == target:
                return
            has_tables = conn.execute("SELECT count(*) FROM sqlite_master WHERE type='table'").fetchone()[0] > 0
            if has_tables:
                stamp = utc_now_iso().replace(":", "").replace("-", "").replace(".", "")
                self.backup(self._path.with_name(f"{self._path.name}.pre-v{current + 1}-{stamp}.bak"))
            for version in range(current + 1, target + 1):
                conn.execute("BEGIN IMMEDIATE")
                try:
                    # Re-read under the write lock: a concurrent 'init' may have migrated already.
                    if int(conn.execute("PRAGMA user_version").fetchone()[0]) >= version:
                        conn.execute("ROLLBACK")
                        continue
                    for stmt in self.MIGRATIONS[version]:
                        conn.execute(stmt)
                    conn.execute(f"PRAGMA user_version = {int(version)}")
                    conn.execute("COMMIT")
                except BaseException:
                    if conn.in_transaction:
                        conn.execute("ROLLBACK")
                    raise

    # -- backup / restore ------------------------------------------------------
    # Safety rules (R4-06, review R2-H1, Ordem 2.1 B1-B6):
    # - the destination is created EXCLUSIVELY; an existing file is never opened or overwritten;
    # - quarantine only ever touches the file this call created, identified by (st_dev, st_ino);
    # - quarantine runs on ANY exception after creation, including KeyboardInterrupt/MemoryError;
    # - the quarantine name is unique (UTC timestamp + counter): an older `.invalid` is never replaced;
    # - leftover journal side files of the destination are refused before, and moved with it after.

    def backup(self, dest: Path) -> Path:
        """Write and verify a backup into a NEW file. A copy that fails quick_check (or is left
        partial by any error) is quarantined as `<name>.<UTC>.invalid` so it can never be mistaken
        for a valid backup. Nothing in this package restores backups automatically."""
        dest = Path(dest)
        with self._lock:
            identity: tuple[int, int] | None = None
            try:
                self._refuse_side_files(dest)
                identity = self._create_exclusive(dest, "backup")
                self._refuse_side_files(dest)  # re-checked after creation (review R3 B-f / R4-03)
                target = sqlite3.connect(dest)
                try:
                    self._c().backup(target)
                    # Self-contained file: no WAL header, so opening the copy never leaves -wal/-shm.
                    self._set_delete_journal(target)
                    verdict = target.execute("PRAGMA quick_check").fetchone()[0]
                finally:
                    target.close()
                if verdict != "ok":
                    raise PersistenceError("backup verification failed (quick_check)")
            except BaseException as exc:
                marked = self._quarantine_backup(dest, identity) if identity is not None else "target untouched"
                if isinstance(exc, PersistenceError):
                    raise PersistenceError(f"{exc}; {marked}", exc.code) from exc
                if isinstance(exc, (sqlite3.Error, OSError)):
                    raise self._err(f"backup failed ({marked})", exc) from exc
                raise
            return dest

    @classmethod
    def restore_backup(cls, backup: Path, dest: Path) -> Path:
        """Restore a backup into a NEW database file and verify it.

        Non-destructive by design: refuses an existing `dest` (never overwrites the live
        database), refuses backups marked `.invalid` (any case) and a `dest` with leftover
        journal side files. Swapping the restored file into service is an operator action
        outside this package (runbook)."""
        backup, dest = Path(backup), Path(dest)
        if backup.name.lower().endswith(INVALID_BACKUP_SUFFIX) or not backup.is_file():
            raise PersistenceError(f"backup {backup.name} missing or marked invalid")
        identity: tuple[int, int] | None = None  # set only once THIS call created dest
        try:
            cls._refuse_side_files(dest)
            src = sqlite3.connect(f"{backup.resolve().as_uri()}?mode=ro", uri=True)
            try:
                if src.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise PersistenceError(f"backup {backup.name} fails quick_check")
                # Review R3 B-d: an empty/foreign SQLite file (e.g. a 0-byte leftover) is a "valid"
                # database for quick_check; it is NOT an Acurix backup.
                version = int(src.execute("PRAGMA user_version").fetchone()[0])
                has_events = src.execute("SELECT count(*) FROM sqlite_master WHERE type='table'"
                                         " AND name='inspection_event'").fetchone()[0]
                if version < 1 or not has_events:
                    raise PersistenceError(f"backup {backup.name} is not an Acurix database "
                                           f"(user_version {version}, "
                                           f"inspection_event {'present' if has_events else 'absent'})")
                identity = cls._create_exclusive(dest, "restore")
                cls._refuse_side_files(dest)  # review R3 B-f: re-checked after creation
                target = sqlite3.connect(dest)
                try:
                    src.backup(target)
                    cls._set_delete_journal(target)  # self-contained (see backup())
                    verdict = target.execute("PRAGMA quick_check").fetchone()[0]
                finally:
                    target.close()
            finally:
                src.close()
            if verdict != "ok":
                raise PersistenceError("restored copy fails quick_check")
        except BaseException as exc:
            marked = cls._quarantine_backup(dest, identity) if identity is not None else "target untouched"
            if isinstance(exc, PersistenceError):
                raise PersistenceError(f"{exc}; {marked}", exc.code) from exc
            if isinstance(exc, (sqlite3.Error, OSError)):
                raise PersistenceError(f"restore failed ({marked}): {type(exc).__name__}",
                                       classify_storage_error(exc)) from exc
            raise  # KeyboardInterrupt, MemoryError, ...: after quarantine, propagate unchanged
        return dest

    _SIDE_SUFFIXES = ("-wal", "-shm", "-journal")

    @staticmethod
    def _set_delete_journal(conn: sqlite3.Connection) -> None:
        # Review R3 B-e: the PRAGMA does not raise when the switch is refused; check its answer.
        mode = conn.execute("PRAGMA journal_mode = DELETE").fetchone()[0]
        if str(mode).lower() != "delete":
            raise PersistenceError(f"copy could not leave journal mode {mode!r} (expected 'delete')")

    @classmethod
    def _refuse_side_files(cls, dest: Path) -> None:
        stale = [dest.name + s for s in cls._SIDE_SUFFIXES if dest.with_name(dest.name + s).exists()]
        if stale:
            raise PersistenceError(f"target has leftover journal files {stale}; refusing (they could be "
                                   "replayed into the new file)")

    @staticmethod
    def _create_exclusive(dest: Path, what: str) -> tuple[int, int]:
        """Create `dest` atomically (O_EXCL); return its identity. Existing file -> refused, untouched."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(dest, "xb") as fh:
                st = os.fstat(fh.fileno())
        except FileExistsError:
            raise PersistenceError(f"{what} target already exists; refusing to overwrite") from None
        return st.st_dev, st.st_ino

    @classmethod
    def _quarantine_backup(cls, dest: Path, identity: tuple[int, int] | None = None) -> str:
        """Move `dest` (and its journal side files) to a UNIQUE `<name>.<UTC>[-n].invalid`.
        With `identity`, a file that is no longer the one this call created is left untouched."""
        try:
            st = dest.stat()
        except FileNotFoundError:
            return "no backup file left"
        except OSError as exc:
            return f"WARNING: {dest.name} not inspectable ({type(exc).__name__}); left in place"
        if identity is not None and (st.st_dev, st.st_ino) != identity:
            return f"WARNING: {dest.name} was replaced by another file; left untouched"
        stamp = utc_now_iso().replace(":", "").replace("-", "").replace(".", "")
        for n in range(1000):
            stem = f"{dest.name}.{stamp}" + (f"-{n}" if n else "")
            invalid = dest.with_name(stem + INVALID_BACKUP_SUFFIX)
            if invalid.exists() or any(dest.with_name(stem + INVALID_BACKUP_SUFFIX + s).exists()
                                       for s in cls._SIDE_SUFFIXES):
                continue
            try:
                # os.rename never replaces an existing target on Windows (FileExistsError).
                os.rename(dest, invalid)
            except FileExistsError:
                continue
            except OSError:
                # Review R3 B-c: delete only if it is STILL the file this call created.
                try:
                    st2 = dest.stat()
                    if identity is not None and (st2.st_dev, st2.st_ino) != identity:
                        return f"WARNING: {dest.name} was replaced by another file; left untouched"
                    dest.unlink()
                    sides = [dest.name + s for s in cls._SIDE_SUFFIXES if dest.with_name(dest.name + s).exists()]
                    return "invalid backup deleted (rename failed)" + (
                        f"; WARNING: side files left {sides} (future restores to this name are refused)"
                        if sides else "")
                except OSError:
                    return f"WARNING: invalid backup {dest.name} could not be marked or deleted"
            for s in cls._SIDE_SUFFIXES:
                side = dest.with_name(dest.name + s)
                if side.exists():
                    with contextlib.suppress(OSError):
                        os.rename(side, invalid.with_name(invalid.name + s))
            return f"marked {invalid.name}"
        return f"WARNING: no free quarantine name for {dest.name}; left in place"

    # -- health --------------------------------------------------------------
    def check(self) -> tuple[bool, str]:
        """Light health probe: connection, schema version and expected tables (no full scan)."""
        try:
            with self._lock:
                conn = self._c()
                conn.execute("SELECT 1").fetchone()
                v = int(conn.execute("PRAGMA user_version").fetchone()[0])
                tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if v != self.target_version:
                return False, f"schema {v} != {self.target_version}"
            missing = EXPECTED_TABLES - tables
            if missing:
                return False, f"missing tables {sorted(missing)}"
            for table, expected in (("inspection_event", EXPECTED_EVENT_COLUMNS),
                                    ("audit_entry", EXPECTED_AUDIT_COLUMNS)):
                with self._lock:
                    cols = {r[1] for r in self._c().execute(f"PRAGMA table_info({table})")}
                missing_cols = expected - cols
                if missing_cols:
                    return False, f"{table} schema mismatch, missing {sorted(missing_cols)}"
            if self.target_version >= 3:
                with self._lock:
                    trig = self._c().execute("SELECT tbl_name, sql FROM sqlite_master WHERE type='trigger' AND name=?",
                                             (IMMUTABILITY_TRIGGER,)).fetchone()
                if not trig:
                    return False, f"immutability trigger {IMMUTABILITY_TRIGGER} missing"
                # Review R3 B-a: same name is not enough; the definition must be the shipped one.
                if trig[0] != "inspection_event" or _norm_sql(trig[1]) != _norm_sql(_MIGRATIONS[3][0]):
                    return False, f"immutability trigger {IMMUTABILITY_TRIGGER} altered"
            return True, f"schema {v}" + (" (read-only)" if self.read_only else "")
        except PersistenceError as exc:
            return False, str(exc)
        except sqlite3.Error as exc:
            return False, str(self._err("check failed", exc))

    def integrity_check(self) -> bool:
        with self._lock:
            try:
                return bool(self._c().execute("PRAGMA quick_check").fetchone()[0] == "ok")
            except sqlite3.Error as exc:
                raise self._err("integrity check failed", exc) from exc

    # -- events --------------------------------------------------------------
    def append(self, env: EventEnvelope) -> AppendOutcome:
        self._require_writable()
        try:
            payload = env.canonical_json()
            digest = env.payload_sha256()
            model = env.model or {}
            rule = env.rule or {}
            persisted_at = utc_now_iso()
            params = (
                env.event_id, env.schema_version, env.site_id, env.station_id, env.session_id,
                env.inspection_id, env.captured_at_utc, env.capture_sequence, env.result.value, env.reason,
                env.inference_status,
                rule.get("id"), rule.get("version"), env.class_id, env.confidence, model.get("id"),
                model.get("version"), model.get("sha256"), env.configuration["version"],
                env.configuration["sha256"], env.review_status.value, env.software_version, payload, digest,
                persisted_at,
            )
        except Exception as exc:
            raise self._err("cannot serialize event", exc) from exc
        with self._lock:
            conn = self._c()
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT payload_sha256 FROM inspection_event WHERE event_id = ?", (env.event_id,)
                ).fetchone()
                if row is not None:
                    conn.execute("ROLLBACK")
                    if row[0] == digest:
                        return AppendOutcome.DUPLICATE
                    raise IdempotencyConflict(f"event_id {env.event_id} already persisted with different payload")
                conn.execute(
                    """INSERT INTO inspection_event (
                        event_id, schema_version, site_id, station_id, session_id, inspection_id, captured_at_utc,
                        capture_sequence, result, reason, inference_status, rule_id, rule_version, class_id,
                        confidence, model_id, model_version, model_sha256, config_version, config_sha256,
                        review_status, software_version, payload_json, payload_sha256, persisted_at_utc
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    params,
                )
                conn.execute("COMMIT")
                if self._last_event_persisted_utc is None or persisted_at > self._last_event_persisted_utc:
                    self._last_event_persisted_utc = persisted_at
                return AppendOutcome.INSERTED
            except IdempotencyConflict:
                raise
            except Exception as exc:
                try:
                    if conn.in_transaction:
                        conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise self._err("append failed", exc) from exc

    @staticmethod
    def _decode(row: tuple[Any, ...]) -> EventEnvelope:
        """Decode a stored payload; any defect becomes an explicit CorruptRecordError (never a
        crash with a raw JSON/contract exception, never a silently trusted record)."""
        event_id, payload, digest = row
        try:
            if not isinstance(payload, str):
                raise TypeError("payload_json is not text")
            if hashlib.sha256(payload.encode("utf-8")).hexdigest() != digest:
                raise ValueError("payload_sha256 mismatch")
            env = EventEnvelope.from_dict(json.loads(payload))
            if env.event_id != event_id:
                raise ValueError("event_id column does not match payload")
            return env
        except (ValueError, TypeError, KeyError, ContractError, UnicodeError) as exc:
            raise CorruptRecordError(
                f"stored event {event_id!s} is corrupt: {type(exc).__name__}: {str(exc)[:200]}") from exc

    _SELECT_EVENT = "SELECT event_id, payload_json, payload_sha256 FROM inspection_event"

    def get(self, event_id: str) -> EventEnvelope | None:
        with self._lock:
            try:
                row = self._c().execute(self._SELECT_EVENT + " WHERE event_id = ?", (event_id,)).fetchone()
            except sqlite3.Error as exc:
                raise self._err("read failed", exc) from exc
        return self._decode(row) if row else None

    def get_by_inspection(self, session_id: str, inspection_id: str) -> EventEnvelope | None:
        with self._lock:
            try:
                row = self._c().execute(
                    self._SELECT_EVENT + " WHERE session_id = ? AND inspection_id = ?",
                    (session_id, inspection_id),
                ).fetchone()
            except sqlite3.Error as exc:
                raise self._err("read failed", exc) from exc
        return self._decode(row) if row else None

    def list_events(self, *, session_id: str | None = None, limit: int = 100, offset: int = 0) -> list[EventEnvelope]:
        if not 1 <= limit <= MAX_PAGE or offset < 0:
            raise ValueError(f"limit must be 1..{MAX_PAGE} and offset >= 0")
        sql = self._SELECT_EVENT
        args: list[Any] = []
        if session_id is not None:
            sql += " WHERE session_id = ?"
            args.append(session_id)
        sql += " ORDER BY captured_at_utc, capture_sequence, event_id LIMIT ? OFFSET ?"
        args += [limit, offset]
        with self._lock:
            try:
                rows = self._c().execute(sql, args).fetchall()
            except sqlite3.Error as exc:
                raise self._err("read failed", exc) from exc
        return [self._decode(r) for r in rows]

    def count_events(self) -> int:
        with self._lock:
            try:
                return int(self._c().execute("SELECT count(*) FROM inspection_event").fetchone()[0])
            except sqlite3.Error as exc:
                raise self._err("read failed", exc) from exc

    # -- audit ---------------------------------------------------------------
    def audit(self, *, actor_ref: str, action: str, target: str, detail: dict[str, Any] | None = None,
              correlation_id: str | None = None) -> None:
        from acurix_edge.observability.logging import redact

        self._require_writable()
        occurred = utc_now_iso()
        detail_json = json.dumps(redact(detail or {}), sort_keys=True, default=str)
        with self._lock:
            conn = self._c()
            try:
                # Read the chain tip and insert under one write lock: no fork of the chain.
                conn.execute("BEGIN IMMEDIATE")
                prev = self._audit_tip(conn)
                entry = audit_entry_sha256(prev, occurred, actor_ref, action, target, detail_json, correlation_id)
                conn.execute(
                    "INSERT INTO audit_entry (occurred_at_utc, actor_ref, action, target, detail_json, correlation_id,"
                    " prev_sha256, entry_sha256) VALUES (?,?,?,?,?,?,?,?)",
                    (occurred, actor_ref, action, target, detail_json, correlation_id, prev, entry),
                )
                conn.execute("COMMIT")
            except Exception as exc:
                try:
                    if conn.in_transaction:
                        conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                if isinstance(exc, PersistenceError):
                    raise
                raise self._err("audit write failed", exc) from exc

    _AUDIT_CHAIN_COLUMNS = ("SELECT audit_id, occurred_at_utc, actor_ref, action, target, detail_json, correlation_id,"
                            " prev_sha256, entry_sha256 FROM audit_entry")

    def _audit_tip(self, conn: sqlite3.Connection) -> str:
        """Hash of the last audit entry. A chained tip is trusted as stored (verify_audit_chain()
        checks it); a legacy (pre-migration-2) tip is recomputed over the legacy prefix once."""
        row = conn.execute("SELECT entry_sha256 FROM audit_entry ORDER BY audit_id DESC LIMIT 1").fetchone()
        if row is None:
            return AUDIT_GENESIS_SHA256
        if row[0] is not None:
            return str(row[0])
        report = self._walk_audit_chain(conn)
        if not report.ok:
            raise PersistenceError(f"audit chain broken at audit_id {report.first_bad_audit_id}: {report.problem}")
        return report.tip_sha256

    @staticmethod
    def _walk_audit_chain(conn: sqlite3.Connection) -> AuditChainReport:
        rep = AuditChainReport()
        prev = AUDIT_GENESIS_SHA256
        for audit_id, occurred, actor, action, target, detail_json, corr, stored_prev, stored_entry in conn.execute(
                SqliteEventStore._AUDIT_CHAIN_COLUMNS + " ORDER BY audit_id"):
            rep.entries += 1
            problem: str | None = None
            if not all(isinstance(v, str) for v in (occurred, actor, action, target, detail_json)) or (
                    corr is not None and not isinstance(corr, str)):
                problem = "non-text audit field"
                computed = ""
            else:
                computed = audit_entry_sha256(prev, occurred, actor, action, target, detail_json, corr)
            if problem is None:
                if stored_prev is None and stored_entry is None:
                    if rep.chained_entries:
                        problem = "unchained entry after the chain started (hashes removed or row inserted)"
                    rep.legacy_entries += 1
                else:
                    rep.chained_entries += 1
                    if stored_prev != prev:
                        problem = "prev_sha256 does not match the previous entry (entry removed or reordered)"
                    elif stored_entry != computed:
                        problem = "entry_sha256 does not match the entry content (entry altered)"
            if problem is not None:
                rep.ok, rep.first_bad_audit_id, rep.problem = False, int(audit_id), problem
                return rep
            prev = computed
        rep.tip_sha256 = prev
        return rep

    def verify_audit_chain(self) -> AuditChainReport:
        """Full scan of audit_entry recomputing every hash (AUD-002). Detects altered, removed
        (not at the tail), reordered or inserted entries. Does NOT detect truncation of the newest
        entries or a full recomputation by someone with write access: keep the reported
        `tip_sha256` outside the database to anchor it."""
        with self._lock:
            try:
                return self._walk_audit_chain(self._c())
            except sqlite3.Error as exc:
                raise self._err("audit chain read failed", exc) from exc

    def last_state_audit(self) -> dict[str, Any] | None:
        """Most recent accepted state transition or fault reset (FAULT latch source, R4-01)."""
        with self._lock:
            try:
                row = self._c().execute(
                    "SELECT action, target, detail_json FROM audit_entry"
                    " WHERE action IN ('STATE_TRANSITION', 'FAULT_RESET') ORDER BY audit_id DESC LIMIT 1"
                ).fetchone()
            except sqlite3.Error as exc:
                raise self._err("read failed", exc) from exc
        if row is None:
            return None
        try:
            detail = json.loads(row[2])
        except ValueError:
            detail = {"undecodable": True}
        return {"action": row[0], "target": row[1], "detail": detail}

    def last_audit_with_action(self, actions: tuple[str, ...], target: str | None = None) -> dict[str, Any] | None:
        """Most recent audit entry whose action is one of `actions` (and, if given, whose target
        equals `target`). Read-only query."""
        if not actions or not all(isinstance(a, str) and a for a in actions):
            raise ValueError("actions must be non-empty strings")
        marks = ",".join("?" for _ in actions)
        where = f"action IN ({marks})" + (" AND target = ?" if target is not None else "")
        params: tuple[str, ...] = actions + ((target,) if target is not None else ())
        with self._lock:
            try:
                row = self._c().execute(
                    f"SELECT action, target, detail_json FROM audit_entry WHERE {where}"
                    " ORDER BY audit_id DESC LIMIT 1", params).fetchone()
            except sqlite3.Error as exc:
                raise self._err("read failed", exc) from exc
        if row is None:
            return None
        try:
            detail = json.loads(row[2])
        except ValueError:
            detail = {"undecodable": True}
        return {"action": row[0], "target": row[1], "detail": detail}

    def list_audit(self, limit: int = 100) -> list[dict[str, Any]]:
        if not 1 <= limit <= MAX_PAGE:
            raise ValueError(f"limit must be 1..{MAX_PAGE}")
        with self._lock:
            try:
                rows = self._c().execute(
                    "SELECT audit_id, occurred_at_utc, actor_ref, action, target, detail_json, correlation_id"
                    " FROM audit_entry ORDER BY audit_id DESC LIMIT ?", (limit,)
                ).fetchall()
            except sqlite3.Error as exc:
                raise self._err("read failed", exc) from exc
        return [
            {"audit_id": r[0], "occurred_at_utc": r[1], "actor_ref": r[2], "action": r[3], "target": r[4],
             "detail": json.loads(r[5]), "correlation_id": r[6]}
            for r in rows
        ]
