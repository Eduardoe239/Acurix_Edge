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

import errno
import hashlib
import json
import sqlite3
import threading
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

from acurix_edge.contracts.errors import ContractError, ErrorCode
from acurix_edge.contracts.events import EventEnvelope, utc_now_iso

SCHEMA_VERSION = 1  # highest migration shipped in this release (migration 2 of Anexo 02A is PROPOSED, not here)
MAX_PAGE = 1000
EXPECTED_TABLES = frozenset({"inspection_event", "audit_entry"})
EXPECTED_EVENT_COLUMNS = frozenset({
    "event_id", "session_id", "inspection_id", "capture_sequence", "result", "inference_status",
    "rule_id", "rule_version", "model_id", "model_version", "model_sha256", "payload_sha256",
})

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
}


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
                    return
                self._path.parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(self._path, timeout=self._timeout_s, isolation_level=None, check_same_thread=False)
                self._conn = conn
                conn.execute("PRAGMA foreign_keys = ON")
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = FULL")
                self._migrate()
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

    def backup(self, dest: Path) -> Path:
        """Write and verify a backup. A copy that fails quick_check (or is left partial by an
        error) is renamed to `<name>.invalid` so it can never be mistaken for a valid backup
        (R4-06). Nothing in this package restores backups automatically."""
        with self._lock:
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                target = sqlite3.connect(dest)
                try:
                    self._c().backup(target)
                    verdict = target.execute("PRAGMA quick_check").fetchone()[0]
                finally:
                    target.close()
            except (sqlite3.Error, OSError, PersistenceError) as exc:
                marked = self._quarantine_backup(dest)
                if isinstance(exc, PersistenceError):
                    raise
                raise self._err(f"backup failed ({marked})", exc) from exc
            if verdict != "ok":
                marked = self._quarantine_backup(dest)
                raise PersistenceError(f"backup verification failed (quick_check); {marked}")
            return dest

    @classmethod
    def restore_backup(cls, backup: Path, dest: Path) -> Path:
        """Restore a backup into a NEW database file and verify it.

        Non-destructive by design: refuses an existing `dest` (never overwrites the live
        database) and refuses backups marked `.invalid`. Swapping the restored file into
        service is an operator action outside this package (runbook)."""
        backup, dest = Path(backup), Path(dest)
        if backup.name.endswith(INVALID_BACKUP_SUFFIX) or not backup.is_file():
            raise PersistenceError(f"backup {backup.name} missing or marked invalid")
        created = False  # quarantine may ONLY touch a file this call created (review R2-H1)
        try:
            src = sqlite3.connect(f"{backup.resolve().as_uri()}?mode=ro", uri=True)
            try:
                if src.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise PersistenceError(f"backup {backup.name} fails quick_check")
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    # Exclusive create (no exists()/connect race): an existing file is never touched.
                    with open(dest, "xb"):
                        pass
                except FileExistsError:
                    raise PersistenceError("restore target already exists; refusing to overwrite") from None
                created = True
                target = sqlite3.connect(dest)
                try:
                    src.backup(target)
                    verdict = target.execute("PRAGMA quick_check").fetchone()[0]
                finally:
                    target.close()
            finally:
                src.close()
        except PersistenceError:
            raise
        except (sqlite3.Error, OSError) as exc:
            marked = cls._quarantine_backup(dest) if created else "target untouched"
            raise PersistenceError(f"restore failed ({marked}): {type(exc).__name__}",
                                   classify_storage_error(exc)) from exc
        if verdict != "ok":
            marked = cls._quarantine_backup(dest)
            raise PersistenceError(f"restored copy fails quick_check; {marked}")
        return dest

    @staticmethod
    def _quarantine_backup(dest: Path) -> str:
        if not dest.exists():
            return "no backup file left"
        invalid = dest.with_name(dest.name + INVALID_BACKUP_SUFFIX)
        try:
            dest.replace(invalid)
            return f"marked {invalid.name}"
        except OSError:
            try:
                dest.unlink()
                return "invalid backup deleted (rename failed)"
            except OSError:
                return f"WARNING: invalid backup {dest.name} could not be marked or deleted"

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
            with self._lock:
                cols = {r[1] for r in self._c().execute("PRAGMA table_info(inspection_event)")}
            missing_cols = EXPECTED_EVENT_COLUMNS - cols
            if missing_cols:
                return False, f"inspection_event schema mismatch, missing {sorted(missing_cols)}"
            return True, f"schema {v}" + (" (read-only)" if self.read_only else "")
        except PersistenceError as exc:
            return False, str(exc)
        except sqlite3.Error as exc:
            return False, str(self._err("check failed", exc))

    def integrity_check(self) -> bool:
        with self._lock:
            try:
                return self._c().execute("PRAGMA quick_check").fetchone()[0] == "ok"
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
            params = (
                env.event_id, env.schema_version, env.site_id, env.station_id, env.session_id,
                env.inspection_id, env.captured_at_utc, env.capture_sequence, env.result.value, env.reason, env.inference_status,
                rule.get("id"), rule.get("version"), env.class_id, env.confidence, model.get("id"),
                model.get("version"), model.get("sha256"), env.configuration["version"],
                env.configuration["sha256"], env.review_status.value, env.software_version, payload, digest,
                utc_now_iso(),
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
        with self._lock:
            try:
                self._c().execute(
                    "INSERT INTO audit_entry (occurred_at_utc, actor_ref, action, target, detail_json, correlation_id)"
                    " VALUES (?,?,?,?,?,?)",
                    (utc_now_iso(), actor_ref, action, target,
                     json.dumps(redact(detail or {}), sort_keys=True, default=str), correlation_id),
                )
            except sqlite3.Error as exc:
                raise self._err("audit write failed", exc) from exc

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
