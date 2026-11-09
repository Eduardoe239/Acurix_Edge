"""Local persistence adapter (SQLite, ADR-002 direction / ADR-014)."""

from acurix_edge.persistence.sqlite_store import (
    SCHEMA_VERSION,
    AppendOutcome,
    IdempotencyConflict,
    PersistenceError,
    SqliteEventStore,
)

__all__ = ["SCHEMA_VERSION", "AppendOutcome", "IdempotencyConflict", "PersistenceError", "SqliteEventStore"]
