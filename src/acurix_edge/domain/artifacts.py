"""Verified loading of inspection artifacts (CAS-005, ADR-016) — generic, model-format agnostic.

Before ANY artifact (future model file or rule-parameter file) is used:
1. it must be listed in the inventory (`models.artifacts`) by exact id + version  -> else NOT_LISTED;
2. the file must exist inside the artifacts folder                              -> else MISSING;
3. the SHA-256 of its bytes must equal the inventory sha256                     -> else HASH_MISMATCH.
Any refusal raises ArtifactRejected, reports an incident (log + `on_incident`, wired to the audit
trail) and the stage stays unavailable (NO_DECISION, never PASS).

APPROVAL IS SEPARATE (owner decision 13/11/2026): the inventory is inventory only. A verified
artifact carries `approved`, decided EXCLUSIVELY by ModelPolicy.is_model_approved (approved_models,
locked empty by ADR-016). Being listed — or passing the hash check — never makes it approved.
Part 2 (model runtime) plugs its loader on top of `ArtifactVerifier.load()`.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from acurix_edge.contracts.errors import ErrorCode
from acurix_edge.contracts.inference import ModelRef
from acurix_edge.domain.model_policy import ModelPolicy

log = logging.getLogger("acurix_edge.artifacts")

PACKAGE_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


class RejectReason(StrEnum):
    NOT_LISTED = "NOT_LISTED"
    MISSING = "MISSING"
    HASH_MISMATCH = "HASH_MISMATCH"


REASON_CODES = {RejectReason.NOT_LISTED: ErrorCode.CFG_002, RejectReason.MISSING: ErrorCode.CAS_003,
                RejectReason.HASH_MISMATCH: ErrorCode.CAS_003}


class ArtifactRejected(RuntimeError):
    def __init__(self, artifact_id: str, version: str, reason: RejectReason, detail: str) -> None:
        self.artifact_id, self.version, self.reason = artifact_id, version, reason
        self.code = REASON_CODES[reason]
        super().__init__(f"{self.code}: artifact {artifact_id}@{version} rejected ({reason}): {detail}")


@dataclass(frozen=True, slots=True)
class InventoryEntry:
    ref: ModelRef
    path: str
    status: str  # always NOT_APPROVED in 0.3.0 (configuration schema)


@dataclass(frozen=True, slots=True)
class VerifiedArtifact:
    ref: ModelRef
    data: bytes
    approved: bool  # from ModelPolicy ONLY; never from the inventory


Incident = Callable[[ArtifactRejected], None]


class ArtifactVerifier:
    def __init__(self, inventory: Iterable[Mapping[str, Any]], policy: ModelPolicy, *,
                 root: Path = PACKAGE_ARTIFACTS_DIR, on_incident: Incident | None = None) -> None:
        self._entries: dict[tuple[str, str], InventoryEntry] = {}
        for raw in inventory:
            ref = ModelRef(id=str(raw["id"]), version=str(raw["version"]), sha256=str(raw["sha256"]))
            self._entries[(ref.id, ref.version)] = InventoryEntry(ref, str(raw["path"]), str(raw["status"]))
        self._policy, self._root, self._on_incident = policy, root, on_incident

    def entry(self, artifact_id: str, version: str) -> InventoryEntry | None:
        return self._entries.get((artifact_id, version))

    def _reject(self, artifact_id: str, version: str, reason: RejectReason, detail: str) -> ArtifactRejected:
        exc = ArtifactRejected(artifact_id, version, reason, detail)
        log.error("artifact rejected", extra={"code": exc.code.value, "ctx": {
            "artifact": f"{artifact_id}@{version}", "reason": reason.value, "detail": detail}})
        if self._on_incident is not None:
            try:
                self._on_incident(exc)
            except Exception:
                log.exception("artifact incident handler failed", extra={"code": "ARTIFACT_INCIDENT_ERROR"})
        return exc

    def load(self, artifact_id: str, version: str) -> VerifiedArtifact:
        entry = self.entry(artifact_id, version)
        if entry is None:
            raise self._reject(artifact_id, version, RejectReason.NOT_LISTED, "not in models.artifacts")
        path = self._root / entry.path
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            raise self._reject(artifact_id, version, RejectReason.MISSING, f"{entry.path} not found") from None
        except OSError as exc:
            raise self._reject(artifact_id, version, RejectReason.MISSING,
                               f"{entry.path} unreadable ({type(exc).__name__})") from None
        digest = hashlib.sha256(data).hexdigest()
        if digest != entry.ref.sha256:
            raise self._reject(artifact_id, version, RejectReason.HASH_MISMATCH,
                               f"sha256 {digest[:12]}... != inventory {entry.ref.sha256[:12]}...")
        return VerifiedArtifact(entry.ref, data, approved=self._policy.is_model_approved(entry.ref))
