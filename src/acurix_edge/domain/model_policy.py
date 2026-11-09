"""Model approval policy — default-deny (ADR-016).

A model may only take part in a decision when ALL hold:
- real model loading is enabled;
- the exact ModelRef (id, version, sha256) is in `approved_models`;
- the inference adapter's exact class is in `allowed_adapters`.

Adapter authorization is decided by the trusted class registered in the policy,
never by the adapter's self-declared `adapter_id` (R4-09): an adapter cannot
authorize itself by claiming another id, and subclasses of an allowed class are
not allowed implicitly.

Foundation 0.1.0 product policy (from configuration): approved_models = {},
real_model_loading = False, allowed_adapters = {NoOpInferenceAdapter}.
Configuration cannot change this in 0.1.0 (see config.settings validation). The
constructor below is a code-level seam used only by tests to exercise the guard;
it is not reachable from configuration or CLI.

Prohibited-family models (ADR-013) are never eligible for this list; that
restriction is enforced by policy/ADR review and the repository scanner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from acurix_edge.adapters.inference_noop import NoOpInferenceAdapter
from acurix_edge.contracts.inference import ModelRef

NOOP_ADAPTER_ID = NoOpInferenceAdapter.adapter_id


@dataclass(frozen=True, slots=True)
class ModelPolicy:
    approved_models: frozenset[ModelRef] = field(default_factory=frozenset)
    real_model_loading: bool = False
    allowed_adapters: frozenset[type] = field(default_factory=lambda: frozenset({NoOpInferenceAdapter}))

    def __post_init__(self) -> None:
        # Structure is validated strictly: a str/list/dict would make `in` checks lie
        # (substring or key matches), so only frozensets of the exact element types are accepted.
        if not isinstance(self.approved_models, frozenset):
            raise ValueError("approved_models must be a frozenset of ModelRef")
        if not all(isinstance(m, ModelRef) for m in self.approved_models):
            raise ValueError("approved_models must contain ModelRef entries")
        if not isinstance(self.real_model_loading, bool):
            raise ValueError("real_model_loading must be bool")
        if not isinstance(self.allowed_adapters, frozenset) or not self.allowed_adapters:
            raise ValueError("allowed_adapters must be a non-empty frozenset of adapter classes")
        for cls in self.allowed_adapters:
            if not isinstance(cls, type):
                raise ValueError("allowed_adapters must contain adapter classes, not ids or instances")
            declared = getattr(cls, "adapter_id", None)
            if not isinstance(declared, str) or not declared:
                raise ValueError(f"adapter class {cls.__name__} has no class-level adapter_id")

    @classmethod
    def from_config(cls, cfg: Any) -> "ModelPolicy":
        """Product policy. Config validation guarantees an empty list and loading disabled in 0.1.0."""
        m = cfg.models
        if not isinstance(m.approved_models, (list, tuple)) or len(m.approved_models) != 0:
            raise ValueError("0.1.0 requires an empty approved_models list (ADR-016)")
        if m.real_model_loading is not False:
            raise ValueError("0.1.0 requires real_model_loading = false (ADR-016)")
        return cls(approved_models=frozenset(), real_model_loading=False,
                   allowed_adapters=frozenset({NoOpInferenceAdapter}))

    def is_adapter_allowed(self, adapter: Any) -> bool:
        """Exact-class check. The instance must also not override the class-level id."""
        cls = type(adapter)
        if cls not in self.allowed_adapters:
            return False
        return getattr(adapter, "adapter_id", None) == getattr(cls, "adapter_id", None)

    def is_model_approved(self, model: ModelRef | None) -> bool:
        return self.real_model_loading is True and isinstance(model, ModelRef) and model in self.approved_models

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": "DEFAULT_DENY",
            "approved_models": sorted((m.to_dict() for m in self.approved_models), key=lambda d: (d["id"], d["version"])),
            "real_model_loading": self.real_model_loading,
            "allowed_adapters": sorted(c.adapter_id for c in self.allowed_adapters),
        }
