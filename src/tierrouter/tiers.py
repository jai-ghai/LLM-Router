"""Load tiers.yaml: tier names and their model name per provider. See config/tiers.example.yaml.

No model is called from here or anywhere else in this package (ARCHITECTURE.md §1) — this module
only resolves a tier name to a model name string.

P1 (PLAN-v2.md §7): `UNREVIEWED_TIER` is a catalog-only sentinel for newly imported models.
It is never a declared tier in tiers.yaml and never a valid `Rule.tier` — `catalog.pick()`
excludes every entry with this tier until an admin assigns a real one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from tierrouter.rules import RuleError

# Catalog-only sentinel (P1). Not part of TierTable.names; never routable.
UNREVIEWED_TIER = "UNREVIEWED"


@dataclass(frozen=True)
class TierSpec:
    name: str
    description: str
    models: dict[str, str]


@dataclass(frozen=True)
class TierTable:
    tiers: tuple[TierSpec, ...]
    default_provider: str
    low_confidence_fallback: str

    def __post_init__(self) -> None:
        names = {t.name for t in self.tiers}
        if len(names) != len(self.tiers):
            raise RuleError("tiers.yaml has duplicate tier names")
        if self.low_confidence_fallback not in names:
            raise RuleError(
                f"low_confidence_fallback '{self.low_confidence_fallback}' is not a declared tier"
            )

    @property
    def names(self) -> set[str]:
        return {t.name for t in self.tiers}

    def rank(self) -> dict[str, int]:
        """Declared order as a capability rank (0 = least capable). Used by the model catalog to
        never silently substitute a LOWER-capability model than the one actually decided on.
        """
        return {t.name: i for i, t in enumerate(self.tiers)}

    def model_for(self, tier_name: str, provider: str | None = None) -> str:
        """The model name for a tier, under `provider` (or `default_provider` if omitted)."""
        provider = provider or self.default_provider
        spec = next((t for t in self.tiers if t.name == tier_name), None)
        if spec is None:
            raise RuleError(f"'{tier_name}' is not a declared tier. Known: {sorted(self.names)}")
        if provider not in spec.models:
            raise RuleError(
                f"tier '{tier_name}' has no model for provider '{provider}'. "
                f"Known providers: {sorted(spec.models)}"
            )
        return spec.models[provider]


def load_tiers(path: str | Path) -> TierTable:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuleError(f"cannot read tiers file {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("tiers"), list) or not data["tiers"]:
        raise RuleError(f"{path}: expected a mapping with a non-empty 'tiers' list")
    specs: list[TierSpec] = []
    for index, entry in enumerate(data["tiers"]):
        if not isinstance(entry, dict) or not entry.get("name") or not isinstance(entry.get("models"), dict):
            raise RuleError(f"{path}: tiers[{index}] needs 'name' and a 'models' mapping")
        specs.append(
            TierSpec(
                name=str(entry["name"]).strip(),
                description=str(entry.get("description") or "").strip(),
                models={str(k): str(v) for k, v in entry["models"].items()},
            )
        )
    default_provider = str(data.get("default_provider") or "").strip()
    if not default_provider:
        raise RuleError(f"{path}: 'default_provider' is required")
    fallback = str(data.get("low_confidence_fallback") or "").strip()
    if not fallback:
        raise RuleError(f"{path}: 'low_confidence_fallback' is required")
    return TierTable(tiers=tuple(specs), default_provider=default_provider, low_confidence_fallback=fallback)
