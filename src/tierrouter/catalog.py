"""Model catalog: multiple candidate models per tier, each with a declared cost and whether it
accepts an image.

Optional layer on top of tiers.yaml's simple "one model per tier per provider" mapping
(tiers.py). When a catalog is supplied, resolve() picks from it instead: the cheapest catalog
model that matches the tier a rule or the classifier decided on, filtered to whatever subset of
the catalog the caller selected (or all of it), the requested provider, and whether the request
has an image. It never substitutes a model BELOW the decided tier — only at or above it, if the
exact tier has no match in the current selection.

No model is called from here or anywhere else in this package (ARCHITECTURE.md §1) — this module
only picks a model id string from a config file, same discipline as tiers.py.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from tierrouter.rules import RuleError


def _passes_quality(
    entry: CatalogEntry, quality: Mapping[tuple[str, str], float], quality_value: str, min_quality: float
) -> bool:
    score = quality.get((entry.id, quality_value))
    return score is None or score >= min_quality


@dataclass(frozen=True)
class CatalogEntry:
    id: str
    provider: str
    tier: str
    cost: float
    supports_image: bool = True
    label: str = ""
    context_length: int | None = None
    """Informational only (tokens) — not used in `pick()`. Refreshable from a live source via
    `experiments/refresh_catalog.py`; see capability.py."""


@dataclass(frozen=True)
class ModelCatalog:
    entries: tuple[CatalogEntry, ...]

    def __post_init__(self) -> None:
        ids = [e.id for e in self.entries]
        if len(ids) != len(set(ids)):
            raise RuleError("catalog.yaml has duplicate model ids")

    def pick(
        self,
        tier: str,
        tier_rank: dict[str, int],
        *,
        selected: set[str] | None = None,
        provider: str | None = None,
        needs_image: bool = False,
        quality: Mapping[tuple[str, str], float] | None = None,
        quality_value: str | None = None,
        min_quality: float | None = None,
    ) -> tuple[CatalogEntry, str | None]:
        """The cheapest entry at `tier`, from the given selection/provider/image/quality filter.

        `quality`/`quality_value`/`min_quality` are an optional human-reviewed quality floor
        (see quality.py) — cost matters, but for some use cases (ARCHITECTURE.md's video-QC
        example) a wrong answer costs more than a cheap model saves. A candidate is excluded
        only if it HAS a recorded average rating for `quality_value` and that rating is below
        `min_quality`; a model nobody has reviewed yet is never excluded for failing a check
        that was never run on it. If the floor would leave nothing to choose from at all, it is
        dropped (noted) rather than failing the request outright — a stale catalog shouldn't be
        able to block every request.

        Returns `(entry, note)`. `note` is `None` when nothing needed explaining, or describes a
        tier substitution and/or a quality-floor effect — never both silently.
        """
        candidates = [
            e
            for e in self.entries
            if (selected is None or e.id in selected)
            and (provider is None or e.provider == provider)
            and (not needs_image or e.supports_image)
        ]
        if not candidates:
            raise RuleError(
                "no catalog model matches the current selection/provider/image requirement "
                f"(selected={sorted(selected) if selected else 'all'}, provider={provider or 'any'}, "
                f"needs_image={needs_image})"
            )

        quality_note: str | None = None
        if quality and min_quality is not None and quality_value is not None:
            passing = [e for e in candidates if _passes_quality(e, quality, quality_value, min_quality)]
            if passing:
                excluded = len(candidates) - len(passing)
                if excluded:
                    quality_note = f"excluded {excluded} reviewed model(s) below the {min_quality:.1f} quality bar for '{quality_value}'"
                candidates = passing
            else:
                quality_note = (
                    f"no model meets the {min_quality:.1f} quality bar for '{quality_value}' in the "
                    "current selection — ignored the bar rather than failing the request"
                )

        wanted_rank = tier_rank.get(tier)
        if wanted_rank is None:
            raise RuleError(f"'{tier}' is not a declared tier")
        exact = [e for e in candidates if e.tier == tier]
        if exact:
            return min(exact, key=lambda e: e.cost), quality_note
        above = [e for e in candidates if tier_rank.get(e.tier, -1) > wanted_rank]
        if above:
            chosen = min(above, key=lambda e: (tier_rank.get(e.tier, 0), e.cost))
            tier_note = (
                f"no '{tier}' model in the current selection; used '{chosen.tier}' model "
                f"'{chosen.id}' instead — never downgraded below the decided tier"
            )
            return chosen, f"{tier_note}; {quality_note}" if quality_note else tier_note
        raise RuleError(
            f"no catalog model at tier '{tier}' or above is available in the current "
            "selection/provider/image filters"
        )


def load_catalog(path: str | Path) -> ModelCatalog:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuleError(f"cannot read catalog file {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("catalog"), list) or not data["catalog"]:
        raise RuleError(f"{path}: expected a mapping with a non-empty 'catalog' list")
    entries: list[CatalogEntry] = []
    for index, entry in enumerate(data["catalog"]):
        if not isinstance(entry, dict) or not entry.get("id") or not entry.get("provider") or not entry.get("tier"):
            raise RuleError(f"{path}: catalog[{index}] needs 'id', 'provider', and 'tier'")
        try:
            cost = float(entry.get("cost", 1.0))
        except (TypeError, ValueError) as exc:
            raise RuleError(f"{path}: catalog[{index}] has a non-numeric 'cost'") from exc
        context_length = entry.get("context_length")
        if context_length is not None:
            try:
                context_length = int(context_length)
            except (TypeError, ValueError) as exc:
                raise RuleError(f"{path}: catalog[{index}] has a non-integer 'context_length'") from exc
        entries.append(
            CatalogEntry(
                id=str(entry["id"]).strip(),
                provider=str(entry["provider"]).strip(),
                tier=str(entry["tier"]).strip(),
                cost=cost,
                supports_image=bool(entry.get("supports_image", True)),
                label=str(entry.get("label") or entry["id"]).strip(),
                context_length=context_length,
            )
        )
    return ModelCatalog(entries=tuple(entries))
