"""Model catalog: multiple candidate models per tier, each with a declared cost and whether it
accepts an image, plus a selection mode ("cost" / "accuracy" / "balanced").

Optional layer on top of tiers.yaml's simple "one model per tier per provider" mapping
(tiers.py). When a catalog is supplied, resolve() picks from it instead: the best catalog
model that matches the tier a rule or the classifier decided on, filtered to whatever subset
of the catalog the caller selected (or all of it), the requested provider, and the modalities
the request needs. It never substitutes a model BELOW the decided tier — only at or above it,
if the exact tier has no match in the current selection (G2 in PLAN-v2.md §3).

`optimize_for` (P2) decides which of the matching candidates wins:
  - "cost": cheapest (the original behaviour; stays the default so nothing changes for
    callers that don't opt in).
  - "accuracy": the candidate with the highest human quality rating for this request's
    (model_id, metadata-field value) pair, cost as tiebreaker. Falls back to "cost" with a
    visible note when no ratings exist for any candidate — honest about why.
  - "balanced": `quality / log1p(cost + 1)` as a bluntly-labelled heuristic; same
    fall-back-to-cost behaviour when ratings are missing.

Unreviewed models (G4) stay eligible in every mode — accuracy mode just can't score them, so
if the pool is all unreviewed, the mode falls back to cost with a note.

No model is called from here or anywhere else in this package (ARCHITECTURE.md §1) — this
module only picks a model id string from a config file, same discipline as tiers.py.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from tierrouter.rules import RuleError
from tierrouter.tiers import UNREVIEWED_TIER

OptimizeFor = Literal["cost", "accuracy", "balanced"]
VALID_OPTIMIZE_FOR: tuple[OptimizeFor, ...] = ("cost", "accuracy", "balanced")


def _passes_quality(
    entry: CatalogEntry, quality: Mapping[tuple[str, str], float], quality_value: str, min_quality: float
) -> bool:
    score = quality.get((entry.id, quality_value))
    return score is None or score >= min_quality


def _choose_by_mode(
    candidates: list[CatalogEntry],
    mode: OptimizeFor,
    quality: Mapping[tuple[str, str], float] | None,
    quality_value: str | None,
) -> tuple[CatalogEntry, str | None]:
    """Pick one entry from a non-empty candidate list using the chosen mode.

    Cost mode is the original behaviour: cheapest, with model id as a stable tiebreaker.
    Accuracy mode maximises the recorded human rating for (id, quality_value) — cost is the
    tiebreaker (cheapest wins among equally-rated), then id. Balanced mode maximises
    `rating / log1p(cost + 1)` on the same rated subset.

    Accuracy and balanced modes need ratings. If the caller didn't supply any, or none of the
    candidates has a rating for this quality_value, both modes fall back to cost and return a
    note that says exactly why — rather than silently pretending the request was ranked by
    quality when it wasn't.
    """
    if not candidates:  # defensive; callers already guard this
        raise RuleError("no candidates to choose from")
    if mode == "cost":
        return min(candidates, key=lambda e: (e.cost, e.id)), None

    # accuracy / balanced both need ratings for THIS request's quality_value.
    # Distinguish "caller didn't pass a quality map at all" (None) from "caller passed
    # an empty map" ({}) — both mean no ratings, but the note should say so honestly.
    # Using `is None` (not `not quality`) so an empty dict still reaches the
    # "no candidate has a rating" branch below.
    if quality is None or not quality_value:
        chosen = min(candidates, key=lambda e: (e.cost, e.id))
        return chosen, (
            f"optimize_for='{mode}' requested but no quality ratings were provided; "
            "used cost instead"
        )
    scored = [(e, quality.get((e.id, quality_value))) for e in candidates]
    rated = [(e, r) for e, r in scored if r is not None]
    if not rated:
        chosen = min(candidates, key=lambda e: (e.cost, e.id))
        return chosen, (
            f"optimize_for='{mode}' requested but no candidate has a rating for "
            f"value='{quality_value}'; used cost instead"
        )
    # Sort by (primary score DESC, cost ASC, id ASC) so the first entry is the winner.
    # Using sorted() for the id tiebreak because negating a string is awkward in a max() key.
    if mode == "accuracy":
        ordered = sorted(rated, key=lambda pair: (-pair[1], pair[0].cost, pair[0].id))
        return ordered[0][0], None
    # balanced
    ordered = sorted(
        rated,
        key=lambda pair: (
            -(pair[1] / math.log1p(pair[0].cost + 1.0)),
            pair[0].cost,
            pair[0].id,
        ),
    )
    return ordered[0][0], None


def _compose_note(*notes: str | None) -> str | None:
    """Join non-empty notes with '; '. Returns None when nothing had to be explained."""
    parts = [n for n in notes if n]
    return "; ".join(parts) if parts else None


@dataclass(frozen=True)
class CatalogEntry:
    id: str
    provider: str
    tier: str
    cost: float
    supports_image: bool = True
    supports_video: bool = False
    supports_audio: bool = False
    label: str = ""
    context_length: int | None = None
    assigned_by: str = ""
    """Informational only (tokens) — not used in `pick()`. Refreshable from a live source via
    `experiments/sync_catalog.py`; see capability.py.

    P3 (PLAN-v2.md §6): `supports_video` / `supports_audio` default False — most catalog
    rows today are text/image; video/audio is opt-in so an absent YAML key never silently
    claims a modality the model doesn't have.

    P1 (PLAN-v2.md §7): `assigned_by` records which admin last set this entry's tier via
    PUT /v1/catalog/{id}. Empty for rows that have never been reviewed through the UI.
    `tier == UNREVIEWED` means the entry is never a pick() candidate (G5).
    """


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
        needs_video: bool = False,
        needs_audio: bool = False,
        quality: Mapping[tuple[str, str], float] | None = None,
        quality_value: str | None = None,
        min_quality: float | None = None,
        optimize_for: OptimizeFor = "cost",
    ) -> tuple[CatalogEntry, str | None]:
        """One entry from `tier`, chosen by `optimize_for` from the selection/provider/modality
        filter, with an optional quality floor layered on top.

        `needs_image` / `needs_video` / `needs_audio` (P3 — PLAN-v2.md §6) each drop candidates
        that don't declare the matching `supports_*` flag. Same filter shape for all three —
        a request that needs video never lands on a text-only model.

        `quality`/`quality_value`/`min_quality` are an optional human-reviewed quality floor
        (see quality.py) — cost matters, but for some use cases (PLAN-v2.md §1's video-QC
        example) a wrong answer costs more than a cheap model saves. A candidate is excluded
        only if it HAS a recorded average rating for `quality_value` and that rating is below
        `min_quality`; a model nobody has reviewed yet is never excluded for failing a check
        that was never run on it (G4). If the floor would leave nothing to choose from at all,
        it is dropped (noted) rather than failing the request — a stale catalog shouldn't be
        able to block every request.

        `optimize_for` controls which of the candidates wins (P2 — PLAN-v2.md §5):
          - "cost" (default): cheapest. Unchanged behaviour, so existing callers see no
            difference.
          - "accuracy": candidate with the highest human rating for (id, quality_value); cost
            is the tiebreaker, id for determinism. Fall back to cost + note when ratings are
            missing — honest rather than silent.
          - "balanced": `rating / log1p(cost + 1)` on the rated subset; same fallback.

        P1 (PLAN-v2.md §7 / G5): entries with `tier == UNREVIEWED` are **always** excluded,
        even when listed in `selected`. An admin must assign a real tier before resolve()
        will pick them.

        Returns `(entry, note)`. `note` is `None` when nothing needed explaining; otherwise it
        describes the tier substitution and/or quality-floor effect and/or optimize_for
        fall-back, joined by `; ` — never silently combined, never silently dropped.
        """
        if optimize_for not in VALID_OPTIMIZE_FOR:
            raise RuleError(
                f"optimize_for must be one of {VALID_OPTIMIZE_FOR!r}; got '{optimize_for}'"
            )
        candidates = [
            e
            for e in self.entries
            if e.tier != UNREVIEWED_TIER
            and (selected is None or e.id in selected)
            and (provider is None or e.provider == provider)
            and (not needs_image or e.supports_image)
            and (not needs_video or e.supports_video)
            and (not needs_audio or e.supports_audio)
        ]
        if not candidates:
            raise RuleError(
                "no catalog model matches the current selection/provider/modality requirement "
                f"(selected={sorted(selected) if selected else 'all'}, provider={provider or 'any'}, "
                f"needs_image={needs_image}, needs_video={needs_video}, needs_audio={needs_audio})"
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
            chosen, mode_note = _choose_by_mode(exact, optimize_for, quality, quality_value)
            return chosen, _compose_note(quality_note, mode_note)
        above = [e for e in candidates if tier_rank.get(e.tier, -1) > wanted_rank]
        if above:
            # Pick the lowest-rank tier that's still above the decided one (G2: never
            # downgrade), then apply the chosen mode WITHIN that bucket — so accuracy mode
            # doesn't jump to a much larger tier just because something there is rated higher.
            lowest_above = min(tier_rank.get(e.tier, 0) for e in above)
            at_lowest = [e for e in above if tier_rank.get(e.tier, -1) == lowest_above]
            chosen, mode_note = _choose_by_mode(at_lowest, optimize_for, quality, quality_value)
            tier_note = (
                f"no '{tier}' model in the current selection; used '{chosen.tier}' model "
                f"'{chosen.id}' instead — never downgraded below the decided tier"
            )
            return chosen, _compose_note(tier_note, quality_note, mode_note)
        raise RuleError(
            f"no catalog model at tier '{tier}' or above is available in the current "
            "selection/provider/modality filters"
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
                # Video/audio default False: an absent key must not claim a modality the
                # operator never declared (unlike image, which historically defaulted True).
                supports_video=bool(entry.get("supports_video", False)),
                supports_audio=bool(entry.get("supports_audio", False)),
                label=str(entry.get("label") or entry["id"]).strip(),
                context_length=context_length,
                assigned_by=str(entry.get("assigned_by") or "").strip(),
            )
        )
    return ModelCatalog(entries=tuple(entries))


def entry_to_yaml_dict(entry: CatalogEntry) -> dict[str, object]:
    """Serialize one CatalogEntry for catalog.yaml — omit empty optional fields so older
    files stay tidy when rewritten by PUT /v1/catalog/{id} or sync_catalog."""
    row: dict[str, object] = {
        "id": entry.id,
        "provider": entry.provider,
        "tier": entry.tier,
        "cost": entry.cost,
        "supports_image": entry.supports_image,
        "supports_video": entry.supports_video,
        "supports_audio": entry.supports_audio,
        "label": entry.label or entry.id,
    }
    if entry.context_length is not None:
        row["context_length"] = entry.context_length
    if entry.assigned_by:
        row["assigned_by"] = entry.assigned_by
    return row


def replace_entry(catalog: ModelCatalog, updated: CatalogEntry) -> ModelCatalog:
    """Return a new ModelCatalog with `updated` swapped in by id. Raises RuleError if the
    id is unknown — callers must not invent rows through the tier-assignment endpoint."""
    found = False
    entries: list[CatalogEntry] = []
    for entry in catalog.entries:
        if entry.id == updated.id:
            entries.append(updated)
            found = True
        else:
            entries.append(entry)
    if not found:
        raise RuleError(f"no catalog entry with id '{updated.id}'")
    return ModelCatalog(entries=tuple(entries))
