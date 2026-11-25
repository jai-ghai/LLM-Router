"""Model catalog: multiple candidate models per tier, each with a declared cost and whether it
accepts an image, plus a selection mode ("cost" / "accuracy" / "balanced").

Optional layer on top of tiers.yaml's simple "one model per tier per provider" mapping
(tiers.py). When a catalog is supplied, resolve() picks from it instead: the best catalog
model that matches the tier a rule or the classifier decided on, filtered to whatever subset
of the catalog the caller selected (or all of it), the requested provider, and the modalities
the request needs. It never substitutes a model BELOW the decided tier — only at or above it,
if the exact tier has no match in the current selection.

`optimize_for` (P2) decides which of the matching candidates wins:
  - "cost": cheapest (the original behaviour; stays the default so nothing changes for
    callers that don't opt in).
  - "accuracy": the candidate with the highest human quality rating for this request's
    (model_id, metadata-field value) pair, cost as tiebreaker.
  - "balanced": `quality / log1p(cost + 1)` as a bluntly-labelled heuristic.

When no candidate has a rating, accuracy and balanced rank by each entry's benchmark
`capability` (accuracy: most capable first; balanced: the cheapest model at or above the median
capability), or by price when no capability is recorded either — always with a visible note,
so the three modes differ instead of all collapsing to the cheapest model.

No model is called from here or anywhere else in this package (ARCHITECTURE.md §1) — this
module only picks a model id string from a config file, same discipline as tiers.py.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path
from typing import Literal

import yaml

from gradist.fit import CatalogScale, Fit, rank_by_fit
from gradist.rules import RuleError
from gradist.skills import SkillNeeds
from gradist.tiers import UNREVIEWED_TIER

OptimizeFor = Literal["cost", "accuracy", "balanced"]
VALID_OPTIMIZE_FOR: tuple[OptimizeFor, ...] = ("cost", "accuracy", "balanced")


def _passes_quality(
    entry: CatalogEntry, quality: Mapping[tuple[str, str], float], quality_value: str, min_quality: float
) -> bool:
    score = quality.get((entry.id, quality_value))
    return score is None or score >= min_quality


def _provider_filter(provider: str | Collection[str] | None) -> frozenset[str] | None:
    """None = any provider. A string may list several, comma-separated."""
    if provider is None:
        return None
    names = provider.split(",") if isinstance(provider, str) else provider
    wanted = frozenset(p.strip().lower() for p in names if p.strip())
    return wanted or None


def _rank_by_mode(
    candidates: list[CatalogEntry],
    mode: OptimizeFor,
    quality: Mapping[tuple[str, str], float] | None,
    quality_value: str | None,
) -> tuple[list[CatalogEntry], str | None]:
    """All candidates, best first for `mode`, plus a note when the ranking needed explaining.

    Cost: cheapest first, id as a stable tiebreaker. Accuracy: highest human rating for
    (id, quality_value), then cost, then id. Balanced: `rating / log1p(cost + 1)`. Rated models
    rank ahead of unrated ones. With no ratings at all, accuracy and balanced use benchmark
    `capability`, then price as a last-resort proxy — and say which one they used.
    """
    if not candidates:  # defensive; callers already guard this
        raise RuleError("no candidates to choose from")
    by_cost = sorted(candidates, key=lambda e: (e.cost, e.id))
    if mode == "cost":
        return by_cost, None

    rated: list[tuple[CatalogEntry, float]] = []
    if quality is not None and quality_value:
        rated = [(e, r) for e in candidates if (r := quality.get((e.id, quality_value))) is not None]
    if rated:
        if mode == "accuracy":
            rated.sort(key=lambda pair: (-pair[1], pair[0].cost, pair[0].id))
        else:
            rated.sort(key=lambda pair: (-(pair[1] / math.log1p(pair[0].cost + 1.0)), pair[0].cost, pair[0].id))
        rated_ids = {e.id for e, _ in rated}
        return [e for e, _ in rated] + [e for e in by_cost if e.id not in rated_ids], None

    why = (
        "no quality ratings were provided"
        if quality is None or not quality_value
        else f"no candidate has a rating for value='{quality_value}'"
    )
    scored = [(e, e.capability) for e in candidates if e.capability is not None]
    if scored:
        unscored = [e for e in by_cost if e.capability is None]
        if mode == "accuracy":
            scored.sort(key=lambda pair: (-pair[1], pair[0].cost, pair[0].id))
            ordered = [e for e, _ in scored]
            how = "highest benchmark capability first"
        else:
            median = statistics.median(c for _, c in scored)
            strong = sorted((e for e, c in scored if c >= median), key=lambda e: (e.cost, e.id))
            weaker = [e for e, c in sorted(scored, key=lambda pair: (-pair[1], pair[0].cost)) if c < median]
            ordered = strong + weaker
            how = f"cheapest model at or above the median capability {median:.0f}"
        return ordered + unscored, f"optimize_for='{mode}' requested but {why}; ranked by benchmark capability ({how})"
    if mode == "accuracy":
        ordered = sorted(candidates, key=lambda e: (-e.cost, e.id))
        how = "priciest first"
    else:
        middle = (len(by_cost) - 1) // 2
        ordered = [e for _, e in sorted(enumerate(by_cost), key=lambda p: (abs(p[0] - middle), p[0]))]
        how = "median price first"
    return ordered, f"optimize_for='{mode}' requested but {why}; ranked by price as a capability proxy ({how})"


def _choose_by_mode(
    candidates: list[CatalogEntry],
    mode: OptimizeFor,
    quality: Mapping[tuple[str, str], float] | None,
    quality_value: str | None,
) -> tuple[CatalogEntry, str | None]:
    ranked, note = _rank_by_mode(candidates, mode, quality, quality_value)
    return ranked[0], note


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
    capability: float | None = None
    """Benchmark capability (Artificial Analysis intelligence index scale). Ranks accuracy and
    balanced picks when no human ratings exist; see experiments/enrich_capability.py."""
    capability_source: str = ""
    """"measured" (published index) or "estimated" (fit on price + release date)."""
    multilingual: float | None = None
    """Artificial Analysis multilingual index — the Hindi / Indian-language skill score.
    See experiments/enrich_skills.py."""
    multilingual_source: str = ""
    """"measured" (published index, 0–1) or "estimated" (fit on capability, price and release date)."""
    supports_structured_output: bool | None = None
    """Whether the model accepts a JSON schema for its output; None when unknown."""
    """Informational only (tokens) — not used in `pick()`. Refreshable from a live source via
    `experiments/sync_catalog.py`; see capability.py.

    `supports_video` / `supports_audio` default False — most catalog
    rows today are text/image; video/audio is opt-in so an absent YAML key never silently
    claims a modality the model doesn't have.

    `assigned_by` records which admin last set this entry's tier via
    PUT /v1/catalog/{id}. Empty for rows that have never been reviewed through the UI.
    `tier == UNREVIEWED` means the entry is never a pick() candidate (G5).
    """


@dataclass(frozen=True)
class Ranking:
    ranked: list[CatalogEntry]
    """Candidates in the bucket that was picked from, best first."""
    note: str | None
    pool_size: int
    """Models left after the selection/provider/modality filters, across all tiers."""
    fits: dict[str, Fit] = field(default_factory=dict)
    """Fit per ranked model id when the request's skill needs were given; empty otherwise."""


@dataclass(frozen=True)
class ModelCatalog:
    entries: tuple[CatalogEntry, ...]

    def __post_init__(self) -> None:
        ids = [e.id for e in self.entries]
        if len(ids) != len(set(ids)):
            raise RuleError("catalog.yaml has duplicate model ids")

    @cached_property
    def scale(self) -> CatalogScale:
        return CatalogScale.of(self.entries)

    def pick(
        self,
        tier: str,
        tier_rank: dict[str, int],
        *,
        selected: set[str] | None = None,
        provider: str | Collection[str] | None = None,
        needs_image: bool = False,
        needs_video: bool = False,
        needs_audio: bool = False,
        quality: Mapping[tuple[str, str], float] | None = None,
        quality_value: str | None = None,
        min_quality: float | None = None,
        optimize_for: OptimizeFor = "cost",
    ) -> tuple[CatalogEntry, str | None]:
        """`pick_ranked(...)`'s winner and note."""
        ranking = self.pick_ranked(
            tier,
            tier_rank,
            selected=selected,
            provider=provider,
            needs_image=needs_image,
            needs_video=needs_video,
            needs_audio=needs_audio,
            quality=quality,
            quality_value=quality_value,
            min_quality=min_quality,
            optimize_for=optimize_for,
        )
        return ranking.ranked[0], ranking.note

    def pick_ranked(
        self,
        tier: str,
        tier_rank: dict[str, int],
        *,
        selected: set[str] | None = None,
        provider: str | Collection[str] | None = None,
        needs_image: bool = False,
        needs_video: bool = False,
        needs_audio: bool = False,
        quality: Mapping[tuple[str, str], float] | None = None,
        quality_value: str | None = None,
        min_quality: float | None = None,
        optimize_for: OptimizeFor = "cost",
        needs: SkillNeeds | None = None,
    ) -> Ranking:
        """Entries from `tier`, ranked by `optimize_for`, from the selection/provider/modality
        filter, with an optional quality floor layered on top.

        With `needs` (the request's skill needs), models whose context window is smaller than
        the input are dropped and the bucket is ranked by fit (fit.py) instead of the fixed
        `_rank_by_mode` sort; `optimize_for` then sets the capability and price weights.

        `needs_image` / `needs_video` / `needs_audio` each drop candidates
        that don't declare the matching `supports_*` flag. Same filter shape for all three —
        a request that needs video never lands on a text-only model.

        `quality`/`quality_value`/`min_quality` are an optional human-reviewed quality floor
        (see quality.py) — cost matters, but for some use cases (the video-QC
        example) a wrong answer costs more than a cheap model saves. A candidate is excluded
        only if it HAS a recorded average rating for `quality_value` and that rating is below
        `min_quality`; a model nobody has reviewed yet is never excluded for failing a check
        that was never run on it (G4). If the floor would leave nothing to choose from at all,
        it is dropped (noted) rather than failing the request — a stale catalog shouldn't be
        able to block every request.

        `optimize_for` orders the candidates; see `_rank_by_mode`.

        Entries with `tier == UNREVIEWED` are **always** excluded,
        even when listed in `selected`. An admin must assign a real tier before resolve()
        will pick them.

        `note` is `None` when nothing needed explaining; otherwise it describes the tier
        substitution and/or quality-floor effect and/or optimize_for proxy, joined by `; `.
        """
        if optimize_for not in VALID_OPTIMIZE_FOR:
            raise RuleError(
                f"optimize_for must be one of {VALID_OPTIMIZE_FOR!r}; got '{optimize_for}'"
            )
        providers = _provider_filter(provider)
        candidates = [
            e
            for e in self.entries
            if e.tier != UNREVIEWED_TIER
            and (selected is None or e.id in selected)
            and (providers is None or e.provider.lower() in providers)
            and (not needs_image or e.supports_image)
            and (not needs_video or e.supports_video)
            and (not needs_audio or e.supports_audio)
        ]
        if not candidates:
            raise RuleError(
                "no catalog model matches the current selection/provider/modality requirement "
                f"(selected={sorted(selected) if selected else 'all'}, "
                f"provider={','.join(sorted(providers)) if providers else 'any'}, "
                f"needs_image={needs_image}, needs_video={needs_video}, needs_audio={needs_audio})"
            )
        pool_size = len(candidates)

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

        context_note: str | None = None
        if needs is not None and needs.input_tokens:
            roomy = [e for e in candidates if e.context_length is None or e.context_length >= needs.input_tokens]
            if roomy:
                dropped = len(candidates) - len(roomy)
                if dropped:
                    context_note = (
                        f"dropped {dropped} model(s) whose context window is smaller than the "
                        f"~{needs.input_tokens:,} input tokens"
                    )
                candidates = roomy
            else:
                context_note = (
                    f"no model's context window fits the ~{needs.input_tokens:,} input tokens — "
                    "kept them all rather than failing the request"
                )

        def rank(bucket: list[CatalogEntry]) -> tuple[list[CatalogEntry], str | None, dict[str, Fit]]:
            if needs is None:
                ordered, how = _rank_by_mode(bucket, optimize_for, quality, quality_value)
                return ordered, how, {}
            ordered, fits, how_fit = rank_by_fit(bucket, needs.needs, self.scale, optimize_for, quality, quality_value)
            return ordered, how_fit, fits

        wanted_rank = tier_rank.get(tier)
        if wanted_rank is None:
            raise RuleError(f"'{tier}' is not a declared tier")
        exact = [e for e in candidates if e.tier == tier]
        if exact:
            ranked, mode_note, fits = rank(exact)
            return Ranking(ranked, _compose_note(quality_note, context_note, mode_note), pool_size, fits)
        above = [e for e in candidates if tier_rank.get(e.tier, -1) > wanted_rank]
        if above:
            # Pick the lowest-rank tier that's still above the decided one (G2: never
            # downgrade), then apply the chosen mode WITHIN that bucket — so accuracy mode
            # doesn't jump to a much larger tier just because something there is rated higher.
            lowest_above = min(tier_rank.get(e.tier, 0) for e in above)
            at_lowest = [e for e in above if tier_rank.get(e.tier, -1) == lowest_above]
            ranked, mode_note, fits = rank(at_lowest)
            tier_note = (
                f"no '{tier}' model in the current selection; used '{ranked[0].tier}' model "
                f"'{ranked[0].id}' instead — never downgraded below the decided tier"
            )
            return Ranking(ranked, _compose_note(tier_note, quality_note, context_note, mode_note), pool_size, fits)
        raise RuleError(
            f"no catalog model at tier '{tier}' or above is available in the current "
            "selection/provider/modality filters"
        )

    def pick_from_pool(
        self,
        *,
        selected: set[str] | None = None,
        provider: str | Collection[str] | None = None,
        needs_image: bool = False,
        needs_video: bool = False,
        needs_audio: bool = False,
        quality: Mapping[tuple[str, str], float] | None = None,
        quality_value: str | None = None,
        min_quality: float | None = None,
        optimize_for: OptimizeFor = "cost",
    ) -> tuple[CatalogEntry, str | None]:
        """Pick the best model from an explicit selection pool — no tier ladder.

        Callers: resolve() when a catalog is configured. Operator flow: select models
        (optionally per task), then cost/accuracy/balanced among that pool only.
        """
        if optimize_for not in VALID_OPTIMIZE_FOR:
            raise RuleError(
                f"optimize_for must be one of {VALID_OPTIMIZE_FOR!r}; got '{optimize_for}'"
            )
        if selected is None:
            candidates = list(self.entries)
        else:
            wanted = {s.lower() for s in selected}
            candidates = [e for e in self.entries if e.id.lower() in wanted]
        providers = _provider_filter(provider)
        candidates = [
            e
            for e in candidates
            if (providers is None or e.provider.lower() in providers)
            and (not needs_image or e.supports_image)
            and (not needs_video or e.supports_video)
            and (not needs_audio or e.supports_audio)
        ]
        if not candidates:
            selected_desc = "all" if selected is None else sorted(selected)
            raise RuleError(
                "no catalog model matches the current selection/provider/modality filters "
                f"(selected={selected_desc}, "
                f"provider={','.join(sorted(providers)) if providers else 'any'}, needs_image={needs_image}, "
                f"needs_video={needs_video}, needs_audio={needs_audio})"
            )

        quality_note: str | None = None
        if quality and min_quality is not None and quality_value is not None:
            passing = [e for e in candidates if _passes_quality(e, quality, quality_value, min_quality)]
            if passing:
                excluded = len(candidates) - len(passing)
                if excluded:
                    quality_note = (
                        f"excluded {excluded} reviewed model(s) below the {min_quality:.1f} "
                        f"quality bar for '{quality_value}'"
                    )
                candidates = passing
            else:
                quality_note = (
                    f"no model meets the {min_quality:.1f} quality bar for '{quality_value}' "
                    "in the selection — ignored the bar rather than failing the request"
                )

        chosen, mode_note = _choose_by_mode(candidates, optimize_for, quality, quality_value)
        pool_note = f"picked from {len(candidates)} model(s) in the selection pool"
        return chosen, _compose_note(pool_note, quality_note, mode_note)


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
                capability=float(entry["capability"]) if entry.get("capability") is not None else None,
                capability_source=str(entry.get("capability_source") or "").strip(),
                multilingual=float(entry["multilingual"]) if entry.get("multilingual") is not None else None,
                multilingual_source=str(entry.get("multilingual_source") or "").strip(),
                supports_structured_output=(
                    bool(entry["supports_structured_output"])
                    if entry.get("supports_structured_output") is not None
                    else None
                ),
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
    if entry.capability is not None:
        row["capability"] = round(entry.capability, 1)
        row["capability_source"] = entry.capability_source
    if entry.multilingual is not None:
        row["multilingual"] = round(entry.multilingual, 3)
        row["multilingual_source"] = entry.multilingual_source
    if entry.supports_structured_output is not None:
        row["supports_structured_output"] = entry.supports_structured_output
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


def suggest_tier_by_cost(cost: float, sorted_costs: list[float], tier_names: list[str]) -> str:
    """Browse-only: map cost → tier using equal-count ranks over `sorted_costs` (one per model).

    Cheapest third of models → first name (usually "small"), most expensive third → last
    ("large"). Same cost → same tier (uses the first index of that cost). Does not change
    routing — pick_from_pool ignores tier.
    """
    if not tier_names:
        raise RuleError("suggest_tier_by_cost needs at least one tier name")
    if not sorted_costs:
        return tier_names[0]
    n = len(sorted_costs)
    # First index with this cost (ties share the cheaper bucket).
    index = 0
    for i, value in enumerate(sorted_costs):
        if value >= cost:
            index = i
            break
        index = i
    bucket = int(index * len(tier_names) / n)
    return tier_names[min(bucket, len(tier_names) - 1)]


def label_catalog_tiers(
    catalog: ModelCatalog,
    tier_names: list[str],
    *,
    only_unreviewed_or_heuristic: bool = True,
) -> ModelCatalog:
    """Assign browse-only tier labels from cost thirds (`assigned_by='heuristic'`).

    `tier_names` must be cheapest→expensive declaration order (e.g. small, medium, large).
    Models are sorted by cost; the cheapest third get `tier_names[0]`, the expensive third
    get `tier_names[-1]`. By default only UNREVIEWED / previous heuristic rows are rewritten.
    """
    if not tier_names:
        raise RuleError("label_catalog_tiers needs at least one tier name")
    # Sort models by cost so buckets are equal-count by row. Same cost → same tier
    # (bucket from the first index of that cost group).
    ranked = sorted(catalog.entries, key=lambda e: (e.cost, e.id))
    n = len(ranked)
    id_to_tier: dict[str, str] = {}
    cost_first_index: dict[float, int] = {}
    for i, e in enumerate(ranked):
        cost_first_index.setdefault(e.cost, i)
    for e in ranked:
        index = cost_first_index[e.cost]
        bucket = min(len(tier_names) - 1, int(index * len(tier_names) / n)) if n else 0
        id_to_tier[e.id] = tier_names[bucket]

    entries: list[CatalogEntry] = []
    for e in catalog.entries:
        keep = only_unreviewed_or_heuristic and e.tier != UNREVIEWED_TIER and e.assigned_by not in (
            "",
            "heuristic",
        )
        if keep:
            entries.append(e)
            continue
        entries.append(replace(e, tier=id_to_tier[e.id], assigned_by="heuristic"))
    return ModelCatalog(entries=tuple(entries))
