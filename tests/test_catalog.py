from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gradist.catalog import CatalogEntry, ModelCatalog, entry_to_yaml_dict, load_catalog
from gradist.rules import RuleError

RANK = {"small": 0, "medium": 1, "large": 2}

CATALOG = ModelCatalog(
    entries=(
        CatalogEntry("a-small", "gemini", "small", cost=1, supports_image=False, label="A Small"),
        CatalogEntry("b-small", "gemini", "small", cost=2, supports_image=True, label="B Small"),
        CatalogEntry("a-medium", "gemini", "medium", cost=4, supports_image=True, label="A Medium"),
        CatalogEntry("a-large", "anthropic", "large", cost=10, supports_image=True, label="A Large"),
    )
)


def test_picks_the_cheapest_exact_tier_match() -> None:
    entry, note = CATALOG.pick("small", RANK)
    assert entry.id == "a-small"
    assert note is None


def test_image_requirement_filters_out_non_image_models() -> None:
    entry, note = CATALOG.pick("small", RANK, needs_image=True)
    assert entry.id == "b-small"  # a-small is cheaper but doesn't support images
    assert note is None


def test_selection_restricts_the_candidate_pool() -> None:
    entry, _note = CATALOG.pick("small", RANK, selected={"b-small"})
    assert entry.id == "b-small"


def test_never_downgrades_below_the_decided_tier() -> None:
    # Only "large" is available once restricted to anthropic — a "small" decision still
    # gets a capable model, never something cheaper/weaker than what was decided.
    entry, note = CATALOG.pick("small", RANK, provider="anthropic")
    assert entry.id == "a-large"
    assert note is not None and "never downgraded" in note


def test_raises_when_nothing_matches_the_filter() -> None:
    with pytest.raises(RuleError, match="no catalog model matches"):
        CATALOG.pick("small", RANK, selected={"does-not-exist"})


def test_raises_when_nothing_at_or_above_the_tier_is_available() -> None:
    with pytest.raises(RuleError, match="tier .* or above"):
        CATALOG.pick("large", RANK, selected={"a-small"})


def test_duplicate_ids_are_refused() -> None:
    with pytest.raises(RuleError, match="duplicate"):
        ModelCatalog(entries=(CatalogEntry("x", "gemini", "small", 1), CatalogEntry("x", "gemini", "large", 2)))


def test_load_catalog_from_file(tmp_path: Path) -> None:
    path = tmp_path / "catalog.yaml"
    path.write_text(
        "catalog:\n"
        "  - id: a-small\n    provider: gemini\n    tier: small\n    cost: 1\n"
        "  - id: a-large\n    provider: gemini\n    tier: large\n    cost: 10\n"
        "    supports_image: false\n    label: A Large\n",
        encoding="utf-8",
    )
    catalog = load_catalog(path)
    assert len(catalog.entries) == 2
    large = next(e for e in catalog.entries if e.id == "a-large")
    assert large.supports_image is False
    assert large.label == "A Large"


@pytest.mark.parametrize("text", ["not: catalog", "catalog: [{id: x}]", "catalog: []"])
def test_load_catalog_rejects_bad_shapes(tmp_path: Path, text: str) -> None:
    path = tmp_path / "catalog.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(RuleError):
        load_catalog(path)
    with pytest.raises(RuleError):
        load_catalog(tmp_path / "missing.yaml")


def test_load_catalog_parses_context_length(tmp_path: Path) -> None:
    path = tmp_path / "catalog.yaml"
    path.write_text(
        "catalog:\n  - id: a\n    provider: gemini\n    tier: small\n    cost: 1\n    context_length: 131072\n",
        encoding="utf-8",
    )
    entry = load_catalog(path).entries[0]
    assert entry.context_length == 131072


def test_quality_floor_excludes_a_reviewed_but_low_scoring_model() -> None:
    # Both are "medium". a-medium is cheaper but scores 2.0; b-medium scores 4.5.
    catalog = ModelCatalog(
        entries=(
            CatalogEntry("a-medium", "gemini", "medium", cost=1),
            CatalogEntry("b-medium", "gemini", "medium", cost=5),
        )
    )
    quality = {("a-medium", "qc_review"): 2.0, ("b-medium", "qc_review"): 4.5}
    entry, note = catalog.pick("medium", RANK, quality=quality, quality_value="qc_review", min_quality=3.0)
    assert entry.id == "b-medium"
    assert note is not None and "excluded" in note


def test_quality_floor_never_excludes_an_unreviewed_model() -> None:
    catalog = ModelCatalog(entries=(CatalogEntry("unreviewed", "gemini", "medium", cost=1),))
    entry, note = catalog.pick("medium", RANK, quality={}, quality_value="qc_review", min_quality=3.0)
    assert entry.id == "unreviewed"
    assert note is None


def test_quality_floor_is_ignored_rather_than_failing_the_request_if_nothing_passes() -> None:
    catalog = ModelCatalog(entries=(CatalogEntry("weak", "gemini", "medium", cost=1),))
    quality = {("weak", "qc_review"): 1.0}
    entry, note = catalog.pick("medium", RANK, quality=quality, quality_value="qc_review", min_quality=4.0)
    assert entry.id == "weak"
    assert note is not None and "ignored the bar" in note


# -----------------------------------------------------------------------------------------
# optimize_for selection mode, plus one
# defence-in-depth test for an invalid mode string.
# -----------------------------------------------------------------------------------------


def test_pick_accuracy_picks_highest_rated_when_ratings_exist() -> None:
    # Two medium candidates; the pricier one has a better human rating. Accuracy mode should
    # prefer the rated-higher model even though it's dearer — the whole point of P2.
    catalog = ModelCatalog(
        entries=(
            CatalogEntry("cheap-medium", "gemini", "medium", cost=1),
            CatalogEntry("good-medium", "gemini", "medium", cost=5),
        )
    )
    quality = {("cheap-medium", "qc_review"): 2.5, ("good-medium", "qc_review"): 4.5}
    entry, note = catalog.pick(
        "medium",
        RANK,
        quality=quality,
        quality_value="qc_review",
        optimize_for="accuracy",
    )
    assert entry.id == "good-medium"
    assert note is None  # cleanest path — no fall-back, no floor, nothing to explain


def test_pick_accuracy_without_ratings_uses_price_as_capability_proxy() -> None:
    # No ratings exist for this (model, qc_review) pair at all. Accuracy mode ranks by price
    # instead (priciest first) AND says so, so it never silently equals cost mode.
    catalog = ModelCatalog(
        entries=(
            CatalogEntry("a", "gemini", "medium", cost=1),
            CatalogEntry("b", "gemini", "medium", cost=5),
        )
    )
    entry, note = catalog.pick(
        "medium",
        RANK,
        quality={},
        quality_value="qc_review",
        optimize_for="accuracy",
    )
    assert entry.id == "b"
    assert note is not None and "no candidate has a rating" in note and "price as a capability proxy" in note


def test_three_modes_differ_without_ratings() -> None:
    catalog = ModelCatalog(
        entries=tuple(CatalogEntry(name, "gemini", "medium", cost=cost) for name, cost in [("a", 1), ("b", 3), ("c", 9)])
    )
    picks = {mode: catalog.pick("medium", RANK, optimize_for=mode)[0].id for mode in ("cost", "balanced", "accuracy")}
    assert picks == {"cost": "a", "balanced": "b", "accuracy": "c"}


def test_capability_beats_price_when_no_ratings() -> None:
    # An old, pricey model must not win accuracy over a newer, cheaper, more capable one.
    catalog = ModelCatalog(
        entries=(
            CatalogEntry("old-pricey", "openai", "large", cost=30, capability=25.0, capability_source="estimated"),
            CatalogEntry("new-strong", "anthropic", "large", cost=4, capability=59.0, capability_source="measured"),
            CatalogEntry("mid", "google", "large", cost=2, capability=50.0, capability_source="measured"),
        )
    )
    accuracy, note = catalog.pick("large", RANK, optimize_for="accuracy")
    assert accuracy.id == "new-strong"
    assert note is not None and "benchmark capability" in note
    assert catalog.pick("large", RANK, optimize_for="balanced")[0].id == "mid"
    assert catalog.pick("large", RANK, optimize_for="cost")[0].id == "mid"


def test_capability_survives_a_yaml_round_trip(tmp_path: Path) -> None:
    entry = CatalogEntry("m", "openai", "large", cost=1, capability=51.25, capability_source="measured")
    path = tmp_path / "catalog.yaml"
    path.write_text(yaml.safe_dump({"catalog": [entry_to_yaml_dict(entry)]}), encoding="utf-8")
    loaded = load_catalog(path).entries[0]
    assert (loaded.capability, loaded.capability_source) == (51.2, "measured")


def test_skill_scores_survive_a_yaml_round_trip_and_stay_unknown_when_absent(tmp_path: Path) -> None:
    scored = CatalogEntry(
        "m", "openai", "large", cost=1, multilingual=0.8564, multilingual_source="estimated",
        supports_structured_output=False,
    )
    bare = CatalogEntry("n", "openai", "large", cost=1)
    path = tmp_path / "catalog.yaml"
    path.write_text(yaml.safe_dump({"catalog": [entry_to_yaml_dict(scored), entry_to_yaml_dict(bare)]}), encoding="utf-8")
    loaded_scored, loaded_bare = load_catalog(path).entries
    assert (loaded_scored.multilingual, loaded_scored.multilingual_source) == (0.856, "estimated")
    assert loaded_scored.supports_structured_output is False
    assert (loaded_bare.multilingual, loaded_bare.supports_structured_output) == (None, None)


def test_pick_ranked_returns_the_whole_bucket_best_first() -> None:
    catalog = ModelCatalog(
        entries=(
            CatalogEntry("a", "gemini", "medium", cost=2),
            CatalogEntry("b", "openai", "medium", cost=1),
            CatalogEntry("c", "openai", "large", cost=9),
        )
    )
    ranking = catalog.pick_ranked("medium", RANK)
    assert [e.id for e in ranking.ranked] == ["b", "a"]
    assert ranking.pool_size == 3


def test_pick_accepts_several_providers() -> None:
    catalog = ModelCatalog(
        entries=(
            CatalogEntry("g", "gemini", "medium", cost=1),
            CatalogEntry("o", "openai", "medium", cost=2),
            CatalogEntry("a", "anthropic", "medium", cost=3),
        )
    )
    assert catalog.pick("medium", RANK, provider={"openai", "anthropic"})[0].id == "o"
    assert catalog.pick("medium", RANK, provider="anthropic, OpenAI")[0].id == "o"
    assert catalog.pick("medium", RANK, provider="anthropic")[0].id == "a"


def test_pick_accuracy_respects_min_quality_floor() -> None:
    # A rated 4.5, B rated 2.0, C unrated (cheapest). Floor=3.0: B is dropped, A and C stay.
    # Accuracy mode then picks A (highest rating). Note must mention the floor exclusion.
    catalog = ModelCatalog(
        entries=(
            CatalogEntry("a", "gemini", "medium", cost=5),
            CatalogEntry("b", "gemini", "medium", cost=3),
            CatalogEntry("c", "gemini", "medium", cost=1),
        )
    )
    quality = {("a", "qc_review"): 4.5, ("b", "qc_review"): 2.0}
    entry, note = catalog.pick(
        "medium",
        RANK,
        quality=quality,
        quality_value="qc_review",
        min_quality=3.0,
        optimize_for="accuracy",
    )
    assert entry.id == "a"
    assert note is not None and "excluded" in note and "3.0" in note


def test_pick_accuracy_unreviewed_model_still_eligible() -> None:
    # G4 regression: an unreviewed model must stay a candidate. When it's the only choice,
    # accuracy mode falls back to cost (there's nothing to rank), but it does NOT filter the
    # unreviewed model out of the pool.
    catalog = ModelCatalog(
        entries=(
            CatalogEntry("rated", "gemini", "medium", cost=5),
            CatalogEntry("unrated", "gemini", "medium", cost=1),
        )
    )
    quality = {("rated", "qc_review"): 4.0}
    # Both available: rated wins on accuracy.
    entry, _ = catalog.pick(
        "medium", RANK, quality=quality, quality_value="qc_review", optimize_for="accuracy"
    )
    assert entry.id == "rated"
    # Only unrated in the selection: it still gets returned (not dropped as "no accuracy score").
    entry, note = catalog.pick(
        "medium",
        RANK,
        selected={"unrated"},
        quality=quality,
        quality_value="qc_review",
        optimize_for="accuracy",
    )
    assert entry.id == "unrated"
    assert note is not None and "no candidate has a rating" in note


def test_pick_balanced_prefers_cheap_when_quality_ties() -> None:
    # Two candidates with identical ratings: balanced mode breaks the tie by preferring the
    # cheaper model (higher rating/cost ratio). Cost mode would pick cheap too; this confirms
    # balanced is at least as cost-aware as cost on a quality-tie.
    catalog = ModelCatalog(
        entries=(
            CatalogEntry("cheap", "gemini", "medium", cost=1),
            CatalogEntry("pricey", "gemini", "medium", cost=10),
        )
    )
    quality = {("cheap", "qc_review"): 4.0, ("pricey", "qc_review"): 4.0}
    entry, note = catalog.pick(
        "medium",
        RANK,
        quality=quality,
        quality_value="qc_review",
        optimize_for="balanced",
    )
    assert entry.id == "cheap"
    assert note is None


def test_pick_never_downgrades_below_decided_tier_in_any_mode() -> None:
    # G2 regression: even when accuracy mode would "prefer" a higher-rated larger model, the
    # per-tier scope keeps the pick at the DECIDED tier. We decide "small"; there is a small
    # candidate rated 2.0 and a large candidate rated 5.0 — small must still win in every mode.
    catalog = ModelCatalog(
        entries=(
            CatalogEntry("ok-small", "gemini", "small", cost=1),
            CatalogEntry("great-large", "gemini", "large", cost=10),
        )
    )
    quality = {("ok-small", "qc_review"): 2.0, ("great-large", "qc_review"): 5.0}
    for mode in ("cost", "accuracy", "balanced"):
        entry, _ = catalog.pick(
            "small",
            RANK,
            quality=quality,
            quality_value="qc_review",
            optimize_for=mode,  # type: ignore[arg-type]
        )
        assert entry.id == "ok-small", f"mode={mode} downgraded/upgraded unexpectedly"


def test_pick_rejects_unknown_optimize_for_values() -> None:
    # Defence in depth: the service layer validates optimize_for too (422), but this guards
    # anyone who passes raw strings in to pick() directly from a script or test.
    catalog = ModelCatalog(entries=(CatalogEntry("a", "gemini", "small", cost=1),))
    with pytest.raises(RuleError, match="optimize_for must be one of"):
        catalog.pick("small", RANK, optimize_for="speed")  # type: ignore[arg-type]


# -----------------------------------------------------------------------------------------
# needs_video / needs_audio filters.
# -----------------------------------------------------------------------------------------


def test_pick_filters_out_entries_without_video_when_needs_video() -> None:
    # Two medium candidates; only the dearer one declares video. needs_video must drop the
    # cheaper text-only model even though cost mode would otherwise prefer it.
    catalog = ModelCatalog(
        entries=(
            CatalogEntry("text-only", "gemini", "medium", cost=1, supports_video=False),
            CatalogEntry("video-ok", "gemini", "medium", cost=5, supports_video=True),
        )
    )
    entry, note = catalog.pick("medium", RANK, needs_video=True)
    assert entry.id == "video-ok"
    assert note is None


def test_pick_raises_rule_error_when_no_video_candidate_in_filter() -> None:
    catalog = ModelCatalog(
        entries=(CatalogEntry("text-only", "gemini", "medium", cost=1, supports_video=False),)
    )
    with pytest.raises(RuleError, match="needs_video=True"):
        catalog.pick("medium", RANK, needs_video=True)


def test_pick_filters_out_entries_without_audio_when_needs_audio() -> None:
    catalog = ModelCatalog(
        entries=(
            CatalogEntry("silent", "gemini", "small", cost=1, supports_audio=False),
            CatalogEntry("hears", "gemini", "small", cost=3, supports_audio=True),
        )
    )
    entry, _ = catalog.pick("small", RANK, needs_audio=True)
    assert entry.id == "hears"


# -----------------------------------------------------------------------------------------
# UNREVIEWED sentinel always excluded.
# -----------------------------------------------------------------------------------------


def test_pick_excludes_unreviewed_in_cost_mode() -> None:
    from gradist.tiers import UNREVIEWED_TIER

    catalog = ModelCatalog(
        entries=(
            CatalogEntry("pending", "gemini", "UNREVIEWED", cost=1),
            CatalogEntry("ready", "gemini", "small", cost=5),
        )
    )
    entry, _ = catalog.pick("small", RANK, optimize_for="cost")
    assert entry.id == "ready"
    assert entry.tier != UNREVIEWED_TIER


def test_pick_excludes_unreviewed_in_accuracy_mode() -> None:
    catalog = ModelCatalog(
        entries=(
            CatalogEntry("pending", "gemini", "UNREVIEWED", cost=1),
            CatalogEntry("ready", "gemini", "small", cost=5),
        )
    )
    quality = {("pending", "qc"): 5.0, ("ready", "qc"): 2.0}
    entry, _ = catalog.pick(
        "small", RANK, quality=quality, quality_value="qc", optimize_for="accuracy"
    )
    assert entry.id == "ready"  # pending is UNREVIEWED even with a better rating


def test_pick_excludes_unreviewed_in_balanced_mode() -> None:
    catalog = ModelCatalog(
        entries=(
            CatalogEntry("pending", "gemini", "UNREVIEWED", cost=1),
            CatalogEntry("ready", "gemini", "small", cost=5),
        )
    )
    quality = {("pending", "qc"): 5.0, ("ready", "qc"): 2.0}
    entry, _ = catalog.pick(
        "small", RANK, quality=quality, quality_value="qc", optimize_for="balanced"
    )
    assert entry.id == "ready"


def test_pick_excludes_unreviewed_even_if_in_selected_models() -> None:
    catalog = ModelCatalog(
        entries=(
            CatalogEntry("pending", "gemini", "UNREVIEWED", cost=1),
            CatalogEntry("ready", "gemini", "small", cost=5),
        )
    )
    # Legacy pick() still refuses UNREVIEWED (tier-ladder path). pick_from_pool allows them.
    with pytest.raises(RuleError, match="no catalog model"):
        catalog.pick("small", RANK, selected={"pending"})


def test_pick_from_pool_ignores_tier_and_allows_explicit_unreviewed() -> None:
    catalog = ModelCatalog(
        entries=(
            CatalogEntry("pending", "gemini", "UNREVIEWED", cost=1),
            CatalogEntry("ready", "gemini", "large", cost=5),
        )
    )
    entry, note = catalog.pick_from_pool(selected={"pending", "ready"}, optimize_for="cost")
    assert entry.id == "pending"
    assert note is not None and "selection pool" in note


def test_pick_from_pool_all_means_entire_catalog() -> None:
    catalog = ModelCatalog(
        entries=(
            CatalogEntry("pending", "gemini", "UNREVIEWED", cost=1),
            CatalogEntry("ready", "gemini", "small", cost=5),
        )
    )
    entry, _ = catalog.pick_from_pool(optimize_for="cost")
    assert entry.id == "pending"


def test_label_catalog_tiers_buckets_by_cost() -> None:
    from gradist.catalog import label_catalog_tiers

    catalog = ModelCatalog(
        entries=(
            CatalogEntry("a", "g", "UNREVIEWED", cost=1),
            CatalogEntry("b", "g", "UNREVIEWED", cost=5),
            CatalogEntry("c", "g", "UNREVIEWED", cost=10),
        )
    )
    labeled = label_catalog_tiers(catalog, ["small", "medium", "large"])
    by_id = {e.id: e for e in labeled.entries}
    assert by_id["a"].tier == "small"
    assert by_id["c"].tier == "large"
    assert by_id["a"].assigned_by == "heuristic"


def test_label_catalog_tiers_preserves_manual_assignments() -> None:
    from gradist.catalog import label_catalog_tiers

    catalog = ModelCatalog(
        entries=(
            CatalogEntry("a", "g", "UNREVIEWED", cost=1),
            CatalogEntry("manual", "g", "medium", cost=99, assigned_by="admin"),
        )
    )
    labeled = label_catalog_tiers(catalog, ["small", "medium", "large"])
    by_id = {e.id: e for e in labeled.entries}
    assert by_id["a"].tier == "small"
    assert by_id["manual"].tier == "medium"
    assert by_id["manual"].assigned_by == "admin"


def test_label_equal_costs_share_same_bucket() -> None:
    from gradist.catalog import label_catalog_tiers

    catalog = ModelCatalog(
        entries=(
            CatalogEntry("a", "g", "UNREVIEWED", cost=1),
            CatalogEntry("b", "g", "UNREVIEWED", cost=1),
            CatalogEntry("c", "g", "UNREVIEWED", cost=1),
        )
    )
    labeled = label_catalog_tiers(catalog, ["small", "medium", "large"])
    # All same cost → same browse label (cheapest bucket).
    assert {e.tier for e in labeled.entries} == {"small"}


def test_label_catalog_tiers_splits_by_model_count_thirds() -> None:
    from gradist.catalog import label_catalog_tiers

    catalog = ModelCatalog(
        entries=tuple(
            CatalogEntry(f"m{i}", "g", "UNREVIEWED", cost=float(i)) for i in range(9)
        )
    )
    labeled = label_catalog_tiers(catalog, ["small", "medium", "large"])
    by_id = {e.id: e.tier for e in labeled.entries}
    assert by_id["m0"] == "small"
    assert by_id["m4"] == "medium"
    assert by_id["m8"] == "large"


def test_pick_from_pool_empty_selection_raises_not_all() -> None:
    with pytest.raises(RuleError, match=r"selected=\[\]"):
        CATALOG.pick_from_pool(selected=set(), optimize_for="cost")
