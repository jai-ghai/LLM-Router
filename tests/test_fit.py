from __future__ import annotations

from gradist.catalog import CatalogEntry, ModelCatalog
from gradist.fit import ESTIMATE_TRUST, describe_fit
from gradist.skills import SKILLS, SkillNeeds

RANKS = {"small": 0, "medium": 1, "large": 2}


def entry(model_id: str, cost: float, **fields: object) -> CatalogEntry:
    return CatalogEntry(id=model_id, provider="p", tier="small", cost=cost, **fields)  # type: ignore[arg-type]


def needs(**values: float) -> SkillNeeds:
    return SkillNeeds(needs={s: values.get(s, 0.0) for s in SKILLS})


CATALOG = ModelCatalog(
    entries=(
        entry("cheap-english", 0.0001, capability=20.0, capability_source="measured", multilingual=0.40, multilingual_source="measured"),
        entry("hindi-strong", 0.0003, capability=22.0, capability_source="measured", multilingual=0.90, multilingual_source="measured"),
        entry("smart-pricey", 0.01, capability=60.0, capability_source="measured", multilingual=0.70, multilingual_source="measured"),
    )
)


def test_without_needs_the_old_cost_sort_is_unchanged() -> None:
    ranking = CATALOG.pick_ranked("small", RANKS, optimize_for="cost")
    assert ranking.ranked[0].id == "cheap-english"
    assert ranking.fits == {}


def test_a_hindi_request_picks_the_hindi_model_over_the_cheapest() -> None:
    ranking = CATALOG.pick_ranked("small", RANKS, optimize_for="cost", needs=needs(hindi=1.0))
    assert ranking.ranked[0].id == "hindi-strong"
    winner = ranking.fits["hindi-strong"]
    assert winner.parts["hindi"] == 0.9
    assert winner.parts["price"] < 0
    assert ranking.note is not None and "ranked by fit" in ranking.note
    assert describe_fit(winner).startswith(f"fit {winner.score:.2f} (hindi +0.90")


def test_a_reasoning_heavy_request_pays_for_the_smart_model() -> None:
    cheap = CATALOG.pick_ranked("small", RANKS, optimize_for="cost", needs=needs(fast=1.0))
    hard = CATALOG.pick_ranked("small", RANKS, optimize_for="balanced", needs=needs(reasoning=1.0, writing=1.0))
    assert cheap.ranked[0].id == "cheap-english"
    assert hard.ranked[0].id == "smart-pricey"


def test_an_estimated_score_counts_for_less_than_a_measured_one() -> None:
    catalog = ModelCatalog(
        entries=(
            entry("guessed", 0.001, multilingual=0.8, multilingual_source="estimated"),
            entry("measured", 0.001, multilingual=0.8, multilingual_source="measured"),
        )
    )
    ranking = catalog.pick_ranked("small", RANKS, needs=needs(hindi=1.0))
    assert [e.id for e in ranking.ranked] == ["measured", "guessed"]
    assert ranking.fits["guessed"].parts["hindi"] == 0.8 * ESTIMATE_TRUST


def test_models_whose_window_is_too_small_for_the_input_are_dropped() -> None:
    catalog = ModelCatalog(
        entries=(
            entry("tiny-window", 0.0001, context_length=8_000),
            entry("big-window", 0.001, context_length=200_000),
        )
    )
    long = SkillNeeds(needs={s: 0.0 for s in SKILLS} | {"long_input": 0.6}, input_tokens=20_000)
    ranking = catalog.pick_ranked("small", RANKS, needs=long)
    assert [e.id for e in ranking.ranked] == ["big-window"]
    assert ranking.note is not None and "context window is smaller" in ranking.note


def test_when_no_window_fits_every_model_is_kept_with_a_note() -> None:
    catalog = ModelCatalog(entries=(entry("tiny-window", 0.0001, context_length=8_000),))
    huge = SkillNeeds(needs={s: 0.0 for s in SKILLS}, input_tokens=50_000)
    ranking = catalog.pick_ranked("small", RANKS, needs=huge)
    assert ranking.ranked[0].id == "tiny-window"
    assert ranking.note is not None and "kept them all" in ranking.note


def test_a_human_rating_replaces_benchmark_capability() -> None:
    quality = {("cheap-english", "news"): 5.0, ("smart-pricey", "news"): 2.0}
    ranking = CATALOG.pick_ranked(
        "small", RANKS, optimize_for="accuracy", quality=quality, quality_value="news", needs=needs(reasoning=1.0)
    )
    assert ranking.ranked[0].id == "cheap-english"
    assert ranking.fits["cheap-english"].skills["reasoning"].source == "rating"
