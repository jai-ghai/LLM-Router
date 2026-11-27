from __future__ import annotations

from gradist.catalog import CatalogEntry, ModelCatalog
from gradist.fit import ESTIMATE_TRUST, CatalogScale, describe_fit, enough_skill, fit, model_skills
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


def test_a_hard_request_with_the_same_needs_pays_for_the_smarter_model() -> None:
    def pick(difficulty: float | None) -> str:
        hindi = SkillNeeds(needs={s: 1.0 if s == "hindi" else 0.0 for s in SKILLS}, difficulty=difficulty)
        return CATALOG.pick_ranked("small", RANKS, optimize_for="balanced", needs=hindi).ranked[0].id

    assert pick(None) == "hindi-strong"
    assert pick(0.25) == "hindi-strong"
    assert pick(1.0) == "smart-pricey"


def test_difficulty_scales_only_the_capability_term_and_is_named_in_the_note() -> None:
    plain = CATALOG.pick_ranked("small", RANKS, optimize_for="balanced", needs=needs(hindi=1.0))
    hard = CATALOG.pick_ranked(
        "small", RANKS, optimize_for="balanced", needs=SkillNeeds(needs=needs(hindi=1.0).needs, difficulty=1.0)
    )
    before, after = plain.fits["hindi-strong"].parts, hard.fits["hindi-strong"].parts
    assert after["capability"] == before["capability"] * 1.5
    assert after["hindi"] == before["hindi"] and after["price"] == before["price"]
    assert hard.note is not None and "0.6 × 1.50 (difficulty 1.00) × capability" in hard.note


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


def test_enough_skill_rises_with_difficulty() -> None:
    assert enough_skill(None) == 1.0
    assert enough_skill(0.0) == 0.4
    assert enough_skill(1.0) == 1.0
    assert enough_skill(0.5) == 0.7


def test_an_easy_request_gives_a_mid_model_full_reasoning_credit() -> None:
    scale = CatalogScale(top_capability=60.0)
    mid = entry("mid", 0.0001, capability=30.0, capability_source="measured")
    top = entry("top", 0.01, capability=60.0, capability_source="measured")
    wanted = {"reasoning": 1.0}
    assert fit(mid, wanted, scale, "balanced", difficulty=0.1).parts["reasoning"] == 1.0
    assert fit(top, wanted, scale, "balanced", difficulty=0.1).parts["reasoning"] == 1.0
    assert fit(mid, wanted, scale, "balanced").parts["reasoning"] == 0.5


def test_a_batch_endpoint_is_never_fast() -> None:
    scale = CatalogScale(top_capability=1.0)
    skills = model_skills(entry("m:batch", 0.0001), scale)
    assert skills["fast"].value == 0.0 and skills["fast"].source == "batch"
    assert model_skills(entry("m", 0.0001), scale)["fast"].value > 0


HEARING = ModelCatalog(
    entries=(
        entry("deaf-cheap", 0.0001, supports_video=True),
        entry("hears", 0.001, supports_video=True, supports_audio=True),
    )
)


def test_a_strong_audio_need_drops_models_that_cannot_hear() -> None:
    ranking = HEARING.pick_ranked("small", RANKS, needs=needs(video=1.0, audio=0.7))
    assert [e.id for e in ranking.ranked] == ["hears"]
    assert ranking.note is not None and "cannot hear" in ranking.note
    weak = HEARING.pick_ranked("small", RANKS, needs=needs(video=1.0, audio=0.5))
    assert len(weak.ranked) == 2


def test_the_audio_filter_never_empties_the_pool() -> None:
    deaf = ModelCatalog(entries=(entry("deaf", 0.0001, supports_video=True),))
    ranking = deaf.pick_ranked("small", RANKS, needs=needs(audio=1.0))
    assert ranking.ranked[0].id == "deaf"
    assert ranking.note is not None and "kept them all" in ranking.note


BATCH = ModelCatalog(entries=(entry("m:batch", 0.0001), entry("m", 0.0002), entry("m:free", 0.0)))


def test_deadline_now_drops_batch_endpoints() -> None:
    assert "m:batch" in [e.id for e in BATCH.pick_ranked("small", RANKS, deadline="offline").ranked]
    now = BATCH.pick_ranked("small", RANKS, deadline="now")
    assert "m:batch" not in [e.id for e in now.ranked]
    assert now.note is not None and "batch endpoint" in now.note
    only_batch = ModelCatalog(entries=(entry("m:batch", 0.0001),))
    assert only_batch.pick_ranked("small", RANKS, deadline="now").ranked[0].id == "m:batch"


def test_exclude_free_drops_free_tiers() -> None:
    assert BATCH.pick_ranked("small", RANKS).ranked[0].id == "m:free"
    ranking = BATCH.pick_ranked("small", RANKS, exclude_free=True)
    assert "m:free" not in [e.id for e in ranking.ranked]
    assert ranking.note is not None and "free-tier" in ranking.note
