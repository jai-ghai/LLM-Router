from __future__ import annotations

import pytest

from gradist.catalog import CatalogEntry, ModelCatalog
from gradist.classifier import TierClassifier
from gradist.features import HashingEmbedder, MetadataEncoder
from gradist.labels import TierLabel
from gradist.resolve import resolve, tier_by_need, validate_setup
from gradist.rules import Rule, RuleError, RuleSet
from gradist.tiers import TierSpec, TierTable

NEWS = ["parliament passed the budget bill", "the minister announced an election"]
SPORTS = ["goal scored in the final match", "the striker won the tournament"]
TIERS = TierTable(
    tiers=(
        TierSpec("small", "", {"gemini": "g-small", "anthropic": "a-small"}),
        TierSpec("medium", "", {"gemini": "g-medium", "anthropic": "a-medium"}),
        TierSpec("large", "", {"gemini": "g-large", "anthropic": "a-large"}),
    ),
    default_provider="gemini",
    low_confidence_fallback="medium",
)


def _encoder() -> MetadataEncoder:
    return MetadataEncoder({"channel_type": ["news", "sports"]})


def _trained_classifier() -> TierClassifier:
    clf = TierClassifier(_encoder(), HashingEmbedder())
    labels = [TierLabel(f"n{i}", {"channel_type": "news"}, t, "small", "human", "") for i, t in enumerate(NEWS)]
    labels += [TierLabel(f"s{i}", {"channel_type": "sports"}, t, "large", "human", "") for i, t in enumerate(SPORTS)]
    clf.fit(labels)
    return clf


def test_rule_wins_outright_over_the_classifier() -> None:
    rules = RuleSet(rules=(Rule.create({"channel_type": "sports"}, "small"),))
    decision = resolve({"channel_type": "sports"}, SPORTS[0], rules, TIERS, classifier=_trained_classifier())
    assert decision.tier == "small"  # the rule, not the classifier's "large"
    assert decision.confidence == 1.0
    assert decision.reason == "rule channel_type=sports"
    assert decision.model_name == "g-small"
    # structured trace: the rule matched, so the classifier/fallback steps were never reached
    assert decision.rule_outcome == "rule channel_type=sports"
    assert decision.classifier_outcome is None
    assert decision.fallback_outcome is None


def test_classifier_decides_when_no_rule_matches() -> None:
    decision = resolve({"channel_type": "news"}, NEWS[0], RuleSet(), TIERS, classifier=_trained_classifier())
    assert decision.tier == "small"
    assert decision.confidence is not None and decision.confidence > 0.5
    assert "classifier, confidence" in decision.reason
    assert decision.model_name == "g-small"
    assert decision.rule_outcome == "no matching rule"
    assert decision.classifier_outcome is not None and decision.classifier_outcome.startswith("small, confidence")
    assert decision.fallback_outcome == "not needed"


def test_low_confidence_falls_back_visibly() -> None:
    decision = resolve(
        {"channel_type": "news"},
        NEWS[0],
        RuleSet(),
        TIERS,
        classifier=_trained_classifier(),
        min_confidence=0.999,
    )
    assert decision.tier == "medium"  # tiers.low_confidence_fallback
    assert decision.confidence is not None and decision.confidence < 0.999
    assert "below the 1.00 threshold" in decision.reason
    assert decision.model_name == "g-medium"
    assert decision.classifier_outcome is not None and "below the" in decision.classifier_outcome
    assert decision.fallback_outcome == "used 'medium'"


def test_no_rule_and_no_classifier_uses_the_fallback() -> None:
    decision = resolve({"channel_type": "news"}, "anything", RuleSet(), TIERS)
    assert decision.tier == "medium"
    assert decision.confidence is None
    assert "no classifier is available" in decision.reason
    assert decision.rule_outcome == "no matching rule"
    assert decision.classifier_outcome == "no classifier configured"
    assert decision.fallback_outcome == "used 'medium'"


def test_provider_selects_the_model_column() -> None:
    rules = RuleSet(rules=(Rule.create({"channel_type": "sports"}, "large"),))
    decision = resolve({"channel_type": "sports"}, "x", rules, TIERS, provider="anthropic")
    assert decision.model_name == "a-large"


def test_rule_naming_an_unknown_tier_raises_at_resolve_time() -> None:
    rules = RuleSet(rules=(Rule.create({"channel_type": "sports"}, "giant"),))
    with pytest.raises(RuleError, match="not a declared tier"):
        resolve({"channel_type": "sports"}, "x", rules, TIERS)


def test_validate_setup_catches_a_bad_rule_before_any_request() -> None:
    good = RuleSet(rules=(Rule.create({"channel_type": "sports"}, "large"),))
    validate_setup(good, TIERS)  # does not raise
    bad = RuleSet(rules=(Rule.create({"channel_type": "sports"}, "giant"),))
    with pytest.raises(RuleError, match="unknown tier"):
        validate_setup(bad, TIERS)


CATALOG = ModelCatalog(
    entries=(
        CatalogEntry("cheap-small", "gemini", "small", cost=1),
        CatalogEntry("pricey-small", "gemini", "small", cost=5),
        CatalogEntry("cheap-large", "gemini", "large", cost=2),
    )
)


# Behaviour under test: resolve() hybrid — catalog + ladder (user: all providers + read content).
def test_resolve_with_a_catalog_uses_ladder_tier_then_picks() -> None:
    # Rule decides large → catalog picks cheapest *large* (not cheapest overall).
    rules = RuleSet(rules=(Rule.create({"channel_type": "sports"}, "large"),))
    decision = resolve({"channel_type": "sports"}, "x", rules, TIERS, catalog=CATALOG)
    assert decision.model_name == "cheap-large"
    assert decision.rule_outcome == "rule channel_type=sports"
    assert "catalog tier='large'" in decision.reason


def test_resolve_with_a_catalog_honours_the_caller_s_selection() -> None:
    rules = RuleSet(rules=(Rule.create({"channel_type": "sports"}, "small"),))
    decision = resolve(
        {"channel_type": "sports"}, "x", rules, TIERS, catalog=CATALOG, selected_models={"pricey-small"}
    )
    assert decision.model_name == "pricey-small"


RANKS = {"small": 0, "medium": 1, "large": 2}


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("cost", "medium"), ("balanced", "large"), ("accuracy", "large")],
)
def test_tier_by_need_moves_with_the_dial(mode: str, expected: str) -> None:
    tier, at_least = tier_by_need({"small": 0.2, "medium": 0.3, "large": 0.5}, RANKS, mode)  # type: ignore[arg-type]
    assert tier == expected
    assert at_least == pytest.approx({"small": 1.0, "medium": 0.8, "large": 0.5})


def test_tier_by_need_cost_stays_small_when_unsure() -> None:
    assert tier_by_need({"small": 0.4, "medium": 0.35, "large": 0.25}, RANKS, "cost")[0] == "small"
    assert tier_by_need({"small": 0.4, "medium": 0.35, "large": 0.25}, RANKS, "accuracy")[0] == "medium"


def test_classifier_decision_carries_tier_evidence_and_signals() -> None:
    decision = resolve(
        {"channel_type": "sports", "duration": "<30s"}, SPORTS[0], RuleSet(), TIERS, classifier=_trained_classifier()
    )
    assert decision.tier_scores_kind == "P(needs ≥ tier)"
    assert decision.tier_scores is not None and set(decision.tier_scores) == {"small", "medium", "large"}
    assert decision.signals["path"] == "domain classifier"
    assert decision.signals["metadata_used"] == ["channel_type"]
    assert decision.signals["metadata_ignored"] == ["duration"]


def test_catalog_decision_lists_the_shortlist_and_honours_several_providers() -> None:
    catalog = ModelCatalog(
        entries=(
            CatalogEntry("g-s", "gemini", "small", cost=1),
            CatalogEntry("a-s", "anthropic", "small", cost=2),
            CatalogEntry("o-s", "openai", "small", cost=3),
        )
    )
    rules = RuleSet(rules=(Rule.create({"channel_type": "news"}, "small"),))
    decision = resolve(
        {"channel_type": "news"}, "x", rules, TIERS, catalog=catalog, provider="openai,anthropic"
    )
    assert decision.model_name == "a-s"
    assert [c["id"] for c in decision.candidates] == ["a-s", "o-s"]
    assert decision.pool_size == 2
    assert decision.signals["providers"] == ["anthropic", "openai"]


def test_empty_request_uses_fallback_not_confident_classifier() -> None:
    decision = resolve({}, "", RuleSet(), TIERS, classifier=_trained_classifier())
    assert decision.tier == "medium"
    assert decision.confidence == 0.0
    assert "empty request" in decision.reason
    assert decision.fallback_outcome == "used 'medium'"


# -----------------------------------------------------------------------------------------
# P2 — optimize_for threading (PLAN-v2.md §5). Two tests enumerated in §5.4.
# -----------------------------------------------------------------------------------------


_P2_CATALOG = ModelCatalog(
    entries=(
        CatalogEntry("cheap-small", "gemini", "small", cost=1),
        CatalogEntry("good-small", "gemini", "small", cost=5),
    )
)


def test_resolve_threads_optimize_for_through_to_catalog() -> None:
    # quality_lookup keys are (model_id, metadata[quality_field] value) — here quality_field
    # is "channel_type", so the second tuple element is "sports" (the value), not the field
    # name. Accuracy mode must actually pick the higher-rated model; if resolve() silently
    # drops optimize_for, this assertion fails.
    rules = RuleSet(rules=(Rule.create({"channel_type": "sports"}, "small"),))
    quality = {("cheap-small", "sports"): 2.0, ("good-small", "sports"): 4.5}
    cost_decision = resolve(
        {"channel_type": "sports"},
        "x",
        rules,
        TIERS,
        catalog=_P2_CATALOG,
        quality=quality,
        quality_field="channel_type",
        optimize_for="cost",
    )
    assert cost_decision.model_name == "cheap-small"
    assert cost_decision.optimize_for == "cost"
    accuracy_decision = resolve(
        {"channel_type": "sports"},
        "x",
        rules,
        TIERS,
        catalog=_P2_CATALOG,
        quality=quality,
        quality_field="channel_type",
        optimize_for="accuracy",
    )
    assert accuracy_decision.model_name == "good-small"
    assert accuracy_decision.optimize_for == "accuracy"


def test_rule_optimize_for_overrides_request_optimize_for_without_catalog() -> None:
    # Without a catalog, the tier ladder still runs and a rule's optimize_for pin wins.
    rule = Rule.create(
        {"channel_type": "sports"}, "small", optimize_for="accuracy"
    )
    rules = RuleSet(rules=(rule,))
    decision = resolve(
        {"channel_type": "sports"},
        "x",
        rules,
        TIERS,
        optimize_for="cost",
    )
    assert decision.tier == "small"
    assert decision.optimize_for == "accuracy"
    assert decision.model_name == "g-small"


# -----------------------------------------------------------------------------------------
# P3 — metadata.modality → needs_video / needs_audio (PLAN-v2.md §6.4).
# -----------------------------------------------------------------------------------------


_P3_CATALOG = ModelCatalog(
    entries=(
        CatalogEntry("text-small", "gemini", "small", cost=1, supports_video=False, supports_audio=False),
        CatalogEntry("video-small", "gemini", "small", cost=5, supports_video=True, supports_audio=False),
        CatalogEntry("audio-small", "gemini", "small", cost=4, supports_video=False, supports_audio=True),
    )
)


def test_metadata_modality_video_sets_needs_video_on_pick() -> None:
    # Convention: metadata["modality"] == "video" → catalog.pick(needs_video=True).
    # Without that filter, cost mode would pick text-small; with it, video-small wins.
    rules = RuleSet(rules=(Rule.create({"channel_type": "sports"}, "small"),))
    plain = resolve({"channel_type": "sports"}, "x", rules, TIERS, catalog=_P3_CATALOG)
    assert plain.model_name == "text-small"
    video = resolve(
        {"channel_type": "sports", "modality": "video"},
        "x",
        rules,
        TIERS,
        catalog=_P3_CATALOG,
    )
    assert video.model_name == "video-small"


def test_metadata_modality_audio_sets_needs_audio_on_pick() -> None:
    rules = RuleSet(rules=(Rule.create({"channel_type": "sports"}, "small"),))
    decision = resolve(
        {"channel_type": "sports", "modality": "AUDIO"},  # case-insensitive via canonical_value
        "x",
        rules,
        TIERS,
        catalog=_P3_CATALOG,
    )
    assert decision.model_name == "audio-small"


def test_resolve_catalog_never_routes_unreviewed_even_if_selected() -> None:
    # Hybrid: UNREVIEWED stays out of pick(); ready at the decided tier wins.
    from gradist.tiers import UNREVIEWED_TIER

    catalog = ModelCatalog(
        entries=(
            CatalogEntry("pending", "gemini", UNREVIEWED_TIER, cost=1),
            CatalogEntry("ready", "gemini", "small", cost=9),
        )
    )
    rules = RuleSet(rules=(Rule.create({"task": "video_processing"}, "small"),))
    decision = resolve(
        {"task": "video_processing"},
        "x",
        rules,
        TIERS,
        catalog=catalog,
        selected_models={"pending", "ready"},
    )
    assert decision.model_name == "ready"
    assert decision.tier == "small"
    assert "task='video_processing'" in decision.reason
