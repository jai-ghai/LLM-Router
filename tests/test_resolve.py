from __future__ import annotations

import pytest

from tierrouter.catalog import CatalogEntry, ModelCatalog
from tierrouter.classifier import TierClassifier
from tierrouter.features import HashingEmbedder, MetadataEncoder
from tierrouter.labels import TierLabel
from tierrouter.resolve import resolve, validate_setup
from tierrouter.rules import Rule, RuleError, RuleSet
from tierrouter.tiers import TierSpec, TierTable

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


def test_resolve_with_a_catalog_picks_the_cheapest_matching_model() -> None:
    rules = RuleSet(rules=(Rule.create({"channel_type": "sports"}, "small"),))
    decision = resolve({"channel_type": "sports"}, "x", rules, TIERS, catalog=CATALOG)
    assert decision.model_name == "cheap-small"


def test_resolve_with_a_catalog_honours_the_caller_s_selection() -> None:
    rules = RuleSet(rules=(Rule.create({"channel_type": "sports"}, "small"),))
    decision = resolve(
        {"channel_type": "sports"}, "x", rules, TIERS, catalog=CATALOG, selected_models={"pricey-small"}
    )
    assert decision.model_name == "pricey-small"
