from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gradist.classifier import TierClassifier
from gradist.features import HashingEmbedder, MetadataEncoder
from gradist.labels import TierLabel
from gradist.quality_head import QualityHeadError, TierQualityHead
from gradist.resolve import has_domain_signal, resolve
from gradist.rules import Rule, RuleSet
from gradist.tiers import TierSpec, TierTable

TIERS = TierTable(
    tiers=(
        TierSpec("small", "", {"p": "m-small"}),
        TierSpec("medium", "", {"p": "m-medium"}),
        TierSpec("large", "", {"p": "m-large"}),
    ),
    default_provider="p",
    low_confidence_fallback="medium",
)
EASY = ["what is two plus two", "capital of france", "spell the word cat", "what colour is the sky"]
HARD = [
    "prove the riemann hypothesis for this zeta variant",
    "derive the closed form of this recursive integral",
    "count lattice paths avoiding the diagonal modulo prime",
    "find all integer solutions of this diophantine system",
]


def _head() -> TierQualityHead:
    head = TierQualityHead(HashingEmbedder(), ("small", "medium", "large"))
    scores = np.array([[0.95, 0.96, 0.97]] * len(EASY) + [[0.10, 0.45, 0.90]] * len(HARD))
    head.fit(EASY + HARD, scores, alpha=0.01)
    return head


def _flat_head() -> TierQualityHead:
    head = TierQualityHead(HashingEmbedder(), ("small", "medium", "large"))
    head.fit(["q"] * 3, np.array([[0.80, 0.88, 0.90]] * 3), alpha=0.01)
    return head


def test_easy_text_goes_cheap_and_hard_text_goes_large() -> None:
    head = _head()
    assert head.choose(EASY[0], "cost")[0] == "small"
    assert head.choose(HARD[0], "accuracy")[0] == "large"


def test_margin_is_the_cost_quality_dial() -> None:
    head = _flat_head()
    assert head.choose("q", "accuracy")[0] == "medium"  # within 0.03 of the best
    assert head.choose("q", "cost")[0] == "small"  # within 0.10 of the best


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    head = _head()
    path = tmp_path / "head.joblib"
    head.save(path)
    loaded = TierQualityHead.load(path, HashingEmbedder())
    assert loaded.tiers == head.tiers
    assert loaded.predict_scores(HARD[1]) == pytest.approx(head.predict_scores(HARD[1]))


def test_fit_rejects_mismatched_scores() -> None:
    with pytest.raises(QualityHeadError, match="shape"):
        TierQualityHead(HashingEmbedder(), ("small", "large")).fit(["a", "b"], np.zeros((2, 3)))


def test_unfitted_head_refuses_to_predict() -> None:
    with pytest.raises(QualityHeadError, match="not fitted"):
        TierQualityHead(HashingEmbedder(), ("small", "large")).predict_scores("x")


def test_domain_signal_ignores_task_and_weak_keys() -> None:
    assert not has_domain_signal({"task": "news_analysis", "modality": "text", "provider": "openai"})
    assert has_domain_signal({"channel_type": "news"})
    assert has_domain_signal({}, image_bytes=b"\xff\xd8")


def _domain_classifier() -> TierClassifier:
    clf = TierClassifier(MetadataEncoder({"channel_type": ["news", "sports"]}), HashingEmbedder())
    clf.fit(
        [
            TierLabel("n", {"channel_type": "news"}, "election bill", "large", "human", ""),
            TierLabel("s", {"channel_type": "sports"}, "match goal", "medium", "human", ""),
        ]
    )
    return clf


def test_resolve_uses_head_for_plain_text() -> None:
    decision = resolve({}, HARD[0], RuleSet(), TIERS, classifier=_domain_classifier(), quality_head=_head())
    assert decision.tier == "large"
    assert decision.confidence is None
    assert decision.reason.startswith("quality head (text only)")


def test_resolve_uses_domain_classifier_when_metadata_present() -> None:
    decision = resolve(
        {"channel_type": "news"}, EASY[0], RuleSet(), TIERS, classifier=_domain_classifier(), quality_head=_head()
    )
    assert decision.reason.startswith("classifier, confidence")


def test_rule_still_beats_the_head() -> None:
    rules = RuleSet(rules=(Rule.create({"task": "video_processing"}, "large"),))
    decision = resolve({"task": "video_processing"}, EASY[0], rules, TIERS, quality_head=_head())
    assert decision.tier == "large"
    assert decision.reason == "rule task=video_processing"


def test_optimize_for_moves_the_head_pick() -> None:
    head = _flat_head()
    cheap = resolve({}, "q", RuleSet(), TIERS, quality_head=head, optimize_for="cost")
    careful = resolve({}, "q", RuleSet(), TIERS, quality_head=head, optimize_for="accuracy")
    assert (cheap.tier, careful.tier) == ("small", "medium")
