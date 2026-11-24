from __future__ import annotations

from pathlib import Path

import pytest

from gradist.classifier import ClassifierError, TierClassifier, evaluate_classifier
from gradist.features import HashingEmbedder, MetadataEncoder
from gradist.labels import TierLabel

NEWS = ["parliament passed the budget bill", "the minister announced an election", "court verdict"]
SPORTS = ["goal scored in the final match", "the striker won the tournament", "wicket and a century"]


def _encoder() -> MetadataEncoder:
    return MetadataEncoder({"channel_type": ["news", "sports"]})


def _labels(texts: list[str], channel: str, tier: str, source: str = "human") -> list[TierLabel]:
    return [
        TierLabel(f"{channel}-{i}", {"channel_type": channel}, text, tier, source, "test")
        for i, text in enumerate(texts)
    ]


def _trained() -> TierClassifier:
    clf = TierClassifier(_encoder(), HashingEmbedder())
    clf.fit(_labels(NEWS, "news", "small") + _labels(SPORTS, "sports", "large"))
    return clf


def test_fit_needs_at_least_one_label() -> None:
    with pytest.raises(ClassifierError):
        TierClassifier(_encoder(), HashingEmbedder()).fit([])


def test_predict_before_fit_raises() -> None:
    with pytest.raises(ClassifierError):
        TierClassifier(_encoder(), HashingEmbedder()).predict({"channel_type": "news"}, "text")


def test_single_tier_label_set_predicts_that_tier_with_full_confidence() -> None:
    clf = TierClassifier(_encoder(), HashingEmbedder())
    clf.fit(_labels(NEWS, "news", "medium"))
    tier, confidence = clf.predict({"channel_type": "sports"}, "anything at all")
    assert (tier, confidence) == ("medium", 1.0)


def test_classifier_separates_two_clearly_different_topics() -> None:
    clf = _trained()
    assert clf.predict({"channel_type": "news"}, "parliament budget election")[0] == "small"
    assert clf.predict({"channel_type": "sports"}, "the goal in the match final")[0] == "large"


def test_confidence_is_a_real_probability() -> None:
    clf = _trained()
    _, confidence = clf.predict({"channel_type": "news"}, "parliament budget election")
    assert 0.0 <= confidence <= 1.0


def test_unknown_metadata_value_still_predicts_from_text() -> None:
    clf = _trained()
    tier, _ = clf.predict({"channel_type": "weather"}, "the goal in the match final")
    assert tier == "large"  # text alone still carries the signal


def test_predict_ignores_fields_the_classifier_was_not_trained_on() -> None:
    clf = _trained()
    tier, _ = clf.predict({"channel_type": "sports", "duration": "<30s"}, "the goal in the match final")
    assert tier == "large"


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    clf = _trained()
    path = tmp_path / "clf.joblib"
    clf.save(path)
    loaded = TierClassifier.load(path, _encoder(), HashingEmbedder())
    request = ({"channel_type": "news"}, "parliament budget election")
    assert loaded.predict(*request) == clf.predict(*request)


def test_save_and_load_round_trip_for_the_single_tier_case(tmp_path: Path) -> None:
    clf = TierClassifier(_encoder(), HashingEmbedder())
    clf.fit(_labels(NEWS, "news", "medium"))
    path = tmp_path / "clf.joblib"
    clf.save(path)
    loaded = TierClassifier.load(path, _encoder(), HashingEmbedder())
    assert loaded.predict({"channel_type": "news"}, "x") == ("medium", 1.0)


def test_bucket_for_matches_the_encoder() -> None:
    clf = _trained()
    assert clf.bucket_for("channel_type", "news") == "news"
    assert clf.bucket_for("channel_type", "weather") == "__unknown__"


def test_evaluate_classifier_reports_overall_and_per_bucket_accuracy() -> None:
    clf = _trained()
    labels = (
        _labels(NEWS, "news", "small")
        + _labels(SPORTS, "sports", "large")
        + _labels(["unrelated weather text"], "weather", "small")
    )
    report = evaluate_classifier(clf, labels, baseline_tier="small", metadata_field="channel_type")
    assert report.n == 7
    assert report.accuracy == pytest.approx(6 / 7)  # the lone weather/unknown example is wrong
    assert report.baseline_accuracy == pytest.approx(4 / 7)  # 3 news + 1 weather are "small"
    assert report.accuracy_by_bucket["news"] == 1.0
    assert report.accuracy_by_bucket["sports"] == 1.0
    assert report.accuracy_by_bucket["__unknown__"] == 0.0
    assert report.bucket_sizes == {"news": 3, "sports": 3, "__unknown__": 1}


def test_evaluate_classifier_needs_at_least_one_label() -> None:
    with pytest.raises(ClassifierError):
        evaluate_classifier(_trained(), [], baseline_tier="small", metadata_field="channel_type")
