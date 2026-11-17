from __future__ import annotations

import pytest

from tierrouter.quality import QualityError, QualityRating, quality_lookup


def test_rating_must_be_1_to_5() -> None:
    QualityRating("m", {}, rating=1, judged_by="alex")
    QualityRating("m", {}, rating=5, judged_by="alex")
    with pytest.raises(QualityError, match="1-5"):
        QualityRating("m", {}, rating=0, judged_by="alex")
    with pytest.raises(QualityError, match="1-5"):
        QualityRating("m", {}, rating=6, judged_by="alex")


def test_rating_requires_a_reviewer() -> None:
    with pytest.raises(QualityError, match="judged_by"):
        QualityRating("m", {}, rating=3, judged_by="  ")


def test_quality_lookup_averages_per_model_and_field_value() -> None:
    ratings = [
        QualityRating("gemini-small", {"task": "qc_review"}, rating=4, judged_by="a"),
        QualityRating("gemini-small", {"task": "qc_review"}, rating=2, judged_by="b"),
        QualityRating("gemini-small", {"task": "headline"}, rating=5, judged_by="a"),
        QualityRating("claude-small", {"task": "qc_review"}, rating=5, judged_by="a"),
    ]
    scores = quality_lookup(ratings, "task")
    assert scores[("gemini-small", "qc_review")] == 3.0
    assert scores[("gemini-small", "headline")] == 5.0
    assert scores[("claude-small", "qc_review")] == 5.0


def test_quality_lookup_is_case_and_key_insensitive() -> None:
    ratings = [QualityRating("m", {"TASK": "QC_Review"}, rating=4, judged_by="a")]
    scores = quality_lookup(ratings, "task")
    assert scores[("m", "qc_review")] == 4.0


def test_quality_lookup_skips_ratings_missing_the_field() -> None:
    ratings = [QualityRating("m", {"other": "x"}, rating=4, judged_by="a")]
    assert quality_lookup(ratings, "task") == {}
