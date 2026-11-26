from __future__ import annotations

import random

import pytest

from gradist.features import HashingEmbedder
from gradist.neighbours import (
    MIN_RATED_REQUESTS,
    SHRINK,
    Neighbour,
    NeighbourIndex,
    ReviewedRow,
    rows_from_reviews,
    vote,
)

HINDI = [
    "मुख्यमंत्री ने {n} जिलों में नई सड़क योजना की घोषणा की",
    "राज्य सरकार ने {n} गांवों के लिए पानी की योजना शुरू की",
    "विधानसभा में {n} नए विधेयक पेश किए गए, विपक्ष ने विरोध किया",
]
SPORTS = [
    "Cricket: India beat Australia by {n} runs in the final over",
    "Football: the striker scored {n} goals in the derby match",
    "Tennis: the champion won the final in {n} straight sets",
]


def reviews(consistent: bool) -> list[ReviewedRow]:
    """30 Hindi and 30 sports requests. Consistent: hindi-model is good on Hindi, bad on sports,
    sports-model the opposite. Otherwise the scores are random."""
    rng = random.Random(7)
    rows = []
    for i in range(30):
        for kind, templates in (("hi", HINDI), ("sp", SPORTS)):
            text = templates[i % 3].format(n=i + 2)
            good = "hindi-model" if kind == "hi" else "sports-model"
            bad = "sports-model" if kind == "hi" else "hindi-model"
            for model in (good, bad):
                score = (5 if model == good else 1) if consistent else rng.choice([1, 5])
                rows.append(ReviewedRow(f"{kind}{i}", text, model, float(score)))
    return rows


def test_a_vote_is_the_similarity_weighted_score_shrunk_by_how_few_neighbours_agree() -> None:
    votes, counts = vote([Neighbour("a", 0.9, "m", 5.0), Neighbour("b", 0.6, "m", 1.0), Neighbour("c", 0.5, "n", 5.0)])
    assert counts == {"m": 2, "n": 1}
    assert votes["m"] == pytest.approx((0.9 * 1 + 0.6 * -1) / 1.5 * 2 / (2 + SHRINK))
    assert votes["n"] == pytest.approx(1 / (1 + SHRINK))


def test_auto_mode_stays_off_until_enough_requests_are_reviewed() -> None:
    index = NeighbourIndex(HashingEmbedder())
    few = reviews(consistent=True)[:20]
    result = index.assess("मुख्यमंत्री ने नई योजना की घोषणा की", few)
    assert not result.active and result.votes == {}
    assert f"of {MIN_RATED_REQUESTS} reviewed requests" in result.status
    assert not index.assess("anything", []).active
    assert NeighbourIndex(HashingEmbedder(), "off").assess("x", few).status.startswith("off")


def test_consistent_reviews_switch_it_on_and_similar_requests_vote_for_the_model_that_did_well() -> None:
    index = NeighbourIndex(HashingEmbedder())
    rows = reviews(consistent=True)
    hindi = index.assess("मुख्यमंत्री ने 40 जिलों में नई सड़क योजना की घोषणा की", rows)
    assert hindi.active, hindi.status
    assert hindi.agreement is not None and hindi.agreement >= 0.9
    assert hindi.votes["hindi-model"] > 0 > hindi.votes["sports-model"]
    sports = index.assess("Cricket: India beat England by 40 runs in the final over", rows)
    assert sports.votes["sports-model"] > 0 > sports.votes["hindi-model"]
    assert sports.as_signal()["neighbours"]


def test_random_reviews_fail_the_held_out_check_and_stay_off() -> None:
    result = NeighbourIndex(HashingEmbedder()).assess("मुख्यमंत्री ने नई योजना की घोषणा की", reviews(consistent=False))
    assert not result.active
    assert result.votes == {}


def test_comparisons_count_as_five_and_one_and_a_tie_as_three_each() -> None:
    rows = rows_from_reviews(
        [("r1", "m1", 4), ("unknown", "m1", 5)],
        [("r1", "m2", "m1", "m1", "m2"), ("r2", None, None, "m1", "m3")],
        {"r1": "text one", "r2": "text two"},
    )
    assert [(r.request_id, r.model_id, r.score) for r in rows] == [
        ("r1", "m1", 4.0),
        ("r1", "m2", 5.0),
        ("r1", "m1", 1.0),
        ("r2", "m1", 3.0),
        ("r2", "m3", 3.0),
    ]
