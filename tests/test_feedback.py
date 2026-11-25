from __future__ import annotations

import pytest

from gradist.catalog import CatalogEntry
from gradist.feedback import PRIOR_WEIGHT, blend, learn_skills
from gradist.fit import CatalogScale, model_skills
from gradist.quality import QualityRating

SCALE = CatalogScale(top_capability=60.0)


def rated(model_id: str, rating: int, needs: dict[str, float] | None) -> QualityRating:
    return QualityRating(model_id, {"task": "news"}, rating, "reviewer", needs=needs)


def test_ratings_count_toward_the_skills_the_request_needed_weighted_by_need() -> None:
    learned = learn_skills(
        [
            rated("m", 1, {"hindi": 1.0, "audio": 0.5, "fast": 1.0}),
            rated("m", 5, {"hindi": 1.0}),
            rated("other", 3, None),
        ]
    )
    assert set(learned) == {"m"}
    assert learned["m"]["hindi"].mean == 0.5
    assert learned["m"]["hindi"].weight == 2.0
    assert learned["m"]["hindi"].ratings == 2
    assert learned["m"]["audio"].mean == 0.0
    assert learned["m"]["audio"].weight == 0.5
    assert "fast" not in learned["m"]


def test_the_catalog_value_is_a_prior_worth_a_few_ratings() -> None:
    evidence = learn_skills([rated("m", 1, {"hindi": 1.0})] * 3)["m"]["hindi"]
    assert blend(0.8, evidence) == pytest.approx(PRIOR_WEIGHT * 0.8 / (PRIOR_WEIGHT + 3))


def test_bad_hindi_ratings_lower_the_hindi_skill_and_say_so() -> None:
    entry = CatalogEntry("m", "p", "small", cost=0.001, multilingual=0.8, multilingual_source="measured")
    learned = learn_skills([rated("m", 1, {"hindi": 1.0})] * 3)["m"]
    hindi = model_skills(entry, SCALE, learned=learned)["hindi"]
    assert hindi.source == "feedback"
    assert hindi.prior == 0.8
    assert hindi.ratings == 3
    assert hindi.value == pytest.approx(0.4)


def test_feedback_never_grants_an_unsupported_modality_or_changes_price() -> None:
    entry = CatalogEntry("m", "p", "small", cost=0.001, supports_video=False)
    learned = learn_skills([rated("m", 5, {"video": 1.0, "fast": 1.0})] * 5)["m"]
    skills = model_skills(entry, SCALE, learned=learned)
    assert skills["video"].value == 0.0
    assert skills["video"].source == "declared"
    assert skills["fast"].source == "price"


def test_a_bucket_rating_already_used_for_capability_is_not_blended_again() -> None:
    entry = CatalogEntry("m", "p", "small", cost=0.001, capability=30.0, capability_source="measured")
    learned = learn_skills([rated("m", 1, {"reasoning": 1.0})])["m"]
    skills = model_skills(entry, SCALE, rating=5.0, learned=learned)
    assert skills["reasoning"].source == "rating"
    assert skills["reasoning"].value == 1.0
