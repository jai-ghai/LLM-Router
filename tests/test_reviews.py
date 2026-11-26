from __future__ import annotations

import pytest

from gradist.reviews import NeedLabel, Preference, ReviewError


def test_a_preference_names_its_winner_and_loser() -> None:
    other_won = Preference("r1", "picked-model", "other-model", "other", "ana", ("hindi", "hindi"))
    assert (other_won.winner, other_won.loser) == ("other-model", "picked-model")
    assert other_won.skills == ("hindi",)
    tie = Preference("r1", "picked-model", "other-model", "tie", "ana")
    assert (tie.winner, tie.loser) == (None, None)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"other": "picked-model"}, "different model"),
        ({"outcome": "maybe"}, "outcome must be"),
        ({"judged_by": " "}, "judged_by"),
        ({"skills": ("poetry",)}, "unknown skill"),
        ({"request_id": ""}, "request_id"),
    ],
)
def test_bad_preferences_are_rejected(fields: dict[str, object], message: str) -> None:
    values: dict[str, object] = {
        "request_id": "r1", "picked": "picked-model", "other": "other-model", "outcome": "tie", "judged_by": "ana"
    }
    with pytest.raises(ReviewError, match=message):
        Preference(**{**values, **fields})  # type: ignore[arg-type]


def test_need_labels_must_name_known_skills_between_zero_and_one() -> None:
    assert NeedLabel("r1", {"hindi": 1.0}, "ana").needs == {"hindi": 1.0}
    with pytest.raises(ReviewError, match="unknown skill"):
        NeedLabel("r1", {"poetry": 0.5}, "ana")
    with pytest.raises(ReviewError, match="between 0 and 1"):
        NeedLabel("r1", {"hindi": 1.2}, "ana")
