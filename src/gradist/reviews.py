"""Reviews beyond star ratings, collected now so later models can learn from them.

- Preference: for one request, was the picked model's output better than another model's, worse,
  or the same? Outputs are produced outside the router; this only records the verdict.
- NeedLabel: what the request really needed, per skill (0–1), as corrected by a reviewer.

Both are validated at construction, like QualityRating, so a bad row never reaches a learner.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from gradist.skills import SKILLS

OUTCOMES = ("picked", "other", "tie")


class ReviewError(ValueError):
    """A preference or need label is malformed."""


def check_skills(skills: Sequence[str]) -> tuple[str, ...]:
    unknown = [s for s in skills if s not in SKILLS]
    if unknown:
        raise ReviewError(f"unknown skill(s) {', '.join(unknown)} (known: {', '.join(SKILLS)})")
    return tuple(dict.fromkeys(skills))


@dataclass(frozen=True)
class Preference:
    request_id: str
    picked: str
    other: str
    outcome: str
    """"picked" (the router's pick was better), "other" (the other model was better) or "tie"."""
    judged_by: str
    skills: tuple[str, ...] = ()
    """Where the difference showed, when the reviewer said."""
    needs: Mapping[str, float] | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ReviewError("request_id is required")
        if not self.picked or not self.other:
            raise ReviewError("both models are required")
        if self.picked == self.other:
            raise ReviewError("compare the pick with a different model")
        if self.outcome not in OUTCOMES:
            raise ReviewError(f"outcome must be one of {', '.join(OUTCOMES)}, got '{self.outcome}'")
        if not self.judged_by.strip():
            raise ReviewError("judged_by is required")
        object.__setattr__(self, "skills", check_skills(self.skills))

    @property
    def winner(self) -> str | None:
        return {"picked": self.picked, "other": self.other}.get(self.outcome)

    @property
    def loser(self) -> str | None:
        return {"picked": self.other, "other": self.picked}.get(self.outcome)


@dataclass(frozen=True)
class NeedLabel:
    request_id: str
    needs: Mapping[str, float]
    judged_by: str
    rule_needs: Mapping[str, float] = field(default_factory=dict)
    """What the rules said, kept so a learner can see the correction, not just the answer."""

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ReviewError("request_id is required")
        if not self.judged_by.strip():
            raise ReviewError("judged_by is required")
        check_skills(list(self.needs))
        for skill, value in self.needs.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ReviewError(f"need for '{skill}' must be between 0 and 1, got {value}")
