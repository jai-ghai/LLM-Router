"""Learn each model's skills from ratings of its real outputs.

A rating tied to a request (POST /v1/quality with `request_id`) carries that request's skill
needs. It counts as evidence for every skill the request needed, weighted by the need: a 1/5 on
a Hindi news clip (hindi 1.0, video 1.0, audio 0.7) mostly lowers the model's hindi and video
skills and its audio skill a little less.

The catalog value is the prior, worth PRIOR_WEIGHT fully-relevant ratings:

    learned = (PRIOR_WEIGHT × catalog value + Σ need × score) / (PRIOR_WEIGHT + Σ need)

with score = (rating − 1) / 4. Not learned: `fast` (it is price), and video / audio on a model
that does not support them (support is a fact, not an opinion).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from gradist.quality import QualityRating

PRIOR_WEIGHT = 3.0
NOT_LEARNED = frozenset({"fast"})


@dataclass(frozen=True)
class SkillEvidence:
    mean: float
    """Need-weighted average score, 0–1."""
    weight: float
    """Σ need over the ratings — how many fully-relevant ratings this is worth."""
    ratings: int


def learn_skills(ratings: Iterable[QualityRating]) -> dict[str, dict[str, SkillEvidence]]:
    """model id → skill → evidence, from ratings that carry the rated request's needs."""
    sums: dict[tuple[str, str], list[float]] = {}
    for rating in ratings:
        if not rating.needs:
            continue
        score = (rating.rating - 1) / 4
        for skill, need in rating.needs.items():
            if need <= 0 or skill in NOT_LEARNED:
                continue
            bucket = sums.setdefault((rating.model_id, skill), [0.0, 0.0, 0.0])
            bucket[0] += need * score
            bucket[1] += need
            bucket[2] += 1
    learned: dict[str, dict[str, SkillEvidence]] = {}
    for (model_id, skill), (weighted, weight, count) in sums.items():
        learned.setdefault(model_id, {})[skill] = SkillEvidence(weighted / weight, weight, int(count))
    return learned


def blend(prior: float, evidence: SkillEvidence) -> float:
    return (PRIOR_WEIGHT * prior + evidence.weight * evidence.mean) / (PRIOR_WEIGHT + evidence.weight)
