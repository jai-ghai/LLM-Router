"""Human-declared quality ratings: did a model's REAL output, for a REAL request, actually turn
out correct?

Capability (catalog.py/capability.py) tells you what a model COULD do and what it costs. It
cannot tell you how well THIS model performs on YOUR tasks — no public benchmark can, because it
was never evaluated on your data. This module is the project's answer: a human who reviewed the
real output rates it, keyed by whichever request metadata field matters for comparing like with
like (a deployment sets this once, e.g. "task" — see ARCHITECTURE.md). Cost still drives the
default choice; a quality floor is opt-in, for cases (the video-QC example) where a wrong answer
costs more than a cheap model saves.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from gradist.rules import canonical_key, canonical_value


class QualityError(ValueError):
    """A quality rating is malformed."""


@dataclass(frozen=True)
class QualityRating:
    model_id: str
    metadata: Mapping[str, str]
    rating: int
    """1-5. Validated at construction, not just at the API boundary, so a bad value can never
    quietly skew an average."""
    judged_by: str
    notes: str = ""

    def __post_init__(self) -> None:
        if not 1 <= self.rating <= 5:
            raise QualityError(f"rating must be 1-5, got {self.rating}")
        if not self.judged_by.strip():
            raise QualityError("judged_by is required — a rating with no reviewer is not a review")


def quality_lookup(ratings: Sequence[QualityRating], field: str) -> dict[tuple[str, str], float]:
    """Every (model_id, metadata[field] value) pair with at least one rating, averaged. Built
    fresh from the store's full rating history each time — cheap at the scale this project
    targets (one trusted operator, not a high-volume production path).
    """
    field = canonical_key(field)
    buckets: dict[tuple[str, str], list[int]] = {}
    for r in ratings:
        raw_value = next((v for k, v in r.metadata.items() if canonical_key(k) == field), None)
        if raw_value is None:
            continue
        value = canonical_value(raw_value)
        buckets.setdefault((r.model_id, value), []).append(r.rating)
    return {key: sum(values) / len(values) for key, values in buckets.items()}
