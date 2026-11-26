"""Similar past requests vote for the models that did well on them (k-nearest neighbours).

Ratings in feedback.py move a model's skills, so every request needing that skill feels them.
This adds what skills cannot: a judgement for requests like THIS one. Each reviewed request is a
row (request text, model, score 1–5); comparisons count as the winner 5, the loser 1, a tie 3
each. For a new request the K most similar rows above MIN_SIMILARITY vote:

    vote(model) = Σ similarity × (score − 3) / 2  ÷  Σ similarity   × n / (n + SHRINK)

so a vote is in −1..1 and one neighbour counts for a quarter of what many agreeing ones do.
fit.py adds SIMILAR_WEIGHT × vote as the "similar" term.

It only switches on ("auto") once there are MIN_RATED_REQUESTS reviewed requests and a
leave-one-out check shows the votes agree with the held-out scores often enough. Before that it
reports why it is off. GRADIST_NEIGHBOURS=on / off overrides the check.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from gradist.features import Embedder, Vector

K = 5
MIN_SIMILARITY = 0.3
SHRINK = 3.0
SIMILAR_WEIGHT = 0.5
MIN_RATED_REQUESTS = 50
MIN_CHECKED = 20
MIN_AGREEMENT = 0.6
MODES = ("auto", "on", "off")


@dataclass(frozen=True)
class ReviewedRow:
    request_id: str
    text: str
    model_id: str
    score: float
    """1–5: a star rating, or 5 / 1 / 3 for a comparison's winner / loser / tie."""


@dataclass(frozen=True)
class Neighbour:
    request_id: str
    similarity: float
    model_id: str
    score: float


@dataclass(frozen=True)
class NeighbourVotes:
    active: bool
    status: str
    votes: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    neighbours: list[Neighbour] = field(default_factory=list)
    agreement: float | None = None

    def as_signal(self) -> dict[str, object]:
        return {
            "active": self.active,
            "status": self.status,
            "agreement": None if self.agreement is None else round(self.agreement, 2),
            "votes": {m: {"vote": round(v, 3), "neighbours": self.counts.get(m, 0)} for m, v in self.votes.items()},
            "neighbours": [
                {"similarity": round(n.similarity, 2), "model_id": n.model_id, "score": n.score} for n in self.neighbours
            ],
        }


def vote(neighbours: Sequence[Neighbour]) -> tuple[dict[str, float], dict[str, int]]:
    sums: dict[str, float] = {}
    weights: dict[str, float] = {}
    counts: dict[str, int] = {}
    for n in neighbours:
        sums[n.model_id] = sums.get(n.model_id, 0.0) + n.similarity * (n.score - 3.0) / 2.0
        weights[n.model_id] = weights.get(n.model_id, 0.0) + n.similarity
        counts[n.model_id] = counts.get(n.model_id, 0) + 1
    votes = {m: sums[m] / weights[m] * counts[m] / (counts[m] + SHRINK) for m in sums if weights[m] > 0}
    return votes, counts


class NeighbourIndex:
    """Embeds reviewed requests once (cached by request id) and finds the nearest ones."""

    def __init__(self, embedder: Embedder, mode: str = "auto") -> None:
        if mode not in MODES:
            raise ValueError(f"GRADIST_NEIGHBOURS must be one of {', '.join(MODES)}, got '{mode}'")
        self.embedder = embedder
        self.mode = mode
        self._vectors: dict[str, Vector] = {}
        self._check: tuple[int, float | None, int] | None = None
        """(rows checked against, agreement, rows that had a vote) — redone when rows change."""

    def assess(self, text: str, rows: Sequence[ReviewedRow]) -> NeighbourVotes:
        requests = {r.request_id for r in rows}
        if self.mode == "off":
            return NeighbourVotes(False, "off (GRADIST_NEIGHBOURS=off)")
        if not text.strip():
            return NeighbourVotes(False, "no request text to compare")
        if not rows:
            return NeighbourVotes(False, f"off: no reviewed requests yet (needs {MIN_RATED_REQUESTS})")
        agreement: float | None = None
        if self.mode == "auto":
            if len(requests) < MIN_RATED_REQUESTS:
                return NeighbourVotes(False, f"off: {len(requests)} of {MIN_RATED_REQUESTS} reviewed requests")
            agreement, checked = self._agreement(rows)
            if checked < MIN_CHECKED or agreement is None:
                return NeighbourVotes(False, f"off: only {checked} held-out reviews had similar requests (needs {MIN_CHECKED})")
            if agreement < MIN_AGREEMENT:
                return NeighbourVotes(
                    False, f"off: votes agreed with {agreement:.0%} of held-out reviews (needs {MIN_AGREEMENT:.0%})", agreement=agreement
                )
        query = self.embedder.embed([text])[0]
        neighbours = self._nearest(query, rows, exclude=None)
        votes, counts = vote(neighbours)
        how = "on (GRADIST_NEIGHBOURS=on)" if self.mode == "on" else f"on: votes agreed with {agreement:.0%} of held-out reviews"
        status = f"{how}; {len(neighbours)} similar reviewed request(s)" if neighbours else f"{how}; no similar reviewed request"
        return NeighbourVotes(True, status, votes, counts, neighbours, agreement)

    def _matrix(self, rows: Sequence[ReviewedRow]) -> Vector:
        missing = [r for r in rows if r.request_id not in self._vectors]
        unique = {r.request_id: r.text for r in missing}
        if unique:
            vectors = self.embedder.embed(list(unique.values()))
            for request_id, vector in zip(unique, vectors, strict=True):
                self._vectors[request_id] = vector
        return np.vstack([self._vectors[r.request_id] for r in rows])

    def _nearest(self, query: Vector, rows: Sequence[ReviewedRow], exclude: str | None) -> list[Neighbour]:
        matrix = self._matrix(rows)
        norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(query) or 1.0)
        similarity = matrix @ query / np.where(norms == 0, 1.0, norms)
        order = np.argsort(-similarity)
        found: list[Neighbour] = []
        for i in order:
            row = rows[int(i)]
            if row.request_id == exclude:
                continue
            if similarity[i] < MIN_SIMILARITY or len(found) == K:
                break
            found.append(Neighbour(row.request_id, float(similarity[i]), row.model_id, row.score))
        return found

    def _agreement(self, rows: Sequence[ReviewedRow]) -> tuple[float | None, int]:
        """Leave one request out: does the vote for its model point the same way as its score?"""
        if self._check is not None and self._check[0] == len(rows):
            return self._check[1], self._check[2]
        matrix = self._matrix(rows)
        agree = checked = 0
        for row, vector in zip(rows, matrix, strict=True):
            if row.score == 3:
                continue
            votes, _ = vote(self._nearest(vector, rows, exclude=row.request_id))
            if row.model_id not in votes or votes[row.model_id] == 0:
                continue
            checked += 1
            agree += (votes[row.model_id] > 0) == (row.score > 3)
        agreement = agree / checked if checked else None
        self._check = (len(rows), agreement, checked)
        return agreement, checked


def rows_from_reviews(
    ratings: Sequence[tuple[str, str, int]],
    preferences: Sequence[tuple[str, str | None, str | None, str, str]],
    texts: dict[str, str],
) -> list[ReviewedRow]:
    """ratings: (request_id, model_id, rating); preferences: (request_id, winner, loser, picked,
    other) with winner/loser None for a tie. Rows whose request text is unknown are skipped."""
    rows: list[ReviewedRow] = []
    for request_id, model_id, rating in ratings:
        if texts.get(request_id):
            rows.append(ReviewedRow(request_id, texts[request_id], model_id, float(rating)))
    for request_id, winner, loser, picked, other in preferences:
        text = texts.get(request_id)
        if not text:
            continue
        if winner and loser:
            rows += [ReviewedRow(request_id, text, winner, 5.0), ReviewedRow(request_id, text, loser, 1.0)]
        else:
            rows += [ReviewedRow(request_id, text, picked, 3.0), ReviewedRow(request_id, text, other, 3.0)]
    return rows
