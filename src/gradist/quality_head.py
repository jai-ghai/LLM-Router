"""Generic-text quality head: predict each tier's expected score, then take the cheapest tier
within a margin of the best prediction.

Trained on outcome data — every candidate model answered the query and was scored — rather than
on tier labels, so it estimates "how well would each tier do here", and `optimize_for` becomes a
dial over that estimate instead of a hard class boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge

from gradist.catalog import OptimizeFor
from gradist.features import Embedder

# Replay on 900 held-out xRouteBench queries (experiments/train_quality_head.py), relative to
# always-large: 0.03 ≈ 99.5% quality at 78% cost, 0.06 ≈ 96% at 66%, 0.10 ≈ 92% at 49%.
MARGINS: dict[str, float] = {"accuracy": 0.03, "balanced": 0.06, "cost": 0.10}


class QualityHeadError(ValueError):
    pass


class TierQualityHead:
    """`tiers` is ordered cheapest → most capable; that order is what "cheapest within margin" uses.

    The text embedder is supplied at construction and again at load(), never pickled — the same
    contract as TierClassifier.
    """

    def __init__(self, text_embedder: Embedder, tiers: Sequence[str]) -> None:
        if len(tiers) < 2:
            raise QualityHeadError("a quality head needs at least two tiers")
        self._text_embedder = text_embedder
        self.tiers: tuple[str, ...] = tuple(tiers)
        self._reg: Ridge | None = None

    def fit(self, texts: Sequence[str], scores: np.ndarray, alpha: float = 3.0) -> None:
        """`scores[i, j]` is the observed score of tier `self.tiers[j]` on `texts[i]`, in [0, 1]."""
        y = np.asarray(scores, dtype=np.float64)
        if len(texts) == 0:
            raise QualityHeadError("fit needs at least one example")
        if y.ndim != 2 or y.shape != (len(texts), len(self.tiers)):
            raise QualityHeadError(f"scores must have shape ({len(texts)}, {len(self.tiers)}); got {y.shape}")
        x = np.asarray(self._text_embedder.embed(list(texts)), dtype=np.float64)
        self._reg = Ridge(alpha=alpha).fit(x, y)

    def predict_scores(self, text: str) -> dict[str, float]:
        if self._reg is None:
            raise QualityHeadError("quality head is not fitted")
        x = np.asarray(self._text_embedder.embed([text]), dtype=np.float64)
        row = self._reg.predict(x)[0]
        return {tier: float(np.clip(score, 0.0, 1.0)) for tier, score in zip(self.tiers, row, strict=True)}

    def choose(self, text: str, optimize_for: OptimizeFor) -> tuple[str, dict[str, float], float]:
        """(tier, predicted scores, margin used)."""
        margin = MARGINS[optimize_for]
        scores = self.predict_scores(text)
        best = max(scores.values())
        tier = next(t for t in self.tiers if scores[t] >= best - margin)
        return tier, scores, margin

    def save(self, path: str | Path) -> None:
        joblib.dump({"tiers": list(self.tiers), "reg": self._reg}, path)

    @classmethod
    def load(cls, path: str | Path, text_embedder: Embedder) -> TierQualityHead:
        """joblib uses pickle — only load files you saved yourself."""
        state = joblib.load(path)
        head = cls(text_embedder, state["tiers"])
        head._reg = state["reg"]
        return head
