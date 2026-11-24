"""QC-risk escalation: how likely is the current generator's output to need a QC fix?

Trained on tabson outcomes (`sub_story_qc_suggestions` applied corrections joined to the
pre-generation inputs in `sub_story_pre_qc`). At resolve time a high-risk sub-story is escalated
to a more capable tier up front; it never lowers a tier and never overrides an operator rule.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from gradist.features import Embedder, MetadataEncoder

# A request must carry this to look like a tabson sub-story (the shape the head was trained on).
REQUIRED_FIELD = "channel_name"


class QcRiskError(ValueError):
    pass


class QcRiskHead:
    """`fields` (metadata vocabulary) and `threshold` are plain data, saved with the model; the text
    embedder is supplied again at load(), never pickled."""

    def __init__(self, fields: dict[str, list[str]], text_embedder: Embedder, escalate_to: str = "large") -> None:
        self.fields = {k: list(v) for k, v in fields.items()}
        self.escalate_to = escalate_to
        self.threshold: float | None = None
        self._encoder = MetadataEncoder(self.fields)
        self._text_embedder = text_embedder
        self._clf: LogisticRegression | None = None

    def vectors(self, metadata: Sequence[dict[str, str]], texts: Sequence[str]) -> np.ndarray:
        meta = np.stack([self._encoder.encode(m) for m in metadata])
        text = np.asarray(self._text_embedder.embed(list(texts)), dtype=np.float64)
        return np.hstack([meta, text])

    def fit_vectors(self, x: np.ndarray, needs_fix: Sequence[bool], escalate_share: float = 0.2) -> None:
        """Fit on precomputed `vectors(...)`; the threshold flags the riskiest `escalate_share`."""
        y = np.asarray(needs_fix, dtype=bool)
        if len(set(y.tolist())) < 2:
            raise QcRiskError("need both fixed and clean sub-stories to fit")
        self._clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(x, y)
        risk = self._clf.predict_proba(x)[:, 1]
        self.threshold = float(np.quantile(risk, 1.0 - escalate_share))

    def risk_of_vectors(self, x: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise QcRiskError("QC risk head is not fitted")
        return np.asarray(self._clf.predict_proba(x)[:, 1])

    def applies_to(self, metadata: dict[str, str], text: str) -> bool:
        return bool(text.strip()) and bool(str(metadata.get(REQUIRED_FIELD, "")).strip())

    def risk(self, metadata: dict[str, str], text: str) -> float:
        return float(self.risk_of_vectors(self.vectors([self._encoder.known_only(metadata)], [text]))[0])

    def save(self, path: str | Path) -> None:
        joblib.dump(
            {"fields": self.fields, "escalate_to": self.escalate_to, "threshold": self.threshold, "clf": self._clf},
            path,
        )

    @classmethod
    def load(cls, path: str | Path, text_embedder: Embedder) -> QcRiskHead:
        """joblib uses pickle — only load files you saved yourself."""
        state = joblib.load(path)
        head = cls(state["fields"], text_embedder, state["escalate_to"])
        head.threshold = state["threshold"]
        head._clf = state["clf"]
        return head
