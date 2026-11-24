"""3-tier classifier: a feature vector -> (tier, confidence), plus an evaluation helper.

Deliberately simple (logistic regression): the earlier project's research (MMR-Bench) found
simple classifiers over embeddings matched more complex ones. The confidence score, not model
complexity, is what a caller should use to decide whether to trust a prediction — see
tiers.yaml's `low_confidence_fallback`, applied in resolve.py (Task 6), not here.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from gradist.features import (
    Embedder,
    ImageEmbedder,
    MetadataEncoder,
    build_feature_vector,
    load_image_bytes,
)
from gradist.labels import TierLabel


class ClassifierError(ValueError):
    """The classifier was asked to do something it cannot: fit on no data, or predict unfitted."""


def _embedder_name(embedder: object) -> str:
    return str(getattr(embedder, "model_name", None) or type(embedder).__name__)


class TierClassifier:
    """fit(labels) once, then predict(metadata, text) -> (tier, confidence) many times.

    A single-tier label set (every label is the same tier) is a valid, if uninteresting, input:
    scikit-learn's LogisticRegression refuses to fit on one class, so that case is handled
    separately and always predicts that one tier with full confidence — same shape as the
    earlier project's LearnedRouter for exactly the same reason.

    The metadata encoder and text embedder are supplied by the caller, not saved inside the
    classifier: they are passed again at predict() and load() time, the same contract the
    earlier project's LearnedRouter used, since an embedder is not reliably picklable and
    swapping it must be an explicit, visible choice, not something frozen into a saved file.
    """

    def __init__(
        self,
        metadata_encoder: MetadataEncoder,
        text_embedder: Embedder,
        image_embedder: ImageEmbedder | None = None,
    ) -> None:
        self._metadata_encoder = metadata_encoder
        self._text_embedder = text_embedder
        self._image_embedder = image_embedder
        self._clf: Any = None
        self._constant: str | None = None

    def _vector(self, metadata: dict[str, str], text: str, image_bytes: bytes | None) -> Any:
        return build_feature_vector(
            metadata, text, self._metadata_encoder, self._text_embedder, image_bytes, self._image_embedder
        )

    def fit(self, labels: list[TierLabel]) -> None:
        if not labels:
            raise ClassifierError("fit needs at least one label")
        tiers = [label.tier for label in labels]
        if len(set(tiers)) == 1:
            self._constant = tiers[0]
            self._clf = None
            return
        x = np.stack(
            [
                self._vector(
                    label.metadata,
                    label.text,
                    load_image_bytes(label.image_path) if label.image_path else None,
                )
                for label in labels
            ]
        )
        clf = LogisticRegression(max_iter=1000)
        clf.fit(x, tiers)
        self._clf = clf
        self._constant = None

    def predict(
        self, metadata: dict[str, str], text: str, image_bytes: bytes | None = None
    ) -> tuple[str, float]:
        """(tier, confidence). confidence is the predicted class's probability, in [0, 1].

        `image_bytes` is the raw file content (as a browser upload arrives), not a path — unlike
        training, which reads `TierLabel.image_path` from disk. A predict call with no image is
        always valid, image_embedder or not; see build_feature_vector's "no image" flag.
        """
        if self._constant is not None:
            return self._constant, 1.0
        if self._clf is None:
            raise ClassifierError("classifier is not fitted; call fit() first")
        proba = self.predict_proba(metadata, text, image_bytes)
        tier = max(proba, key=lambda t: proba[t])
        return tier, proba[tier]

    def predict_proba(
        self, metadata: dict[str, str], text: str, image_bytes: bytes | None = None
    ) -> dict[str, float]:
        """Probability per tier the classifier was trained on (sums to 1)."""
        if self._constant is not None:
            return {self._constant: 1.0}
        if self._clf is None:
            raise ClassifierError("classifier is not fitted; call fit() first")
        x = self._vector(self._metadata_encoder.known_only(metadata), text, image_bytes)
        proba = self._clf.predict_proba(x.reshape(1, -1))[0]
        return {str(tier): float(p) for tier, p in zip(self._clf.classes_, proba, strict=True)}

    def explain(self, metadata: dict[str, str], text: str, image_bytes: bytes | None = None) -> dict[str, Any]:
        """What the classifier actually read from this request, and how much each input moved it.

        `proba` is the same distribution `predict_proba` returns. `without_image`,
        `without_text` and `without_metadata` re-score the request with that one part blanked
        (no image flag, empty text, every field unknown), so a caller can show which input the
        decision rests on. Blanked parts reuse the already-computed embeddings — the image is
        embedded once.
        """
        known = self._metadata_encoder.known_only(metadata)
        image_embedder = self._image_embedder
        report: dict[str, Any] = {
            "metadata_buckets": {k: self._metadata_encoder.bucket_label(k, v) for k, v in sorted(known.items())},
            "metadata_columns": self._metadata_encoder.width,
            "text_embedder": _embedder_name(self._text_embedder),
            "image_embedder": _embedder_name(image_embedder) if image_embedder is not None else None,
            "image_dims": image_embedder.width if image_embedder is not None else 0,
            "image_used": image_embedder is not None and image_bytes is not None,
        }
        meta_vec = self._metadata_encoder.encode(known)
        text_vec = self._text_embedder.embed([text])[0]
        report["text_dims"] = int(text_vec.shape[0])
        if self._constant is not None:
            constant = {self._constant: 1.0}
            report.update(proba=constant, without_image=constant, without_text=constant, without_metadata=constant)
            return report
        if self._clf is None:
            raise ClassifierError("classifier is not fitted; call fit() first")

        blank_image: list[Any] = []
        image_part: list[Any] = []
        if image_embedder is not None:
            blank_image = [np.zeros(image_embedder.width, dtype=np.float64), np.array([0.0])]
            image_part = (
                [image_embedder.embed([image_bytes])[0], np.array([1.0])] if image_bytes is not None else blank_image
            )

        def score(meta: Any, txt: Any, img: list[Any]) -> dict[str, float]:
            proba = self._clf.predict_proba(np.concatenate([meta, txt, *img]).reshape(1, -1))[0]
            return {str(t): float(p) for t, p in zip(self._clf.classes_, proba, strict=True)}

        report["proba"] = score(meta_vec, text_vec, image_part)
        report["without_image"] = score(meta_vec, text_vec, blank_image)
        report["without_text"] = score(meta_vec, self._text_embedder.embed([""])[0], image_part)
        report["without_metadata"] = score(self._metadata_encoder.encode({}), text_vec, image_part)
        return report

    @property
    def metadata_fields(self) -> list[str]:
        return self._metadata_encoder.field_names

    @property
    def uses_images(self) -> bool:
        """True when this classifier was built with an image embedder (so a predict call must

        either supply bytes it can use, or deliberately rely on the "no image" zero+flag path)."""
        return self._image_embedder is not None

    def bucket_for(self, field: str, value: str | None) -> str:
        """Pass-through to the metadata encoder, for evaluate_classifier's per-field report."""
        return self._metadata_encoder.bucket_label(field, value)

    def save(self, path: str | Path) -> None:
        joblib.dump({"clf": self._clf, "constant": self._constant}, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        metadata_encoder: MetadataEncoder,
        text_embedder: Embedder,
        image_embedder: ImageEmbedder | None = None,
    ) -> TierClassifier:
        """Load a saved classifier. `joblib` uses `pickle`, which can execute arbitrary code for

        a file from anywhere other than your own artifact store — only load files you saved.
        Pass the SAME `image_embedder` (or lack of one) that was used to fit it, or predictions
        will silently use a different-width feature vector than the one it was trained on.
        """
        state = joblib.load(path)
        instance = cls(metadata_encoder, text_embedder, image_embedder)
        instance._clf = state["clf"]
        instance._constant = state["constant"]
        return instance


@dataclass(frozen=True)
class EvalReport:
    n: int
    accuracy: float
    baseline_accuracy: float
    accuracy_by_bucket: dict[str, float]
    bucket_sizes: dict[str, int]


def evaluate_classifier(
    classifier: TierClassifier,
    labels: list[TierLabel],
    baseline_tier: str,
    metadata_field: str,
) -> EvalReport:
    """Classifier accuracy vs. always-`baseline_tier`, both overall and per `metadata_field`

    bucket (including "__unknown__"). Per PLAN.md §8: a classifier that cannot clearly beat the
    baseline here is not ready to decide anything.
    """
    if not labels:
        raise ClassifierError("evaluate_classifier needs at least one label")
    correct = 0
    baseline_correct = 0
    bucket_correct: dict[str, int] = defaultdict(int)
    bucket_totals: dict[str, int] = defaultdict(int)
    for label in labels:
        image_bytes = load_image_bytes(label.image_path) if label.image_path else None
        predicted, _ = classifier.predict(label.metadata, label.text, image_bytes)
        bucket = classifier.bucket_for(metadata_field, label.metadata.get(metadata_field))
        bucket_totals[bucket] += 1
        if predicted == label.tier:
            correct += 1
            bucket_correct[bucket] += 1
        if baseline_tier == label.tier:
            baseline_correct += 1
    return EvalReport(
        n=len(labels),
        accuracy=correct / len(labels),
        baseline_accuracy=baseline_correct / len(labels),
        accuracy_by_bucket={b: bucket_correct[b] / bucket_totals[b] for b in bucket_totals},
        bucket_sizes=dict(bucket_totals),
    )
