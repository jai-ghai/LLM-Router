"""Does on-screen text (OCR) help the domain classifier? Held-out comparison.

Builds the same labels and feature vector as POST /v1/train, once with the frame's OCR text
appended to the request text and once without, and scores both by stratified 5-fold
cross-validation. Accuracy is reported on all labels and on the rows that carry a frame (the
only rows OCR can change). OCR comes from data/ocr_cache.json, which a training run with
GRADIST_OCR=1 fills; frames missing from the cache are read now.

Usage:
    python experiments/eval_ocr.py [--data data]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from gradist.classifier import with_screen_text
from gradist.features import (
    Dinov2Embedder,
    MetadataEncoder,
    SentenceTransformerEmbedder,
    load_image_bytes,
)
from gradist.labels import labels_from_rules, merge_labels
from gradist.ocr import ScreenReader, join_lines
from gradist.service.app import MINILM, _infer_fields
from gradist.service.store import Store
from gradist.tiers import load_tiers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--tiers", default="config/tiers.example.yaml")
    args = parser.parse_args()

    store = Store(Path(args.data))
    tiers = load_tiers(args.tiers)
    examples = store.list_examples()
    labels = merge_labels(labels_from_rules(examples, store.load_rules()), store.list_judgments(tiers.names))
    encoder = MetadataEncoder(_infer_fields(examples) or {"_unused": []})
    text_embedder = SentenceTransformerEmbedder(MINILM)
    image_embedder = Dinov2Embedder()
    reader = ScreenReader(cache_path=store.root / "ocr_cache.json")

    has_frame = np.array([bool(lbl.image_path) for lbl in labels])
    frames = {i: load_image_bytes(lbl.image_path) for i, lbl in enumerate(labels) if lbl.image_path}
    screen = {i: join_lines(reader.read(b)) for i, b in frames.items()}
    reader.save_cache()

    meta = np.stack([encoder.encode(encoder.known_only(lbl.metadata)) for lbl in labels])
    image = np.zeros((len(labels), image_embedder.width + 1))
    for i, b in frames.items():
        image[i, :-1] = image_embedder.embed([b])[0]
        image[i, -1] = 1.0
    plain = text_embedder.embed([lbl.text for lbl in labels])
    with_ocr = text_embedder.embed(
        [with_screen_text(lbl.text, screen.get(i, "")) for i, lbl in enumerate(labels)]
    )
    y = np.array([lbl.tier for lbl in labels])

    print(
        f"{len(labels)} labels, {int(has_frame.sum())} with a frame, "
        f"{sum(1 for s in screen.values() if s)} frames with readable text"
    )
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(meta, y))
    for name, text_part in (("without OCR", plain), ("with OCR", with_ocr)):
        x = np.hstack([meta, text_part, image])
        pred = np.empty_like(y)
        for train_idx, test_idx in folds:
            clf = LogisticRegression(max_iter=1000).fit(x[train_idx], y[train_idx])
            pred[test_idx] = clf.predict(x[test_idx])
        correct = pred == y
        print(
            f"{name:12} held-out accuracy: all {correct.mean():.3f} · "
            f"frame rows {correct[has_frame].mean():.3f}"
        )


if __name__ == "__main__":
    main()
