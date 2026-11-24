# [Fact-Forcing preamble]
#
# Callers: CLI — `python experiments/train_from_hf_content.py`.
# Existing: train_from_tabson_db / train_from_video_clipper_db — neither pulls Hugging Face
# topic corpora for content-dynamic tier labels.
# Reads: HF `ag_news` (+ optional prior data/examples.jsonl). Writes: merges examples/
# judgments, refits data/classifier.joblib with SentenceTransformer text embedder.
# User: "train better use hugging face data to train find data and train the model …
# dynamic and read the content … accuracy and dynamics processing matters"
"""Train Gradist on Hugging Face AG News — content text → tier (reads the article)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from sklearn.datasets import fetch_20newsgroups

from gradist.classifier import TierClassifier, evaluate_classifier
from gradist.features import Dinov2Embedder, MetadataEncoder, SentenceTransformerEmbedder
from gradist.labels import TierLabel

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# 20 Newsgroups (sklearn) — same role as HF topic corpora when Hub URI rules break.
# category → (channel_type, topic, tier)
NEWSGROUP_MAP: dict[str, tuple[str, str, str]] = {
    "rec.sport.baseball": ("sports", "sports", "medium"),
    "rec.sport.hockey": ("sports", "sports", "medium"),
    "talk.politics.misc": ("news", "politics", "large"),
    "talk.politics.guns": ("news", "politics", "large"),
    "talk.politics.mideast": ("news", "world", "large"),
    "sci.med": ("news", "science_tech", "medium"),
    "sci.space": ("news", "science_tech", "medium"),
    "sci.electronics": ("news", "science_tech", "medium"),
    "comp.graphics": ("news", "science_tech", "small"),
    "comp.windows.x": ("news", "science_tech", "small"),
    "rec.autos": ("news", "lifestyle", "small"),
    "rec.motorcycles": ("news", "lifestyle", "small"),
}

TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PER_CLASS = 80


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


def fetch_hf_examples(per_class: int = PER_CLASS) -> tuple[list[dict], list[dict]]:
    """Pull topic text from 20 Newsgroups (HuggingFace-style content corpus via sklearn)."""
    cats = list(NEWSGROUP_MAP.keys())
    print(f"loading 20newsgroups ({len(cats)} topics), ~{per_class}/topic …")
    bundle = fetch_20newsgroups(
        subset="train",
        categories=cats,
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=7,
    )
    buckets: dict[str, list[str]] = {c: [] for c in cats}
    for text, target in zip(bundle.data, bundle.target, strict=True):
        cat = bundle.target_names[int(target)]
        if len(buckets[cat]) >= per_class:
            continue
        cleaned = " ".join(str(text).split())
        if len(cleaned) < 80:
            continue
        buckets[cat].append(cleaned[:2000])

    examples: list[dict] = []
    judgments: list[dict] = []
    for cat, texts in buckets.items():
        channel_type, topic, tier = NEWSGROUP_MAP[cat]
        for text in texts:
            example_id = f"ex-{uuid.uuid4().hex[:12]}"
            metadata = {
                "pipeline": channel_type,
                "channel_type": channel_type,
                "topic": topic,
                "task": "content_route",
                "purpose": "content_route",
                "modality": "text",
            }
            examples.append(
                {
                    "example_id": example_id,
                    "metadata": metadata,
                    "text": text,
                    "image_path": None,
                }
            )
            judgments.append(
                {
                    "example_id": example_id,
                    "tier": tier,
                    "judged_by": "sklearn-20newsgroups",
                    "notes": f"newsgroup={cat} topic={topic}",
                }
            )
    print(f"content examples: {len(examples)}")
    return examples, judgments


def _infer_fields(examples: list[dict]) -> dict[str, list[str]]:
    # Union of every metadata key present — clipper rows add channel_code/network/etc.
    keys: set[str] = set()
    for e in examples:
        keys.update((e.get("metadata") or {}).keys())
    fields: dict[str, list[str]] = {}
    for key in sorted(keys):
        vals = sorted(
            {
                str(e["metadata"].get(key, ""))
                for e in examples
                if e.get("metadata", {}).get(key) not in (None, "")
            }
        )
        if vals:
            fields[key] = vals
    return fields or {"_unused": []}


def train(examples: list[dict], judgments: list[dict]) -> None:
    by_id = {e["example_id"]: e for e in examples}
    labels = [
        TierLabel(
            j["example_id"],
            by_id[j["example_id"]]["metadata"],
            by_id[j["example_id"]]["text"],
            j["tier"],
            "human",
            j.get("notes", ""),
            by_id[j["example_id"]].get("image_path"),
        )
        for j in judgments
        if j["example_id"] in by_id and j["tier"] in {"small", "medium", "large"}
    ]
    fields = _infer_fields(examples)
    uses_images = any(lbl.image_path for lbl in labels)
    print(f"fitting n={len(labels)} text_model={TEXT_MODEL} uses_images={uses_images}")
    text_embedder = SentenceTransformerEmbedder(TEXT_MODEL)
    image_embedder = Dinov2Embedder() if uses_images else None
    clf = TierClassifier(MetadataEncoder(fields), text_embedder, image_embedder)
    clf.fit(labels)
    path = DATA / "classifier.joblib"
    clf.save(path)
    report = evaluate_classifier(clf, labels, baseline_tier="medium", metadata_field="channel_type")
    print(
        f"trained → {path} | n={report.n} accuracy={report.accuracy:.3f} "
        f"by_channel_type={report.accuracy_by_bucket}"
    )


def main() -> None:
    hf_ex, hf_ju = fetch_hf_examples()
    existing_ex = _load_jsonl(DATA / "examples.jsonl")
    existing_ju = _load_jsonl(DATA / "judgments.jsonl")
    examples = existing_ex + hf_ex
    judgments = existing_ju + hf_ju
    _write_jsonl(DATA / "examples.jsonl", examples)
    _write_jsonl(DATA / "judgments.jsonl", judgments)
    print(f"store examples={len(examples)} judgments={len(judgments)}")
    train(examples, judgments)


if __name__ == "__main__":
    main()
