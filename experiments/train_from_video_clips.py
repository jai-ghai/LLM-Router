# [Fact-Forcing preamble]
#
# Callers: CLI only — `python experiments/train_from_video_clips.py` (no in-repo import yet).
# Existing files: experiments/train_from_tabson_db.py is text/metadata-only (image_path=None);
# experiments/extract_clip_thumbnails.py only extracts frames — neither merges clip thumbnails
# into Gradist examples/judgments and refits an image-aware classifier.
# Reads: data/video_clips/*.mp4|… + optional data/video_clips/manifest.jsonl;
# data/examples.jsonl + judgments.jsonl (merge). Writes: data/thumbnails/*.png,
# appends image examples/judgments, refits data/classifier.joblib (Dinov2 when available).
# User: "ha yeh kar lo ..mai video clips dunga or tum thumbnail nikl kar train kar lena ..okay ?"
"""Extract thumbnails from dropped clips → labeled Gradist examples → image-aware train."""

from __future__ import annotations

import argparse
import json
import re
import uuid
from pathlib import Path

from experiments.extract_clip_thumbnails import CLIPS, THUMBS, extract_all, list_clips
from gradist.classifier import TierClassifier, evaluate_classifier
from gradist.features import Dinov2Embedder, HashingEmbedder, MetadataEncoder
from gradist.labels import TierLabel

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MANIFEST = CLIPS / "manifest.jsonl"
VALID_TIERS = {"small", "medium", "large"}

# Filename convention when no manifest row:
#   news__v3_single__large__clip01.mp4
_NAME_RE = re.compile(
    r"^(?P<channel_type>[^_]+)__(?P<task>[^_]+)__(?P<tier>small|medium|large)(?:__.+)?$",
    re.IGNORECASE,
)


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


def _load_manifest(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in _load_jsonl(path):
        name = str(row.get("file") or row.get("clip") or "").strip()
        if not name:
            continue
        out[Path(name).name] = row
    return out


def _meta_from_filename(stem: str) -> dict[str, str] | None:
    m = _NAME_RE.match(stem)
    if not m:
        return None
    channel_type = m.group("channel_type").lower()
    task = m.group("task").lower()
    tier = m.group("tier").lower()
    return {
        "channel_type": channel_type,
        "pipeline": channel_type if channel_type in {"news", "playout", "sports"} else "news",
        "task": task,
        "purpose": task,
        "tier": tier,
    }


def _label_for_clip(clip: Path, manifest: dict[str, dict]) -> dict[str, str]:
    row = manifest.get(clip.name)
    if row:
        tier = str(row.get("tier", "")).lower()
        if tier not in VALID_TIERS:
            raise SystemExit(f"{clip.name}: manifest tier must be small|medium|large, got {tier!r}")
        task = str(row.get("task") or row.get("purpose") or "video_processing")
        channel_type = str(row.get("channel_type") or row.get("pipeline") or "news")
        pipeline = str(row.get("pipeline") or channel_type)
        text = str(row.get("text") or f"{channel_type} {task} clip {clip.stem}")
        return {
            "tier": tier,
            "task": task,
            "purpose": str(row.get("purpose") or task),
            "pipeline": pipeline,
            "channel_type": channel_type,
            "channel_code": str(row.get("channel_code") or "clip"),
            "text": text,
            "modality": "video",
        }
    parsed = _meta_from_filename(clip.stem)
    if parsed is None:
        raise SystemExit(
            f"{clip.name}: add a row to {MANIFEST.relative_to(ROOT)} or rename to "
            "channel__task__tier__name.mp4 (e.g. news__v3_single__large__ep01.mp4)"
        )
    return {
        **{k: v for k, v in parsed.items() if k != "tier"},
        "tier": parsed["tier"],
        "channel_code": "clip",
        "text": f"{parsed['channel_type']} {parsed['task']} clip {clip.stem}",
        "modality": "video",
    }


def _infer_fields(examples: list[dict]) -> dict[str, list[str]]:
    keys = ("pipeline", "task", "purpose", "channel_type", "channel_code", "modality")
    fields: dict[str, list[str]] = {}
    for key in keys:
        vals = sorted({str(e["metadata"].get(key, "")) for e in examples if e["metadata"].get(key)})
        if vals:
            fields[key] = vals
    return fields or {"_unused": []}


def merge_clip_examples(thumbs: list[dict], manifest: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    by_clip = {Path(t["clip"]).name: t for t in thumbs}
    new_examples: list[dict] = []
    new_judgments: list[dict] = []
    for clip in list_clips(CLIPS):
        thumb_info = by_clip.get(clip.name)
        if thumb_info is None:
            raise SystemExit(f"no thumbnail for {clip.name}; run extract first")
        label = _label_for_clip(clip, manifest)
        example_id = f"ex-{uuid.uuid4().hex[:12]}"
        thumb_path = ROOT / thumb_info["thumbnail"]
        images_dir = DATA / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        image_path = images_dir / f"{example_id}.png"
        image_path.write_bytes(thumb_path.read_bytes())
        meta = {
            "pipeline": label["pipeline"],
            "task": label["task"],
            "purpose": label["purpose"],
            "channel_type": label["channel_type"],
            "channel_code": label["channel_code"],
            "modality": label["modality"],
        }
        new_examples.append(
            {
                "example_id": example_id,
                "metadata": meta,
                "text": label["text"],
                "image_path": str(image_path),
            }
        )
        new_judgments.append(
            {
                "example_id": example_id,
                "tier": label["tier"],
                "judged_by": "video-clip-thumb",
                "notes": f"thumbnail from {clip.name}",
            }
        )
    return new_examples, new_judgments


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
        if j["example_id"] in by_id and j["tier"] in VALID_TIERS
    ]
    if len(labels) < 2:
        raise SystemExit(f"need ≥2 labeled examples to train; have {len(labels)}")
    fields = _infer_fields(examples)
    encoder = MetadataEncoder(fields)
    uses_images = any(lbl.image_path for lbl in labels)
    image_embedder = Dinov2Embedder() if uses_images else None
    clf = TierClassifier(encoder, HashingEmbedder(), image_embedder)
    clf.fit(labels)
    path = DATA / "classifier.joblib"
    clf.save(path)
    report = evaluate_classifier(clf, labels, baseline_tier="medium", metadata_field="task")
    print(
        f"trained → {path} | n={report.n} accuracy={report.accuracy:.3f} "
        f"uses_images={uses_images}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replace-text-only",
        action="store_true",
        help="drop prior examples that have no image_path before merging clip rows",
    )
    args = parser.parse_args()

    CLIPS.mkdir(parents=True, exist_ok=True)
    clips = list_clips(CLIPS)
    if not clips:
        raise SystemExit(
            f"No clips in {CLIPS}. Drop videos there (and optional manifest.jsonl), then re-run."
        )

    print(f"clips: {len(clips)}")
    thumbs = extract_all(CLIPS, THUMBS)
    print(f"thumbnails: {len(thumbs)}")

    manifest = _load_manifest(MANIFEST)
    new_ex, new_ju = merge_clip_examples(thumbs, manifest)

    existing_ex = _load_jsonl(DATA / "examples.jsonl")
    existing_ju = _load_jsonl(DATA / "judgments.jsonl")
    if args.replace_text_only:
        keep_ids = {e["example_id"] for e in existing_ex if e.get("image_path")}
        existing_ex = [e for e in existing_ex if e["example_id"] in keep_ids]
        existing_ju = [j for j in existing_ju if j["example_id"] in keep_ids]

    examples = existing_ex + new_ex
    judgments = existing_ju + new_ju
    _write_jsonl(DATA / "examples.jsonl", examples)
    _write_jsonl(DATA / "judgments.jsonl", judgments)
    print(f"examples={len(examples)} judgments={len(judgments)} (new clips={len(new_ex)})")
    train(examples, judgments)


if __name__ == "__main__":
    main()
