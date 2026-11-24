# [Fact-Forcing preamble]
#
# Callers: CLI only — `python experiments/train_from_video_clipper_db.py`.
# Existing files: train_from_tabson_db.py (gemini telemetry, no images);
# train_from_video_clips.py (local drop-folder clips). Neither reads MySQL
# video_clipper + frame_details + channel_master for thumbnail training.
# Reads (RO): video_clipper, frame_details, channel_master, optional v_gemini_call.
# Writes: data/thumbnails/clipper_*.jpg, data/images/ex-*.jpg, merges
# data/examples.jsonl + judgments.jsonl, refits data/classifier.joblib (Dinov2).
# User: "okay .. you find the data of clips in video clipper table .. use that data
# and extract thumbnail and judge you will extract extarct output of channel name
# and other stuff .. and train the model accordingly ."
"""Pull Tabson video_clipper frames + channel metadata → image-aware Gradist train."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid
from collections import Counter, defaultdict
from pathlib import Path

from gradist.classifier import TierClassifier, evaluate_classifier
from gradist.features import Dinov2Embedder, HashingEmbedder, MetadataEncoder
from gradist.labels import TierLabel

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
IMAGES = DATA / "images"
THUMBS = DATA / "thumbnails"
VALID_TIERS = {"small", "medium", "large"}

# Local imports after ROOT is known (script run as `python experiments/...`).
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.train_from_tabson_db import MODEL_TO_TIER, connect  # noqa: E402

# Clip analysis defaults (tabson-python-inferencing STAGE1 / MODEL_ID).
GENRE_TASK_TIER: dict[str, tuple[str, str]] = {
    "news": ("video_processing", "large"),
    "business news": ("video_processing", "large"),
    "ent news": ("video_processing", "large"),
    "sports": ("video_processing", "large"),
    "gec": ("playout_observation", "medium"),
    "movies": ("playout_observation", "medium"),
    "music": ("playout_observation", "medium"),
    "kids": ("playout_observation", "medium"),
}


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


def _download(url: str, timeout: int = 40) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "gradist-clipper-train/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _judge(genre: str, channel_model_tier: str | None) -> tuple[str, str, str]:
    """Genre → task/tier for clip analysis.

    Do NOT use channel-wide gemini telemetry majority: helper purposes (spell/headline)
    dominate and would label STAGE1 clip work as small/flash-lite incorrectly.
    `channel_model_tier` is accepted for logging/experiments but ignored for labeling.
    """
    del channel_model_tier  # reserved; see docstring
    g = (genre or "news").strip().lower()
    task, tier = GENRE_TASK_TIER.get(g, ("video_processing", "large"))
    return task, tier, f"genre-default genre={g} → {task}/{tier}"


def _channel_telemetry_tiers(cur, limit_calls: int = 80_000) -> dict[str, str]:
    cur.execute(
        """
        SELECT channel_code, model, COUNT(*) AS n
        FROM v_gemini_call
        WHERE status = 'success'
          AND call_id > (SELECT MAX(call_id) - %s FROM v_gemini_call)
          AND channel_code IS NOT NULL
          AND model IS NOT NULL
        GROUP BY channel_code, model
        """,
        (limit_calls,),
    )
    votes: dict[str, Counter[str]] = defaultdict(Counter)
    for row in cur.fetchall():
        tier = MODEL_TO_TIER.get(str(row["model"]))
        if not tier:
            continue
        votes[str(row["channel_code"])][tier] += int(row["n"])
    return {ch: counter.most_common(1)[0][0] for ch, counter in votes.items() if counter}


def fetch_clip_rows(cur, *, per_channel: int = 3, max_channels: int = 40) -> list[dict]:
    """Per-channel light queries (avoids one giant scan over frame_details)."""
    cur.execute(
        """
        SELECT channel_code, channel_name, genre, network, language_id, market, cluster
        FROM channel_master
        WHERE genre IN ('News', 'Business News', 'Ent News', 'Sports')
          AND channel_code IN (SELECT DISTINCT channel_code FROM video_clipper)
        ORDER BY channel_name
        LIMIT %s
        """,
        (max_channels,),
    )
    channels = cur.fetchall()
    selected: list[dict] = []
    for ch in channels:
        cur.execute(
            """
            SELECT id AS clip_id, channel_code, mp4url, sub_story_status, playout_status
            FROM video_clipper
            WHERE channel_code = %s
              AND id < (SELECT MAX(id) - 50 FROM video_clipper WHERE channel_code = %s)
            ORDER BY id DESC
            LIMIT %s
            """,
            (ch["channel_code"], ch["channel_code"], per_channel * 4),
        )
        clips = cur.fetchall()
        got = 0
        for clip in clips:
            if got >= per_channel:
                break
            cur.execute(
                """
                SELECT image_url, image_order_no
                FROM frame_details
                WHERE video_clipper_id = %s
                  AND image_url LIKE 'https://%%'
                  AND image_order_no BETWEEN 90 AND 200
                ORDER BY ABS(image_order_no - 140) ASC
                LIMIT 1
                """,
                (clip["clip_id"],),
            )
            fr = cur.fetchone()
            if not fr:
                continue
            selected.append(
                {
                    **clip,
                    **fr,
                    "channel_name": ch["channel_name"],
                    "genre": ch["genre"],
                    "network": ch["network"],
                    "language_id": ch["language_id"],
                    "market": ch["market"],
                    "cluster": ch["cluster"],
                }
            )
            got += 1
        print(f"channel {ch['channel_name']}: {got} clips with frames")
    return selected


def build_examples(rows: list[dict], channel_tiers: dict[str, str]) -> tuple[list[dict], list[dict]]:
    THUMBS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    examples: list[dict] = []
    judgments: list[dict] = []
    failures = 0
    for row in rows:
        clip_id = int(row["clip_id"])
        url = str(row["image_url"])
        try:
            raw = _download(url)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"skip clip={clip_id}: download failed: {exc}")
            failures += 1
            continue
        if len(raw) < 500:
            print(f"skip clip={clip_id}: tiny image ({len(raw)} bytes)")
            failures += 1
            continue

        channel_code = str(row["channel_code"])
        channel_name = str(row["channel_name"] or f"channel_{channel_code}")
        genre = str(row["genre"] or "News")
        network = str(row["network"] or "")
        task, tier, notes = _judge(genre, channel_tiers.get(channel_code))
        channel_type = genre.strip().lower().replace(" ", "_")

        example_id = f"ex-{uuid.uuid4().hex[:12]}"
        thumb_path = THUMBS / f"clipper_{clip_id}.jpg"
        image_path = IMAGES / f"{example_id}.jpg"
        thumb_path.write_bytes(raw)
        image_path.write_bytes(raw)

        text = (
            f"channel={channel_name}; genre={genre}; network={network}; "
            f"market={row.get('market') or ''}; cluster={row.get('cluster') or ''}; "
            f"clip_id={clip_id}; task={task}"
        )
        metadata = {
            "pipeline": "news" if "news" in channel_type else channel_type,
            "task": task,
            "purpose": task,
            "channel_code": channel_code,
            "channel_type": channel_type,
            "channel_name": channel_name,
            "genre": genre,
            "network": network,
            "modality": "video",
        }
        examples.append(
            {
                "example_id": example_id,
                "metadata": metadata,
                "text": text,
                "image_path": str(image_path),
            }
        )
        judgments.append(
            {
                "example_id": example_id,
                "tier": tier,
                "judged_by": "video-clipper-db",
                "notes": f"{notes}; channel={channel_name}",
            }
        )
        print(f"ok clip={clip_id} channel={channel_name!r} tier={tier} bytes={len(raw)}")

    print(f"downloaded={len(examples)} failed={failures}")
    return examples, judgments


def _infer_fields(examples: list[dict]) -> dict[str, list[str]]:
    keys = (
        "pipeline",
        "task",
        "purpose",
        "channel_type",
        "channel_code",
        "channel_name",
        "genre",
        "network",
        "modality",
    )
    fields: dict[str, list[str]] = {}
    for key in keys:
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
        if j["example_id"] in by_id and j["tier"] in VALID_TIERS
    ]
    if len(labels) < 2:
        raise SystemExit(f"need ≥2 labels; have {len(labels)}")
    fields = _infer_fields(examples)
    uses_images = any(lbl.image_path for lbl in labels)
    print(f"fitting classifier n={len(labels)} uses_images={uses_images} …")
    image_embedder = Dinov2Embedder() if uses_images else None
    clf = TierClassifier(MetadataEncoder(fields), HashingEmbedder(), image_embedder)
    clf.fit(labels)
    path = DATA / "classifier.joblib"
    clf.save(path)
    report = evaluate_classifier(clf, labels, baseline_tier="medium", metadata_field="channel_name")
    print(
        f"trained → {path} | n={report.n} accuracy={report.accuracy:.3f} "
        f"baseline={report.baseline_accuracy:.3f} uses_images={uses_images}"
    )


def main() -> None:
    conn = connect()
    try:
        cur = conn.cursor()
        print("loading channel telemetry tiers…")
        channel_tiers = _channel_telemetry_tiers(cur)
        print(f"telemetry channels with tier: {len(channel_tiers)}")
        print("sampling video_clipper + frame_details…")
        rows = fetch_clip_rows(cur, per_channel=3, max_channels=40)
        print(f"candidate rows: {len(rows)} across {len({r['channel_code'] for r in rows})} channels")
    finally:
        conn.close()

    if not rows:
        raise SystemExit("no clipper+frame rows found in recent window")

    new_ex, new_ju = build_examples(rows, channel_tiers)
    if len(new_ex) < 2:
        raise SystemExit("too few thumbnails downloaded")

    existing_ex = _load_jsonl(DATA / "examples.jsonl")
    existing_ju = _load_jsonl(DATA / "judgments.jsonl")
    examples = existing_ex + new_ex
    judgments = existing_ju + new_ju
    _write_jsonl(DATA / "examples.jsonl", examples)
    _write_jsonl(DATA / "judgments.jsonl", judgments)
    print(f"store examples={len(examples)} judgments={len(judgments)} (new={len(new_ex)})")
    train(examples, judgments)


if __name__ == "__main__":
    main()
