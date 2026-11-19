"""Outcome labels from tabson QC: did the generator's sub-story need a substantive fix?

Inputs are only what is known BEFORE generation (transcript, channel, language, program type,
duration) — never the generated fields themselves, which would leak the answer. The label is 1
when the QC model (gemini-3-flash-preview + RAG) applied a correction to a substantive field.

Writes data/qc_outcomes.jsonl, then reports how predictable the label is on a time-ordered
holdout (train on older sub-stories, test on the newest). Read-only against MySQL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.inspect_tabson_db import connect
from experiments.train_from_hf_content import TEXT_MODEL
from gradist.features import MetadataEncoder, SentenceTransformerEmbedder

OUT = ROOT / "data" / "qc_outcomes.jsonl"
SUBSTANTIVE = (
    "story_genre",
    "substory_genre",
    "substory_category",
    "story_format",
    "telecast_format",
    "substory",
    "geography",
)

QUERY = f"""
SELECT p.substory_id, p.captured_at, p.language_id, p.program_type, p.duration_seconds,
       LEFT(p.text_content, 1500) AS text_content,
       cm.channel_name, cm.genre AS channel_genre, cm.network, cm.cluster AS channel_cluster,
       EXISTS(SELECT 1 FROM sub_story_qc_suggestions q WHERE q.substory_id = p.substory_id) AS qc_seen,
       (SELECT COUNT(*) FROM sub_story_qc_suggestions q
         WHERE q.substory_id = p.substory_id AND q.status = 'applied'
           AND q.field_name IN ({",".join(repr(f) for f in SUBSTANTIVE)})) AS fixes
FROM sub_story_pre_qc p
LEFT JOIN channel_master cm ON cm.channel_code = p.channel_code
WHERE p.id > %s AND CHAR_LENGTH(p.text_content) > 80
ORDER BY p.id
"""


def duration_bucket(seconds: object) -> str:
    s = int(seconds or 0)
    return "<30s" if s < 30 else "30s-2m" if s < 120 else "2m-5m" if s < 300 else "5m+"


def pull(window: int) -> list[dict]:
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT MAX(id) AS m FROM sub_story_pre_qc")
        top = int(cur.fetchone()["m"])
        cur.execute(QUERY, (top - window,))
        rows = cur.fetchall()
    finally:
        conn.close()
    out = []
    for r in rows:
        out.append(
            {
                "substory_id": int(r["substory_id"]),
                "captured_at": str(r["captured_at"]),
                "text": " ".join(str(r["text_content"] or "").split()),
                "metadata": {
                    "channel_name": str(r["channel_name"] or ""),
                    "channel_genre": str(r["channel_genre"] or ""),
                    "network": str(r["network"] or ""),
                    "channel_cluster": str(r["channel_cluster"] or ""),
                    "language_id": str(r["language_id"] or ""),
                    "program_type": str(r["program_type"] or ""),
                    "duration": duration_bucket(r["duration_seconds"]),
                },
                "qc_seen": bool(r["qc_seen"]),
                "needs_fix": int(r["fixes"] or 0) > 0,
            }
        )
    return out


def evaluate(rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: r["captured_at"])
    cut = int(len(rows) * 0.8)
    train, test = rows[:cut], rows[cut:]
    fields: dict[str, set[str]] = {}
    for r in train:
        for k, v in r["metadata"].items():
            if v:
                fields.setdefault(k, set()).add(v)
    encoder = MetadataEncoder({k: sorted(v) for k, v in fields.items()})
    embedder = SentenceTransformerEmbedder(TEXT_MODEL)

    def features(batch: list[dict], with_text: bool, with_meta: bool) -> np.ndarray:
        parts = []
        if with_meta:
            parts.append(np.stack([encoder.encode(r["metadata"]) for r in batch]))
        if with_text:
            parts.append(np.asarray(embedder.embed([r["text"] for r in batch]), dtype=np.float64))
        return np.hstack(parts)

    y_tr = np.array([r["needs_fix"] for r in train])
    y_te = np.array([r["needs_fix"] for r in test])
    print(f"train={len(train)} (fix rate {y_tr.mean():.3f})  test={len(test)} (fix rate {y_te.mean():.3f})")
    for name, wt, wm in (("metadata only", False, True), ("text only", True, False), ("text + metadata", True, True)):
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        clf.fit(features(train, wt, wm), y_tr)
        proba = clf.predict_proba(features(test, wt, wm))[:, 1]
        auc = roc_auc_score(y_te, proba)
        top = np.argsort(-proba)[: max(1, len(test) // 5)]
        print(f"  {name:16s} AUC={auc:.3f}  fix rate in riskiest 20% = {y_te[top].mean():.3f} (base {y_te.mean():.3f})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=30000, help="most recent sub_story_pre_qc ids to scan")
    args = parser.parse_args()
    rows = pull(args.window)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    seen = sum(r["qc_seen"] for r in rows)
    print(f"pulled {len(rows)} sub-stories → {OUT}  (qc rows for {seen}, needs_fix {sum(r['needs_fix'] for r in rows)})")
    evaluate(rows)


if __name__ == "__main__":
    main()
