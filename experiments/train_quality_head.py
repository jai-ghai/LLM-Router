"""Fit the generic-text quality head on xRouteBench outcomes and replay-evaluate each mode.

Writes data/quality_head.joblib, which the service loads at startup. Run
`experiments/probe_xroutebench.py` first to download data/xroutebench/.
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.train_from_hf_content import TEXT_MODEL
from experiments.train_from_xroutebench import DATA, TIERS, load_pool, per_query_outcomes
from gradist.features import SentenceTransformerEmbedder
from gradist.quality_head import MARGINS, TierQualityHead


def main() -> None:
    pool = load_pool()
    train_q = per_query_outcomes("train", pool)
    test_q = [r for r in per_query_outcomes("test", pool) if r["label"]]
    random.Random(11).shuffle(test_q)
    test_q = test_q[:900]

    head = TierQualityHead(SentenceTransformerEmbedder(TEXT_MODEL), TIERS)
    head.fit([r["text"] for r in train_q], np.array([[r["perf"][t] for t in TIERS] for r in train_q]))
    path = DATA / "quality_head.joblib"
    head.save(path)
    print(f"fitted on {len(train_q)} queries → {path}")

    def replay(pick: list[str]) -> tuple[float, float]:
        perf = sum(r["perf"][t] for r, t in zip(test_q, pick, strict=True)) / len(test_q)
        cost = sum(r["cost"][t] for r, t in zip(test_q, pick, strict=True)) / len(test_q)
        return perf, cost

    large_perf, large_cost = replay(["large"] * len(test_q))
    scores = [head.predict_scores(r["text"]) for r in test_q]
    print(f"\nreplay on {len(test_q)} held-out queries (relative to always-large)")
    for mode, margin in MARGINS.items():
        pick = [next(t for t in TIERS if s[t] >= max(s.values()) - margin) for s in scores]
        perf, cost = replay(pick)
        print(
            f"  {mode:9s} margin={margin:.2f}  quality {100 * perf / large_perf:5.1f}%  "
            f"cost {100 * cost / large_cost:5.1f}%  {dict(Counter(pick))}"
        )


if __name__ == "__main__":
    main()
