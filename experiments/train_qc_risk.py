"""Fit the QC-risk escalation head on data/qc_outcomes.jsonl and save data/qc_risk.joblib.

Run `experiments/qc_outcome_dataset.py` first. Embeds once (cached), evaluates on a
time-ordered holdout, then refits on everything for the saved model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.train_from_hf_content import TEXT_MODEL
from gradist.features import SentenceTransformerEmbedder
from gradist.qc_risk import QcRiskHead

DATA = ROOT / "data"
ESCALATE_SHARE = 0.2


def main() -> None:
    rows = [json.loads(line) for line in (DATA / "qc_outcomes.jsonl").read_text(encoding="utf-8").splitlines()]
    rows.sort(key=lambda r: r["captured_at"])
    fields: dict[str, set[str]] = {}
    for r in rows:
        for k, v in r["metadata"].items():
            if v:
                fields.setdefault(k, set()).add(v)
    head = QcRiskHead({k: sorted(v) for k, v in fields.items()}, SentenceTransformerEmbedder(TEXT_MODEL))

    cache = DATA / "qc_outcomes_vectors.npz"
    if cache.exists() and np.load(cache)["x"].shape[0] == len(rows):
        x = np.load(cache)["x"]
    else:
        print(f"embedding {len(rows)} sub-stories…")
        x = head.vectors([r["metadata"] for r in rows], [r["text"] for r in rows])
        np.savez_compressed(cache, x=x)
    y = np.array([r["needs_fix"] for r in rows])

    cut = int(len(rows) * 0.8)
    head.fit_vectors(x[:cut], y[:cut], ESCALATE_SHARE)
    risk = head.risk_of_vectors(x[cut:])
    assert head.threshold is not None
    flagged = risk >= head.threshold
    y_te = y[cut:]
    print(f"holdout n={len(y_te)} AUC={roc_auc_score(y_te, risk):.3f} threshold={head.threshold:.3f}")
    print(
        f"  escalated {flagged.mean():.1%} of sub-stories; fix rate escalated={y_te[flagged].mean():.3f} "
        f"vs rest={y_te[~flagged].mean():.3f} (overall {y_te.mean():.3f}); "
        f"captures {y_te[flagged].sum() / max(1, y_te.sum()):.1%} of all fixes"
    )

    head.fit_vectors(x, y, ESCALATE_SHARE)
    path = DATA / "qc_risk.joblib"
    head.save(path)
    print(f"refit on all {len(rows)} → {path} (threshold {head.threshold:.3f})")


if __name__ == "__main__":
    main()
