"""Pick how many xRouteBench rows to mix into training without eroding domain routing.

Embeds everything once (domain examples, xRouteBench train pool, replay test set), then sweeps
per-tier sample size and LogisticRegression C. For each setting it reports:
  * domain held-out accuracy (20% of our own tabson / clipper / newsgroup labels), and
  * replay quality + cost on held-out xRouteBench test queries.
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.train_from_hf_content import TEXT_MODEL, _infer_fields, _load_jsonl
from experiments.train_from_xroutebench import (
    DATA,
    JUDGED_BY,
    METADATA,
    TIERS,
    balanced_sample,
    load_pool,
    per_query_outcomes,
)
from gradist.features import (
    Dinov2Embedder,
    MetadataEncoder,
    SentenceTransformerEmbedder,
    load_image_bytes,
)


def featurize(rows: list[dict], encoder: MetadataEncoder, text_emb, img_emb) -> np.ndarray:
    meta = np.stack([encoder.encode(r["metadata"]) for r in rows])
    text = np.asarray(text_emb.embed([r["text"] for r in rows]), dtype=np.float64)
    img = np.zeros((len(rows), img_emb.width + 1))
    for i, r in enumerate(rows):
        if r.get("image_path"):
            img[i, :-1] = img_emb.embed([load_image_bytes(r["image_path"])])[0]
            img[i, -1] = 1.0
    return np.hstack([meta, text, img])


def main() -> None:
    examples = _load_jsonl(DATA / "examples.jsonl")
    judgments = _load_jsonl(DATA / "judgments.jsonl")
    tier_of = {j["example_id"]: j["tier"] for j in judgments if j.get("judged_by") != JUDGED_BY}
    domain = [dict(e, tier=tier_of[e["example_id"]]) for e in examples if e["example_id"] in tier_of]
    rng = random.Random(3)
    rng.shuffle(domain)
    cut = int(len(domain) * 0.8)
    dom_train, dom_test = domain[:cut], domain[cut:]

    pool = load_pool()
    xr_pool = balanced_sample(per_query_outcomes("train", pool), 1200)
    xr_rows = [{"metadata": dict(METADATA), "text": r["text"], "tier": r["label"]} for r in xr_pool]
    test_q = [r for r in per_query_outcomes("test", pool) if r["label"]]
    random.Random(11).shuffle(test_q)
    test_q = test_q[:900]

    cache = DATA / "xroutebench" / "features_cache.npz"
    if cache.exists():
        z = np.load(cache)
        x_dom_tr, x_dom_te, x_xr, x_test = z["dom_tr"], z["dom_te"], z["xr"], z["test"]
    else:
        encoder = MetadataEncoder(_infer_fields(examples))
        text_emb = SentenceTransformerEmbedder(TEXT_MODEL)
        img_emb = Dinov2Embedder()
        print("embedding…")
        x_dom_tr, x_dom_te = featurize(dom_train, encoder, text_emb, img_emb), featurize(dom_test, encoder, text_emb, img_emb)
        x_xr = featurize(xr_rows, encoder, text_emb, img_emb)
        x_test = featurize([{"metadata": dict(METADATA), "text": r["text"]} for r in test_q], encoder, text_emb, img_emb)
        np.savez_compressed(cache, dom_tr=x_dom_tr, dom_te=x_dom_te, xr=x_xr, test=x_test)
    y_dom_tr = [r["tier"] for r in dom_train]
    y_dom_te = [r["tier"] for r in dom_test]
    y_xr = np.array([r["tier"] for r in xr_rows])

    def replay(pred: list[str]) -> tuple[float, float, dict]:
        perf = sum(r["perf"][t] for r, t in zip(test_q, pred, strict=True)) / len(test_q)
        cost = sum(r["cost"][t] for r, t in zip(test_q, pred, strict=True)) / len(test_q)
        return perf, cost, dict(Counter(pred))

    large_perf, large_cost, _ = replay(["large"] * len(test_q))
    print(f"domain train={len(dom_train)} test={len(dom_test)}  xr pool={Counter(y_xr.tolist())}  replay n={len(test_q)}")
    print(f"\n{'per_tier':>8s} {'C':>5s} {'domain acc':>10s} {'replay % large':>14s} {'cost % large':>12s}  mix")
    for per_tier in (0, 100, 200, 400, 800, 1200):
        idx = np.concatenate([np.where(y_xr == t)[0][:per_tier] for t in TIERS]) if per_tier else np.array([], dtype=int)
        x = np.vstack([x_dom_tr, x_xr[idx]]) if per_tier else x_dom_tr
        y = y_dom_tr + y_xr[idx].tolist()
        for c in (0.3, 1.0, 3.0):
            clf = LogisticRegression(max_iter=2000, C=c).fit(x, y)
            dom_acc = float(np.mean(clf.predict(x_dom_te) == np.array(y_dom_te)))
            perf, cost, mix = replay(clf.predict(x_test).tolist())
            print(f"{per_tier:8d} {c:5.1f} {dom_acc:10.3f} {100 * perf / large_perf:13.1f}% {100 * cost / large_cost:11.1f}%  {mix}")

    # Generic-text heads trained on xRouteBench only (LLMRouter-style alternatives).
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.neural_network import MLPClassifier

    def show(name: str, pred: list[str]) -> None:
        perf, cost, mix = replay(pred)
        print(f"{name:34s} {100 * perf / large_perf:13.1f}% {100 * cost / large_cost:11.1f}%  {mix}")

    print(f"\n{'generic head (xRouteBench only)':34s} {'replay % large':>14s} {'cost % large':>12s}  mix")
    for t in TIERS:
        show(f"always {t}", [t] * len(test_q))
    show("oracle", [r["label"] for r in test_q])
    for k in (10, 25, 50):
        show(f"KNN classifier k={k}", KNeighborsClassifier(k, metric="cosine").fit(x_xr, y_xr).predict(x_test).tolist())
    show("MLP classifier", MLPClassifier((256,), max_iter=400, random_state=0).fit(x_xr, y_xr).predict(x_test).tolist())

    # Predict each tier's score, then take the cheapest tier within `margin` of the best prediction.
    y_perf = np.array([[r["perf"][t] for t in TIERS] for r in xr_pool])
    regressors = {
        "ridge": Ridge(alpha=3.0).fit(x_xr, y_perf),
        "knn-reg k=25": KNeighborsRegressor(25, metric="cosine").fit(x_xr, y_perf),
    }
    for name, reg in regressors.items():
        pred_perf = reg.predict(x_test)
        for margin in (0.0, 0.03, 0.06, 0.1, 0.15):
            pred = [TIERS[next(i for i in range(3) if p[i] >= p.max() - margin)] for p in pred_perf]
            show(f"{name} score→cheapest, margin={margin:.2f}", pred)


if __name__ == "__main__":
    main()
