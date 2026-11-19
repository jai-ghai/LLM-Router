"""Outcome-labelled tier training from ulab-ai/xRouteBench, plus a replay evaluation.

Every xRouteBench query was answered by 18 open models and scored (`performance` in [0, 1]).
Models are bucketed into our tiers by parameter count; a query's label is the cheapest tier
whose mean score is within MARGIN of the best tier's — i.e. "the smallest tier that actually
did the job", not a guess. Queries no tier solves carry no routing signal and are dropped.

Replay evaluation routes held-out test queries and looks up the recorded score/cost of the
chosen tier, so "accuracy" means quality retained and money saved, not label agreement.

Run `experiments/probe_xroutebench.py` first to download data/xroutebench/.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.train_from_hf_content import (
    TEXT_MODEL,
    _infer_fields,
    _load_jsonl,
    _write_jsonl,
    train,
)
from gradist.classifier import TierClassifier
from gradist.features import Dinov2Embedder, MetadataEncoder, SentenceTransformerEmbedder

DATA = ROOT / "data"
XRB = DATA / "xroutebench"
TIERS = ("small", "medium", "large")
MARGIN = 0.1
MIN_BEST = 0.2
JUDGED_BY = "xroutebench-outcome"
METADATA = {"modality": "text"}


def size_tier(size: str) -> str:
    billions = float(str(size).rstrip("Bb"))
    if billions <= 10:
        return "small"
    if billions <= 80:
        return "medium"
    return "large"


def load_pool() -> dict[str, dict]:
    rows = json.loads((XRB / "llm_candidates.json").read_text(encoding="utf-8"))
    return {
        r["model_name"]: {
            "tier": size_tier(r["size"]),
            "in": float(r["input_price_per_1m"]),
            "out": float(r["output_price_per_1m"]),
        }
        for r in rows
    }


def user_query(raw: str) -> str:
    """Drop the benchmark's '[System Instruction]' boilerplate; route on the actual question."""
    text = raw.split("[User Query]", 1)[-1]
    return " ".join(text.split())[:2000]


def per_query_outcomes(split: str, pool: dict[str, dict]) -> list[dict]:
    """One record per query: text, mean score and mean $ cost for each tier."""
    grouped: dict[tuple[str, str], dict] = {}
    for line in (XRB / f"generic_{split}.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        model = pool.get(r["model_name"])
        if model is None:
            continue
        key = (r["task_name"], str(r["task_id"]) + "|" + str(r["query"])[:200])
        g = grouped.setdefault(
            key,
            {"task": r["task_name"], "text": user_query(str(r["query"])), "perf": defaultdict(list), "cost": defaultdict(list)},
        )
        cost = (float(r["input_tokens"] or 0) * model["in"] + float(r["output_tokens"] or 0) * model["out"]) / 1e6
        g["perf"][model["tier"]].append(float(r["performance"] or 0))
        g["cost"][model["tier"]].append(cost)
    out = []
    for g in grouped.values():
        if not all(g["perf"][t] for t in TIERS):
            continue
        perf = {t: sum(g["perf"][t]) / len(g["perf"][t]) for t in TIERS}
        cost = {t: sum(g["cost"][t]) / len(g["cost"][t]) for t in TIERS}
        best = max(perf.values())
        label = None if best < MIN_BEST else next(t for t in TIERS if perf[t] >= best - MARGIN)
        out.append({"task": g["task"], "text": g["text"], "perf": perf, "cost": cost, "label": label})
    return out


def balanced_sample(records: list[dict], per_tier: int, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    by_tier: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r["label"]:
            by_tier[r["label"]].append(r)
    picked: list[dict] = []
    for tier in TIERS:
        bucket = by_tier[tier]
        rng.shuffle(bucket)
        picked.extend(bucket[:per_tier])
    return picked


def load_classifier(examples: list[dict]) -> TierClassifier:
    uses_images = any(e.get("image_path") for e in examples)
    return TierClassifier.load(
        DATA / "classifier.joblib",
        MetadataEncoder(_infer_fields(examples)),
        SentenceTransformerEmbedder(TEXT_MODEL),
        Dinov2Embedder() if uses_images else None,
    )


def replay(name: str, choose, records: list[dict]) -> dict:
    perf = cost = 0.0
    dist: Counter[str] = Counter()
    for r in records:
        tier = choose(r)
        dist[tier] += 1
        perf += r["perf"][tier]
        cost += r["cost"][tier]
    n = len(records)
    return {"name": name, "perf": perf / n, "cost_per_1k": 1000 * cost / n, "dist": dict(dist)}


def print_replay(rows: list[dict]) -> None:
    large = next(r for r in rows if r["name"] == "always large")
    print(f"\n{'policy':22s} {'avg score':>9s} {'% of large':>10s} {'$ / 1k req':>10s} {'cost vs large':>13s}  tier mix")
    for r in rows:
        print(
            f"{r['name']:22s} {r['perf']:9.3f} {100 * r['perf'] / large['perf']:9.1f}% "
            f"{r['cost_per_1k']:10.4f} {100 * r['cost_per_1k'] / large['cost_per_1k']:12.1f}%  {r['dist']}"
        )


def evaluate(examples: list[dict], test: list[dict], label: str) -> dict:
    clf = load_classifier(examples)
    return replay(label, lambda r: clf.predict(dict(METADATA), r["text"])[0], test)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-tier", type=int, default=600)
    parser.add_argument("--test-n", type=int, default=900)
    args = parser.parse_args()

    pool = load_pool()
    print("tier buckets:", {t: sorted(m for m, v in pool.items() if v["tier"] == t) for t in TIERS})
    train_q = per_query_outcomes("train", pool)
    test_q = per_query_outcomes("test", pool)
    print(f"train queries={len(train_q)} labels={Counter(r['label'] for r in train_q)}")
    print(f"test  queries={len(test_q)}  labels={Counter(r['label'] for r in test_q)}")

    rng = random.Random(11)
    test_eval = [r for r in test_q if r["label"]]
    rng.shuffle(test_eval)
    test_eval = test_eval[: args.test_n]

    examples = _load_jsonl(DATA / "examples.jsonl")
    judgments = _load_jsonl(DATA / "judgments.jsonl")
    stale = {j["example_id"] for j in judgments if j.get("judged_by") == JUDGED_BY}
    examples = [e for e in examples if e["example_id"] not in stale]
    judgments = [j for j in judgments if j["example_id"] not in stale]

    baselines = [replay(f"always {t}", lambda r, t=t: t, test_eval) for t in TIERS]
    baselines.append(replay("oracle (best possible)", lambda r: r["label"], test_eval))
    print(f"\nreplay set: {len(test_eval)} held-out test queries")
    before = evaluate(examples, test_eval, "classifier BEFORE")

    backup = DATA / "classifier.pre_xroute.joblib"
    if not backup.exists():
        shutil.copy2(DATA / "classifier.joblib", backup)

    new_ex, new_ju = [], []
    for r in balanced_sample(train_q, args.per_tier):
        example_id = f"ex-{uuid.uuid4().hex[:12]}"
        new_ex.append({"example_id": example_id, "metadata": dict(METADATA), "text": r["text"], "image_path": None})
        scores = " ".join(f"{t}={r['perf'][t]:.2f}" for t in TIERS)
        new_ju.append({"example_id": example_id, "tier": r["label"], "judged_by": JUDGED_BY, "notes": f"task={r['task']} {scores}"})
    print(f"\nadding {len(new_ex)} outcome-labelled examples: {Counter(j['tier'] for j in new_ju)}")
    examples += new_ex
    judgments += new_ju
    _write_jsonl(DATA / "examples.jsonl", examples)
    _write_jsonl(DATA / "judgments.jsonl", judgments)
    train(examples, judgments)

    after = evaluate(examples, test_eval, "classifier AFTER")
    print_replay([*baselines, before, after])
    per_task: dict[str, list[dict]] = defaultdict(list)
    for r in test_eval:
        per_task[re.sub(r"_\d.*", "", r["task"])].append(r)
    print("\noracle tier mix per task (what the outcomes say each task needs):")
    for task, rows in sorted(per_task.items()):
        print(f"  {task:16s} n={len(rows):4d} {dict(Counter(r['label'] for r in rows))}")


if __name__ == "__main__":
    main()
