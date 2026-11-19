"""Download ulab-ai/xRouteBench (generic + candidate pool) into data/xroutebench/ and print its shape."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset

OUT = Path("data/xroutebench")
REPO = "ulab-ai/xRouteBench"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pool = load_dataset(REPO, "llm_candidates")["train"]
    pool_rows = [dict(r) for r in pool]
    (OUT / "llm_candidates.json").write_text(json.dumps(pool_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"candidates: {len(pool_rows)}")
    for r in pool_rows:
        print(f"  {r['model_name']:40s} size={r['size']!s:8s} in=${r['input_price_per_1m']} out=${r['output_price_per_1m']} service={r['service']}")

    ds = load_dataset(REPO, "llmrouter_generic")
    for split in ("train", "test"):
        rows = [dict(r) for r in ds[split]]
        with (OUT / f"generic_{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        print(f"\n{split}: {len(rows)} rows; columns={sorted(rows[0].keys())}")
        print("  tasks:", Counter(r.get("task_name") for r in rows).most_common(15))
        print("  models:", len({r.get("model_name") for r in rows}))
        print("  distinct queries:", len({str(r.get("query"))[:300] for r in rows}))
    sample = {k: (str(v)[:160]) for k, v in rows[0].items()}
    print("\nsample row:", json.dumps(sample, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
