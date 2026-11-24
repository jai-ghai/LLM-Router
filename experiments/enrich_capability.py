"""Write a benchmark `capability` onto every catalog.yaml entry.

Source: OpenRouter's public model list, `benchmarks.artificial_analysis.intelligence_index`
(no API key). A `:batch` / `:free` variant takes its base model's score. Models without a
published index get an estimate from a least-squares fit of the measured ones on
log10(full price) and release date, clipped to the measured range, and are marked
`capability_source: estimated`.

Usage:
    python experiments/enrich_capability.py --catalog config/catalog.yaml [--dry-run]
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path

import httpx
import numpy as np
import yaml

from gradist.catalog import CatalogEntry, ModelCatalog, entry_to_yaml_dict, load_catalog

MODELS_URL = "https://openrouter.ai/api/v1/models"
SECONDS_PER_YEAR = 3.156e7


def _base(model_id: str) -> str:
    return model_id.split(":", 1)[0]


def _full_price(entry: CatalogEntry) -> float:
    return entry.cost * 2 if entry.id.endswith(":batch") else entry.cost


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default="config/catalog.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    models = httpx.get(MODELS_URL, timeout=30).json()["data"]
    index: dict[str, float] = {}
    created: dict[str, float] = {}
    for model in models:
        bare = model["id"].split("/", 1)[-1]
        created[bare] = float(model.get("created") or 0)
        aa = (model.get("benchmarks") or {}).get("artificial_analysis") or {}
        if aa.get("intelligence_index") is not None:
            index[bare] = float(aa["intelligence_index"])

    catalog = load_catalog(args.catalog)

    def measured(e: CatalogEntry) -> float | None:
        return index.get(e.id, index.get(_base(e.id)))

    def features(e: CatalogEntry) -> list[float]:
        released = created.get(e.id) or created.get(_base(e.id)) or 0.0
        return [math.log10(_full_price(e) + 1e-6), released / SECONDS_PER_YEAR, 1.0]

    known = [(e, v) for e in catalog.entries if (v := measured(e)) is not None]
    x = np.array([features(e) for e, _ in known])
    y = np.array([v for _, v in known])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ coef
    r2 = 1 - ((y - fitted) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    low, high = float(y.min()), float(y.max())

    entries: list[CatalogEntry] = []
    for e in catalog.entries:
        value = measured(e)
        if value is not None:
            entries.append(dataclasses.replace(e, capability=value, capability_source="measured"))
        else:
            estimate = float(np.clip(np.array(features(e)) @ coef, low, high))
            entries.append(dataclasses.replace(e, capability=estimate, capability_source="estimated"))

    print(f"measured {len(known)} / {len(entries)}; estimate fit R²={r2:.2f} on log price + release date")
    for e in sorted(entries, key=lambda e: -(e.capability or 0))[:10]:
        print(f"  {e.capability:5.1f} {e.capability_source:9} {e.id}")
    if args.dry_run:
        return
    document = {"catalog": [entry_to_yaml_dict(e) for e in ModelCatalog(entries=tuple(entries)).entries]}
    Path(args.catalog).write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"wrote {args.catalog}")


if __name__ == "__main__":
    main()
