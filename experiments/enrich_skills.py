"""Write per-skill scores onto every catalog.yaml entry, next to `capability`.

Source: OpenRouter's public model list (no API key).

- `multilingual`: `benchmarks.artificial_analysis.multilingual_index`, the Hindi / Indian-language
  skill. Models without a published index get an estimate from a least-squares fit of the
  measured ones on capability, log10(full price) and release date, clipped to the measured range, and are
  marked `multilingual_source: estimated`. Run enrich_capability.py first so every entry has
  a capability.
- `supports_structured_output`: "structured_outputs" in the model's supported parameters.

The other skills already live on the entry: reasoning and writing use `capability`, video and
hearing use `supports_video` / `supports_audio`, long input uses `context_length`, and quick
and cheap uses `cost`. Disagreements between those flags and OpenRouter's input modalities are
printed, not changed.

Usage:
    python experiments/enrich_skills.py --catalog config/catalog.yaml [--dry-run]
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path
from typing import Any

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

    models: dict[str, dict[str, Any]] = {
        m["id"].split("/", 1)[-1]: m for m in httpx.get(MODELS_URL, timeout=60).json()["data"]
    }
    catalog = load_catalog(args.catalog)

    def source(e: CatalogEntry) -> dict[str, Any] | None:
        return models.get(e.id) or models.get(_base(e.id))

    def measured(e: CatalogEntry) -> float | None:
        model = source(e)
        aa = ((model or {}).get("benchmarks") or {}).get("artificial_analysis") or {}
        value = aa.get("multilingual_index")
        return float(value) if value is not None else None

    def features(e: CatalogEntry) -> list[float]:
        released = float((source(e) or {}).get("created") or 0) / SECONDS_PER_YEAR
        return [e.capability or 0.0, math.log10(_full_price(e) + 1e-6), released, 1.0]

    known = [(e, v) for e in catalog.entries if (v := measured(e)) is not None and e.capability is not None]
    x = np.array([features(e) for e, _ in known])
    y = np.array([v for _, v in known])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ coef
    r2 = 1 - ((y - fitted) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    low, high = float(y.min()), float(y.max())

    entries: list[CatalogEntry] = []
    modality_mismatches: list[str] = []
    for e in catalog.entries:
        model = source(e)
        structured = "structured_outputs" in (model.get("supported_parameters") or []) if model else None
        value = measured(e)
        if value is not None:
            score, how = value, "measured"
        elif e.capability is not None:
            score, how = float(np.clip(np.array(features(e)) @ coef, low, high)), "estimated"
        else:
            score, how = None, ""
        entries.append(
            dataclasses.replace(e, multilingual=score, multilingual_source=how, supports_structured_output=structured)
        )
        if model:
            inputs = set((model.get("architecture") or {}).get("input_modalities") or [])
            for flag, modality in (("supports_video", "video"), ("supports_audio", "audio")):
                if getattr(e, flag) != (modality in inputs):
                    modality_mismatches.append(f"{e.id}: catalog {flag}={getattr(e, flag)}, OpenRouter inputs={sorted(inputs)}")

    n_measured = sum(1 for e in entries if e.multilingual_source == "measured")
    n_structured = sum(1 for e in entries if e.supports_structured_output)
    n_unknown = sum(1 for e in entries if e.supports_structured_output is None)
    print(f"multilingual: measured {n_measured} / {len(entries)}; estimate fit R²={r2:.2f} on capability + log price + release date")
    print(f"structured output: {n_structured} yes, {len(entries) - n_structured - n_unknown} no, {n_unknown} unknown")
    for e in sorted(entries, key=lambda e: -(e.multilingual or 0))[:10]:
        print(f"  {e.multilingual:.3f} {e.multilingual_source:9} {e.id}")
    if modality_mismatches:
        print(f"{len(modality_mismatches)} video/audio flags disagree with OpenRouter (not changed):")
        for line in modality_mismatches[:15]:
            print(f"  {line}")
    if args.dry_run:
        return
    document = {"catalog": [entry_to_yaml_dict(e) for e in ModelCatalog(entries=tuple(entries)).entries]}
    Path(args.catalog).write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"wrote {args.catalog}")


if __name__ == "__main__":
    main()
