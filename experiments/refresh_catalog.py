"""Refresh a catalog.yaml's cost/context-length/modality-support facts from OpenRouter's live
model list (image + video + audio — P3 / PLAN-v2.md §6).

Does NOT add new models or change any model's `tier` automatically — `tier` is a business
judgment (how much capability THIS deployment's use cases need from this model), which no public
API can answer. This only refreshes the facts a public source CAN answer, for models already in
the file; review the printed changes and re-run with --write to save them.

Usage:
    python experiments/refresh_catalog.py path/to/catalog.yaml [--write]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from tierrouter.capability import CapabilityError, fetch_openrouter_capabilities, match_by_suffix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("catalog_path", type=Path)
    parser.add_argument("--write", action="store_true", help="save the refreshed facts back to the file")
    args = parser.parse_args()

    document = yaml.safe_load(args.catalog_path.read_text(encoding="utf-8"))
    entries = document.get("catalog") or []
    if not entries:
        print(f"{args.catalog_path}: no 'catalog' entries found", file=sys.stderr)
        return 1

    try:
        capabilities = fetch_openrouter_capabilities()
    except CapabilityError as exc:
        print(f"Could not refresh from OpenRouter: {exc}", file=sys.stderr)
        return 1
    print(f"Fetched {len(capabilities)} models from OpenRouter.\n")

    changed = False
    for entry in entries:
        catalog_id = str(entry.get("id", ""))
        match = match_by_suffix(catalog_id, capabilities)
        if match is None:
            print(f"  not found on OpenRouter: {catalog_id} (left as-is — tier/label untouched either way)")
            continue
        updates: dict[str, object] = {}
        if match.supports_image != entry.get("supports_image", True):
            updates["supports_image"] = match.supports_image
        # P3 — PLAN-v2.md §6: sync video/audio the same way as image. Defaults match
        # CatalogEntry (video/audio False) so an absent YAML key is treated as False.
        if match.supports_video != bool(entry.get("supports_video", False)):
            updates["supports_video"] = match.supports_video
        if match.supports_audio != bool(entry.get("supports_audio", False)):
            updates["supports_audio"] = match.supports_audio
        if match.cost_per_1k_prompt is not None and round(match.cost_per_1k_prompt, 6) != entry.get("cost"):
            updates["cost"] = round(match.cost_per_1k_prompt, 6)
        if match.context_length is not None and match.context_length != entry.get("context_length"):
            updates["context_length"] = match.context_length
        if updates:
            print(f"  {catalog_id}: {updates}")
            entry.update(updates)
            changed = True
        else:
            print(f"  {catalog_id}: already up to date")

    print()
    if not changed:
        print("No changes.")
    elif args.write:
        args.catalog_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        print(f"Wrote {args.catalog_path}")
    else:
        print("(dry run — pass --write to save these changes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
