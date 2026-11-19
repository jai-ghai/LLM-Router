# [Fact-Forcing preamble]
#
# Callers: CLI (`python experiments/sync_catalog.py …`); experiments/refresh_catalog.py
# shim delegates here with --only-existing; tests/test_sync_catalog.py imports the pure
# helpers (refresh_existing_entries / add_new_entries) with injected capabilities.
#
# No existing file serves the same purpose: refresh_catalog.py only updated facts; this
# script is the merged import + refresh tool (Q6.5.a / PLAN-v2.md §7).
#
# Data schemas: reads/writes catalog.yaml (`catalog: [{id, provider, tier, cost, …}]`);
# reads OpenRouter /api/v1/models via capability.py (metadata only — no inference).
#
# No secrets in plaintext: no API key; OpenRouter models list is public.
#
# User's verbatim instruction: "yes i want it" — P1 Dynamic OpenRouter import + UNREVIEWED
# (PLAN-v2.md §7).

"""Sync a catalog.yaml against OpenRouter's live model list.

Two modes (exactly one required — Q6.5.c):

  --only-existing   Refresh cost / context_length / modality flags for rows already in
                    the file. Never adds rows; never changes tier. Same behaviour as the
                    old experiments/refresh_catalog.py.

  --add-new         Also append models that match --providers / --modalities and are not
                    already in the file. New rows get tier=UNREVIEWED so pick() will not
                    route to them until an admin assigns a real tier (G5).

Usage:
    python experiments/sync_catalog.py \\
        --out config/catalog.yaml \\
        --providers openai,anthropic,google \\
        --only-existing | --add-new \\
        [--modalities image,video] \\
        [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from gradist.capability import (
    CapabilityError,
    ModelCapability,
    fetch_openrouter_capabilities,
    match_by_suffix,
)
from gradist.tiers import UNREVIEWED_TIER

COMMON_PROVIDERS = (
    "openai",
    "anthropic",
    "google",
    "meta-llama",
    "qwen",
    "deepseek",
    "mistralai",
)


def bare_id(openrouter_id: str) -> str:
    """Strip the provider/ prefix — catalog.yaml uses bare ids (match_by_suffix convention)."""
    return openrouter_id.split("/", 1)[-1]


def _modality_ok(cap: ModelCapability, wanted: set[str]) -> bool:
    """True when the capability supports at least one of the requested modalities.
    An empty wanted set means "no modality filter"."""
    if not wanted:
        return True
    mapping = {
        "image": cap.supports_image,
        "video": cap.supports_video,
        "audio": cap.supports_audio,
        "text": True,  # every model is assumed text-capable
    }
    return any(mapping.get(m, False) for m in wanted)


def refresh_existing_entries(
    entries: list[dict[str, Any]],
    capabilities: list[ModelCapability],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Update facts on existing rows. Never changes `tier` or `assigned_by`.
    Returns (entries, human-readable change lines)."""
    lines: list[str] = []
    for entry in entries:
        catalog_id = str(entry.get("id", ""))
        match = match_by_suffix(catalog_id, capabilities)
        if match is None:
            lines.append(f"  not found on OpenRouter: {catalog_id} (left as-is)")
            continue
        updates: dict[str, object] = {}
        if match.supports_image != entry.get("supports_image", True):
            updates["supports_image"] = match.supports_image
        if match.supports_video != bool(entry.get("supports_video", False)):
            updates["supports_video"] = match.supports_video
        if match.supports_audio != bool(entry.get("supports_audio", False)):
            updates["supports_audio"] = match.supports_audio
        if match.cost_per_1k_prompt is not None and round(match.cost_per_1k_prompt, 6) != entry.get(
            "cost"
        ):
            updates["cost"] = round(match.cost_per_1k_prompt, 6)
        if match.context_length is not None and match.context_length != entry.get("context_length"):
            updates["context_length"] = match.context_length
        if updates:
            lines.append(f"  {catalog_id}: {updates}")
            entry.update(updates)
        else:
            lines.append(f"  {catalog_id}: already up to date")
    return entries, lines


def add_new_entries(
    entries: list[dict[str, Any]],
    capabilities: list[ModelCapability],
    *,
    providers: set[str],
    modalities: set[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Append UNREVIEWED rows for OpenRouter models that match the filters and are not
    already present (matched by bare id, case-insensitive)."""
    existing = {str(e.get("id", "")).strip().lower() for e in entries}
    lines: list[str] = []
    added = 0
    for cap in capabilities:
        if cap.provider.lower() not in providers:
            continue
        if not _modality_ok(cap, modalities):
            continue
        bid = bare_id(cap.id)
        if bid.lower() in existing:
            continue
        row: dict[str, Any] = {
            "id": bid,
            "provider": cap.provider,
            "tier": UNREVIEWED_TIER,
            "cost": round(cap.cost_per_1k_prompt, 6) if cap.cost_per_1k_prompt is not None else 1.0,
            "supports_image": cap.supports_image,
            "supports_video": cap.supports_video,
            "supports_audio": cap.supports_audio,
            "label": cap.name or bid,
        }
        if cap.context_length is not None:
            row["context_length"] = cap.context_length
        entries.append(row)
        existing.add(bid.lower())
        lines.append(f"  + {bid} (provider={cap.provider}, tier={UNREVIEWED_TIER})")
        added += 1
    if added == 0:
        lines.append("  (no new models matched the filters)")
    return entries, lines


def _parse_csv(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {part.strip().lower() for part in raw.split(",") if part.strip()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, required=True, help="path to catalog.yaml to read/write")
    parser.add_argument(
        "--providers",
        default=None,
        help="comma-separated OpenRouter provider namespaces (required)",
    )
    parser.add_argument(
        "--modalities",
        default=None,
        help="comma-separated modality filter for --add-new (image,video,audio,text)",
    )
    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument(
        "--only-existing",
        action="store_true",
        help="refresh facts for existing rows only (never add)",
    )
    mode.add_argument(
        "--add-new",
        action="store_true",
        help="also append matching models as tier=UNREVIEWED",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print changes without writing the file",
    )
    args = parser.parse_args(argv)

    if not args.only_existing and not args.add_new:
        parser.error(
            "exactly one of --only-existing or --add-new is required "
            "(refusing to guess — see PLAN-v2.md §7 / Q6.5.c)"
        )
    if not args.providers or not args.providers.strip():
        parser.error(
            "--providers is required (comma-separated). Common values: "
            + ",".join(COMMON_PROVIDERS)
        )

    providers = _parse_csv(args.providers)
    modalities = _parse_csv(args.modalities)

    out: Path = args.out
    if out.exists():
        document = yaml.safe_load(out.read_text(encoding="utf-8")) or {}
        if not isinstance(document, dict):
            print(f"{out}: expected a YAML mapping", file=sys.stderr)
            return 1
        entries = list(document.get("catalog") or [])
    else:
        if args.only_existing:
            print(f"{out}: file does not exist (nothing to refresh)", file=sys.stderr)
            return 1
        document = {}
        entries = []

    try:
        capabilities = fetch_openrouter_capabilities()
    except CapabilityError as exc:
        print(f"Could not fetch OpenRouter models: {exc}", file=sys.stderr)
        return 1
    print(f"Fetched {len(capabilities)} models from OpenRouter.\n")

    entries, refresh_lines = refresh_existing_entries(entries, capabilities)
    for line in refresh_lines:
        print(line)

    add_lines: list[str] = []
    if args.add_new:
        print("\nAdding new models matching filters…")
        entries, add_lines = add_new_entries(
            entries, capabilities, providers=providers, modalities=modalities
        )
        for line in add_lines:
            print(line)

    document["catalog"] = entries
    print()
    if args.dry_run:
        print(f"(dry run — would write {len(entries)} entries to {out})")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    print(f"Wrote {len(entries)} entries to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
