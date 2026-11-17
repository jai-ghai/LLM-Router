"""Build the FastAPI app from environment variables.

Run: uvicorn tierrouter.service.main:create --factory --port 8000

Variables:
  TIERROUTER_TIERS         path to tiers.yaml (required)
  TIERROUTER_DATA          folder for the store (examples, judgments, rules, images); default ./data
  TIERROUTER_RULES_SEED    a rules.yaml to copy in ONLY if the store's own rules.yaml is still empty
                           (never overwrites rules you have already edited through the UI)
  TIERROUTER_CORS_ORIGINS  comma-separated browser origins allowed to call the API (the UI)
  TIERROUTER_CATALOG       optional path to catalog.yaml (a selectable multi-model catalog). If
                           unset, resolve() uses tiers.yaml's single model per tier, as before.
  TIERROUTER_QUALITY_FIELD optional metadata field name (e.g. "task") to bucket human quality
                           ratings by. Required for /v1/resolve's `min_quality` to do anything —
                           without it, ratings can still be recorded via /v1/quality but never
                           affect a decision.
"""

from __future__ import annotations

import os

from fastapi import FastAPI

from tierrouter.catalog import load_catalog
from tierrouter.rules import rules_from_file
from tierrouter.service.app import create_app
from tierrouter.service.store import Store
from tierrouter.tiers import load_tiers


def create() -> FastAPI:
    tiers_path = os.environ.get("TIERROUTER_TIERS")
    if not tiers_path:
        raise RuntimeError("TIERROUTER_TIERS must be set to a tiers.yaml path")
    tiers = load_tiers(tiers_path)

    store = Store(os.environ.get("TIERROUTER_DATA") or "data")
    seed_path = os.environ.get("TIERROUTER_RULES_SEED")
    if seed_path and store.load_rules().is_empty():
        store.save_rules(rules_from_file(seed_path))

    catalog_path = os.environ.get("TIERROUTER_CATALOG")
    catalog = load_catalog(catalog_path) if catalog_path else None
    quality_field = os.environ.get("TIERROUTER_QUALITY_FIELD") or None

    raw_origins = os.environ.get("TIERROUTER_CORS_ORIGINS", "")
    origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    return create_app(tiers, store, cors_origins=origins or None, catalog=catalog, quality_field=quality_field)
