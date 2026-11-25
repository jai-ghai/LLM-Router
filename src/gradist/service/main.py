# [Fact-Forcing preamble]
#
# The file(s) and line(s) that will call it: `uvicorn gradist.service.main:create
# --factory` invokes `create()` on service start. Tests in `tests/test_service.py` do NOT
# use this file; they call `create_app` directly with injected fixtures.
#
# No existing file serves the same purpose: this IS the env-driven factory.
#
# The data schemas it reads or writes: reads env vars listed in the module docstring;
# constructs `Store`, `UsersStore` (at `store.root / "users.jsonl"`), and `SessionConfig`,
# passing all three to `create_app`. No direct JSONL I/O from this file.
#
# No secrets stored in plaintext: `GRADIST_SESSION_SECRET` is held in process memory
# only, passed to `SessionMiddleware`, never persisted. Enforced-required: the service
# refuses to start if the env var is missing or empty (no insecure fallback).
#
# The user's verbatim instruction driving this file: "create auth or required things because
# maybe user have different workflow and they can create difference judgments".

"""Build the FastAPI app from environment variables.

Run: uvicorn gradist.service.main:create --factory --port 8000

Variables:
  GRADIST_TIERS                    path to tiers.yaml (required)
  GRADIST_DATA                     folder for the store (examples, judgments, rules,
                                      history, quality, users, images); default ./data
  GRADIST_RULES_SEED               a rules.yaml to copy in ONLY if the store's own
                                      rules.yaml is still empty (never overwrites rules you
                                      have already edited through the UI)
  GRADIST_CORS_ORIGINS             comma-separated browser origins allowed to call the
                                      API (the UI). Default when unset: http://localhost:5174
                                      (matches ui/vite.config.ts) so local credentialed
                                      session cookies work without extra setup.
  GRADIST_CATALOG                  optional path to catalog.yaml (a selectable
                                      multi-model catalog). If unset, resolve() uses
                                      tiers.yaml's single model per tier.
  GRADIST_QUALITY_FIELD            optional metadata field name (e.g. "task") to bucket
                                      human quality ratings by. Required for /v1/resolve's
                                      `min_quality` to do anything.
  GRADIST_SESSION_SECRET           REQUIRED. Random secret used by SessionMiddleware to
                                      sign the session cookie. The service refuses to start
                                      without it. Generate with:
                                        python -c "import secrets; print(secrets.token_urlsafe(48))"
                                      Rotate by restarting with a new value (all existing
                                      sessions invalidate, which is the intended behaviour).
  GRADIST_ALLOW_READ_WITHOUT_AUTH  when set to "1", GET endpoints are reachable without a
                                      session (for monitoring/ops). Mutating endpoints are
                                      ALWAYS auth-gated regardless of this flag.
"""

from __future__ import annotations

import os

from fastapi import FastAPI

from gradist.catalog import load_catalog
from gradist.rules import rules_from_file
from gradist.service.app import create_app
from gradist.service.auth import SessionConfig
from gradist.service.store import Store
from gradist.service.users_store import UsersStore
from gradist.tiers import load_tiers


def create() -> FastAPI:
    tiers_path = os.environ.get("GRADIST_TIERS")
    if not tiers_path:
        raise RuntimeError("GRADIST_TIERS must be set to a tiers.yaml path")
    tiers = load_tiers(tiers_path)

    session_secret = os.environ.get("GRADIST_SESSION_SECRET") or ""
    if not session_secret.strip():
        raise RuntimeError(
            "GRADIST_SESSION_SECRET must be set to a non-empty value before the service "
            "can start. Generate one with: "
            'python -c "import secrets; print(secrets.token_urlsafe(48))"'
        )
    session_config = SessionConfig(secret=session_secret.strip())

    data_dir = os.environ.get("GRADIST_DATA") or "data"
    catalog_path = os.environ.get("GRADIST_CATALOG")
    # Pass catalog_path into Store so PUT /v1/catalog/{id} can persist tier assignments (P1).
    store = Store(data_dir, catalog_path=catalog_path)
    users = UsersStore(path=store.root / "users.jsonl")
    seed_path = os.environ.get("GRADIST_RULES_SEED")
    if seed_path and store.load_rules().is_empty():
        store.save_rules(rules_from_file(seed_path))

    catalog = load_catalog(catalog_path) if catalog_path else None
    quality_field = os.environ.get("GRADIST_QUALITY_FIELD") or None

    raw_origins = os.environ.get("GRADIST_CORS_ORIGINS", "")
    origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    # Local UI (vite.config.ts port 5174) needs an explicit origin + allow_credentials=True;
    # an empty list skips CORSMiddleware entirely and browsers reject credentials: 'include'.
    if not origins:
        origins = ["http://localhost:5174"]
    allow_read = os.environ.get("GRADIST_ALLOW_READ_WITHOUT_AUTH") == "1"
    return create_app(
        tiers,
        store,
        users,
        session_config,
        cors_origins=origins,
        catalog=catalog,
        quality_field=quality_field,
        allow_read_without_auth=allow_read,
    )
