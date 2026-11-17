"""HTTP service around the core library, for the UI. Not part of the general library contract
(ARCHITECTURE.md §1 still holds: nothing here calls a model) — this is a thin, local testing
wrapper so a browser can exercise resolve/train/evaluate without a Python shell. No auth: for
one trusted local user, not multi-tenant production (see service/app.py)."""
