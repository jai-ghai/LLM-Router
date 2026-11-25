"""Vercel entrypoint: the API built by `gradist.service.main.create`.

GRADIST_SESSION_SECRET must be set in the Vercel project. The other settings default to the
committed config. On Vercel the filesystem is read-only except /tmp, and data/ is not deployed,
so the store starts empty there and does not survive a cold start.
"""

import os

os.environ.setdefault("GRADIST_TIERS", "config/tiers.general.yaml")
os.environ.setdefault("GRADIST_CATALOG", "config/catalog.yaml")
os.environ.setdefault("GRADIST_QUALITY_FIELD", "task")
if os.environ.get("VERCEL"):
    os.environ.setdefault("GRADIST_DATA", "/tmp/gradist")

from gradist.service.main import create

app = create()
