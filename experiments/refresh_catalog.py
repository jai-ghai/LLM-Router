"""Deprecated shim — delegates to `experiments/sync_catalog.py --only-existing`.

Kept so anyone with the old habit (`python experiments/refresh_catalog.py path [--write]`)
still gets the same refresh-facts behaviour. New callers should use sync_catalog.py
directly (PLAN-v2.md §7 / Q6.5.a).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def _load_sync_catalog():
    path = Path(__file__).with_name("sync_catalog.py")
    spec = importlib.util.spec_from_file_location("gradist_sync_catalog", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    # Ensure src/ is on path so sync_catalog can import gradist.*
    root = Path(__file__).resolve().parents[1]
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog_path", type=Path)
    parser.add_argument("--write", action="store_true", help="save changes (default is dry-run)")
    args = parser.parse_args()
    sync = _load_sync_catalog()
    argv = [
        "--out",
        str(args.catalog_path),
        "--providers",
        ",".join(sync.COMMON_PROVIDERS),
        "--only-existing",
    ]
    if not args.write:
        argv.append("--dry-run")
    return sync.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
