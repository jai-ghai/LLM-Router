# Behaviour under test: experiments/sync_catalog.py helpers — refresh_existing_entries
# (facts only, never changes tier) and add_new_entries (emits UNREVIEWED rows).
# Existing test files: none for this module (new in P1). refresh_catalog behaviour is
# covered here via refresh_existing_entries.
# User's verbatim instruction: "yes i want it".
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from gradist.capability import ModelCapability
from gradist.tiers import UNREVIEWED_TIER

_SYNC_PATH = Path(__file__).resolve().parents[1] / "experiments" / "sync_catalog.py"


def _load_sync():
    # Ensure src/ is importable the same way the script expects.
    src = str(Path(__file__).resolve().parents[1] / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    spec = importlib.util.spec_from_file_location("gradist_sync_catalog_test", _SYNC_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sync = _load_sync()


def _cap(
    oid: str,
    *,
    cost: float = 0.001,
    image: bool = True,
    video: bool = False,
    audio: bool = False,
    name: str = "",
    context: int = 8000,
) -> ModelCapability:
    provider = oid.split("/", 1)[0]
    return ModelCapability(
        id=oid,
        provider=provider,
        context_length=context,
        supports_image=image,
        cost_per_1k_prompt=cost,
        cost_per_1k_completion=cost * 2,
        supports_video=video,
        supports_audio=audio,
        name=name,
    )


def test_add_new_emits_unreviewed_entries_for_unknown_models() -> None:
    entries: list[dict] = [
        {"id": "already-here", "provider": "openai", "tier": "small", "cost": 1},
    ]
    caps = [
        _cap("openai/already-here", cost=0.002, name="Already"),
        _cap("openai/brand-new", cost=0.003, name="Brand New"),
        _cap("anthropic/skip-me", cost=0.004),  # wrong provider filter
    ]
    updated, lines = sync.add_new_entries(
        entries, caps, providers={"openai"}, modalities=set()
    )
    ids = [e["id"] for e in updated]
    assert "already-here" in ids
    assert "brand-new" in ids
    assert "skip-me" not in ids
    new = next(e for e in updated if e["id"] == "brand-new")
    assert new["tier"] == UNREVIEWED_TIER
    assert new["provider"] == "openai"
    assert new["label"] == "Brand New"
    assert any("brand-new" in line for line in lines)


def test_only_existing_leaves_tier_untouched() -> None:
    # Regression vs. today's refresh_catalog: facts update, tier stays put.
    entries = [
        {
            "id": "gemini-flash",
            "provider": "google",
            "tier": "medium",
            "cost": 1.0,
            "supports_image": False,
            "supports_video": False,
            "assigned_by": "jai",
        }
    ]
    caps = [
        _cap(
            "google/gemini-flash",
            cost=0.0025,
            image=True,
            video=True,
            context=128000,
            name="Gemini Flash",
        )
    ]
    updated, lines = sync.refresh_existing_entries(entries, caps)
    row = updated[0]
    assert row["tier"] == "medium"  # NEVER changed by refresh
    assert row["assigned_by"] == "jai"
    assert row["supports_image"] is True
    assert row["supports_video"] is True
    assert row["cost"] == pytest.approx(0.0025)
    assert row["context_length"] == 128000
    assert any("gemini-flash" in line for line in lines)


def test_main_requires_mode_and_providers() -> None:
    with pytest.raises(SystemExit):
        sync.main(["--out", "x.yaml"])  # no mode, no providers
    with pytest.raises(SystemExit):
        sync.main(["--out", "x.yaml", "--only-existing"])  # no providers
