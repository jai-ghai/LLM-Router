from __future__ import annotations

from pathlib import Path

import pytest

from tierrouter.catalog import CatalogEntry, ModelCatalog, load_catalog
from tierrouter.rules import RuleError

RANK = {"small": 0, "medium": 1, "large": 2}

CATALOG = ModelCatalog(
    entries=(
        CatalogEntry("a-small", "gemini", "small", cost=1, supports_image=False, label="A Small"),
        CatalogEntry("b-small", "gemini", "small", cost=2, supports_image=True, label="B Small"),
        CatalogEntry("a-medium", "gemini", "medium", cost=4, supports_image=True, label="A Medium"),
        CatalogEntry("a-large", "anthropic", "large", cost=10, supports_image=True, label="A Large"),
    )
)


def test_picks_the_cheapest_exact_tier_match() -> None:
    entry, note = CATALOG.pick("small", RANK)
    assert entry.id == "a-small"
    assert note is None


def test_image_requirement_filters_out_non_image_models() -> None:
    entry, note = CATALOG.pick("small", RANK, needs_image=True)
    assert entry.id == "b-small"  # a-small is cheaper but doesn't support images
    assert note is None


def test_selection_restricts_the_candidate_pool() -> None:
    entry, _note = CATALOG.pick("small", RANK, selected={"b-small"})
    assert entry.id == "b-small"


def test_never_downgrades_below_the_decided_tier() -> None:
    # Only "large" is available once restricted to anthropic — a "small" decision still
    # gets a capable model, never something cheaper/weaker than what was decided.
    entry, note = CATALOG.pick("small", RANK, provider="anthropic")
    assert entry.id == "a-large"
    assert note is not None and "never downgraded" in note


def test_raises_when_nothing_matches_the_filter() -> None:
    with pytest.raises(RuleError, match="no catalog model matches"):
        CATALOG.pick("small", RANK, selected={"does-not-exist"})


def test_raises_when_nothing_at_or_above_the_tier_is_available() -> None:
    with pytest.raises(RuleError, match="tier .* or above"):
        CATALOG.pick("large", RANK, selected={"a-small"})


def test_duplicate_ids_are_refused() -> None:
    with pytest.raises(RuleError, match="duplicate"):
        ModelCatalog(entries=(CatalogEntry("x", "gemini", "small", 1), CatalogEntry("x", "gemini", "large", 2)))


def test_load_catalog_from_file(tmp_path: Path) -> None:
    path = tmp_path / "catalog.yaml"
    path.write_text(
        "catalog:\n"
        "  - id: a-small\n    provider: gemini\n    tier: small\n    cost: 1\n"
        "  - id: a-large\n    provider: gemini\n    tier: large\n    cost: 10\n"
        "    supports_image: false\n    label: A Large\n",
        encoding="utf-8",
    )
    catalog = load_catalog(path)
    assert len(catalog.entries) == 2
    large = next(e for e in catalog.entries if e.id == "a-large")
    assert large.supports_image is False
    assert large.label == "A Large"


@pytest.mark.parametrize("text", ["not: catalog", "catalog: [{id: x}]", "catalog: []"])
def test_load_catalog_rejects_bad_shapes(tmp_path: Path, text: str) -> None:
    path = tmp_path / "catalog.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(RuleError):
        load_catalog(path)
    with pytest.raises(RuleError):
        load_catalog(tmp_path / "missing.yaml")


def test_load_catalog_parses_context_length(tmp_path: Path) -> None:
    path = tmp_path / "catalog.yaml"
    path.write_text(
        "catalog:\n  - id: a\n    provider: gemini\n    tier: small\n    cost: 1\n    context_length: 131072\n",
        encoding="utf-8",
    )
    entry = load_catalog(path).entries[0]
    assert entry.context_length == 131072


def test_quality_floor_excludes_a_reviewed_but_low_scoring_model() -> None:
    # Both are "medium". a-medium is cheaper but scores 2.0; b-medium scores 4.5.
    catalog = ModelCatalog(
        entries=(
            CatalogEntry("a-medium", "gemini", "medium", cost=1),
            CatalogEntry("b-medium", "gemini", "medium", cost=5),
        )
    )
    quality = {("a-medium", "qc_review"): 2.0, ("b-medium", "qc_review"): 4.5}
    entry, note = catalog.pick("medium", RANK, quality=quality, quality_value="qc_review", min_quality=3.0)
    assert entry.id == "b-medium"
    assert note is not None and "excluded" in note


def test_quality_floor_never_excludes_an_unreviewed_model() -> None:
    catalog = ModelCatalog(entries=(CatalogEntry("unreviewed", "gemini", "medium", cost=1),))
    entry, note = catalog.pick("medium", RANK, quality={}, quality_value="qc_review", min_quality=3.0)
    assert entry.id == "unreviewed"
    assert note is None


def test_quality_floor_is_ignored_rather_than_failing_the_request_if_nothing_passes() -> None:
    catalog = ModelCatalog(entries=(CatalogEntry("weak", "gemini", "medium", cost=1),))
    quality = {("weak", "qc_review"): 1.0}
    entry, note = catalog.pick("medium", RANK, quality=quality, quality_value="qc_review", min_quality=4.0)
    assert entry.id == "weak"
    assert note is not None and "ignored the bar" in note
