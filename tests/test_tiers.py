from __future__ import annotations

from pathlib import Path

import pytest

from gradist.rules import RuleError
from gradist.tiers import load_tiers


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


GOOD = """
tiers:
  - name: small
    models: {gemini: g-small, anthropic: a-small}
  - name: medium
    models: {gemini: g-medium, anthropic: a-medium}
default_provider: gemini
low_confidence_fallback: medium
"""


def test_load_and_model_for(tmp_path: Path) -> None:
    table = load_tiers(_write(tmp_path / "tiers.yaml", GOOD))
    assert table.names == {"small", "medium"}
    assert table.model_for("small") == "g-small"
    assert table.model_for("small", provider="anthropic") == "a-small"


def test_unknown_tier_or_provider_raises(tmp_path: Path) -> None:
    table = load_tiers(_write(tmp_path / "tiers.yaml", GOOD))
    with pytest.raises(RuleError, match="not a declared tier"):
        table.model_for("large")
    with pytest.raises(RuleError, match="no model for provider"):
        table.model_for("small", provider="bedrock")


def test_fallback_must_be_a_declared_tier(tmp_path: Path) -> None:
    bad = GOOD.replace("low_confidence_fallback: medium", "low_confidence_fallback: giant")
    with pytest.raises(RuleError, match="not a declared tier"):
        load_tiers(_write(tmp_path / "tiers.yaml", bad))


def test_duplicate_tier_names_are_refused(tmp_path: Path) -> None:
    dup = GOOD.replace("name: medium", "name: small")
    with pytest.raises(RuleError, match="duplicate"):
        load_tiers(_write(tmp_path / "tiers.yaml", dup))


@pytest.mark.parametrize(
    "text",
    ["not: a table", "tiers: []", "tiers: [{models: {}}]", "tiers: [{name: x, models: {}}]\n"],
)
def test_malformed_files_are_refused(tmp_path: Path, text: str) -> None:
    with pytest.raises(RuleError):
        load_tiers(_write(tmp_path / "tiers.yaml", text))
    with pytest.raises(RuleError):
        load_tiers(tmp_path / "missing.yaml")


def test_example_config_loads(tmp_path: Path) -> None:
    example = Path(__file__).resolve().parents[1] / "config" / "tiers.example.yaml"
    table = load_tiers(example)
    assert table.names == {"small", "medium", "large"}
    assert table.low_confidence_fallback == "medium"
