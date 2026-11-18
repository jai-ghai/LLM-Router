from __future__ import annotations

from pathlib import Path

import pytest

from tierrouter.rules import Rule, RuleError, RuleSet, canonical_value, rules_from_file


def test_values_compare_without_case_padding_or_zeros() -> None:
    assert canonical_value("1010013") == canonical_value(" 01010013 ") == canonical_value(1010013)
    assert canonical_value(" Hindi ") == "hindi"


def test_most_specific_rule_wins_ties_go_to_earlier() -> None:
    rules = RuleSet(
        rules=(
            Rule.create({"team": "news"}, "medium"),
            Rule.create({"team": "news", "language": "hi"}, "large"),
            Rule.create({"language": "hi"}, "medium"),
        )
    )
    assert rules.resolve({"team": "news", "language": "hi"}) == ("large", "rule team=news and language=hi")
    assert rules.resolve({"team": "news", "language": "en"}) == ("medium", "rule team=news")
    assert rules.resolve({"language": "HI"}) == ("medium", "rule language=hi")
    assert rules.resolve({}) is None


def test_conflicting_rules_are_refused() -> None:
    with pytest.raises(RuleError, match="conflicting"):
        RuleSet(rules=(Rule.create({"a": "1"}, "small"), Rule.create({"a": "1"}, "large")))


def test_validate_rejects_unknown_tiers() -> None:
    RuleSet(rules=(Rule.create({"a": "1"}, "small"),)).validate({"small", "medium", "large"})
    with pytest.raises(RuleError, match="unknown tier"):
        RuleSet(rules=(Rule.create({"a": "1"}, "giant"),)).validate({"small", "medium", "large"})


def test_rules_from_file(tmp_path: Path) -> None:
    path = tmp_path / "rules.yaml"
    path.write_text(
        "rules:\n  - when: {team: News}\n    tier: large\n  - when: {}\n    tier: small\n",
        encoding="utf-8",
    )
    with pytest.raises(RuleError, match="empty"):
        rules_from_file(path)


def test_rules_from_file_valid(tmp_path: Path) -> None:
    path = tmp_path / "rules.yaml"
    path.write_text("rules:\n  - when: {team: News}\n    tier: large\n", encoding="utf-8")
    rules = rules_from_file(path)
    assert rules.resolve({"team": "news"}) == ("large", "rule team=news")


@pytest.mark.parametrize("text", ["not: rules", "rules: [{tier: x}]", "rules: 1"])
def test_rules_from_file_rejects_bad_shapes(tmp_path: Path, text: str) -> None:
    path = tmp_path / "rules.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(RuleError):
        rules_from_file(path)
    with pytest.raises(RuleError):
        rules_from_file(tmp_path / "missing.yaml")


def test_empty_ruleset_resolves_nothing() -> None:
    assert RuleSet().is_empty()
    assert RuleSet().resolve({"team": "news"}) is None


def test_a_field_may_list_several_accepted_values() -> None:
    rule = Rule.create({"task": ["qc_review", "audit"]}, "large")
    assert rule.matches({"task": "qc_review"})
    assert rule.matches({"task": "AUDIT"})  # case-insensitive, same as a single value
    assert not rule.matches({"task": "headline_similarity"})
    assert rule.describe() == "task=audit|qc_review"


def test_priority_breaks_a_tie_between_equally_specific_rules() -> None:
    rules = RuleSet(
        rules=(
            Rule.create({"task": ["qc_review", "audit"]}, "large"),
            Rule.create({"task": "qc_review"}, "small", priority=5),
        )
    )
    # Both match "qc_review" with one condition each (equal specificity) — priority wins.
    assert rules.resolve({"task": "qc_review"}) == ("small", "rule task=qc_review")
    # Only the first rule matches "audit".
    assert rules.resolve({"task": "audit"}) == ("large", "rule task=audit|qc_review")


def test_rules_from_file_parses_lists_and_priority(tmp_path: Path) -> None:
    path = tmp_path / "rules.yaml"
    path.write_text(
        "rules:\n"
        "  - when: {task: [qc_review, audit]}\n    tier: large\n"
        "  - when: {task: qc_review}\n    tier: small\n    priority: 5\n",
        encoding="utf-8",
    )
    rules = rules_from_file(path)
    assert rules.resolve({"task": "qc_review"}) == ("small", "rule task=qc_review")


def test_coverage_counts_matching_attribute_sets() -> None:
    rules = RuleSet(rules=(Rule.create({"task": "qc_review"}, "large"),))
    attrs = [{"task": "qc_review"}, {"task": "headline_similarity"}, {"task": "qc_review"}]
    assert rules.coverage(attrs) == 2


# -----------------------------------------------------------------------------------------
# P2 — Rule.optimize_for (PLAN-v2.md §5). Two tests enumerated in §5.4, plus one for
# YAML rejecting an unknown mode.
# -----------------------------------------------------------------------------------------


def test_rule_carries_optimize_for_and_create_rejects_unknown() -> None:
    # Happy path: create() accepts a valid mode and stores it; match() returns the same
    # Rule so resolve() can read it. Bad mode: create() raises RuleError rather than
    # silently accepting a typo that would then fail later at pick() time.
    rule = Rule.create({"task": "qc_review"}, "large", optimize_for="accuracy")
    assert rule.optimize_for == "accuracy"
    assert Rule.create({"task": "qc_review"}, "large").optimize_for is None
    with pytest.raises(RuleError, match="optimize_for must be one of"):
        Rule.create({"task": "qc_review"}, "large", optimize_for="speed")  # type: ignore[arg-type]


def test_rules_from_file_parses_optimize_for(tmp_path: Path) -> None:
    # YAML round-trip: a rule with optimize_for loads correctly; a rule without it stays
    # None (so older rules.yaml files keep working without an upgrade migration).
    path = tmp_path / "rules.yaml"
    path.write_text(
        "rules:\n"
        "  - when: {task: qc_review}\n    tier: large\n    optimize_for: accuracy\n"
        "  - when: {task: headline}\n    tier: small\n",
        encoding="utf-8",
    )
    rules = rules_from_file(path)
    matched = rules.match({"task": "qc_review"})
    assert matched is not None and matched.optimize_for == "accuracy"
    matched_plain = rules.match({"task": "headline"})
    assert matched_plain is not None and matched_plain.optimize_for is None


def test_rules_from_file_rejects_unknown_optimize_for(tmp_path: Path) -> None:
    path = tmp_path / "rules.yaml"
    path.write_text(
        "rules:\n  - when: {task: qc_review}\n    tier: large\n    optimize_for: speed\n",
        encoding="utf-8",
    )
    with pytest.raises(RuleError, match="optimize_for"):
        rules_from_file(path)
