from __future__ import annotations

from pathlib import Path

import pytest

from gradist.labels import (
    LabelError,
    RawExample,
    export_for_human_judging,
    labels_from_rules,
    load_examples,
    load_human_judgments,
    merge_labels,
    stratified_sample,
    uncovered_examples,
)
from gradist.rules import Rule, RuleSet

EXAMPLES = [
    RawExample("e1", {"channel_type": "news"}, "parliament passed the budget"),
    RawExample("e2", {"channel_type": "sports"}, "goal scored in the final"),
    RawExample("e3", {"channel_type": "weather"}, "heavy rain expected tomorrow"),
]
RULES = RuleSet(rules=(Rule.create({"channel_type": "news"}, "small"),))


def _write_jsonl(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_load_examples(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "ex.jsonl",
        ['{"example_id": "e1", "metadata": {"channel_type": "news"}, "text": "hi"}'],
    )
    loaded = load_examples(path)
    assert loaded == [RawExample("e1", {"channel_type": "news"}, "hi")]


@pytest.mark.parametrize(
    "lines",
    [
        ["not json"],
        ['{"metadata": {}}'],
        ['{"example_id": "e1", "metadata": "nope"}'],
        ['{"example_id": "e1", "metadata": {}}', '{"example_id": "e1", "metadata": {}}'],
    ],
)
def test_load_examples_rejects_bad_rows(tmp_path: Path, lines: list[str]) -> None:
    with pytest.raises(LabelError):
        load_examples(_write_jsonl(tmp_path / "ex.jsonl", lines))
    with pytest.raises(LabelError):
        load_examples(tmp_path / "missing.jsonl")


def test_labels_from_rules_and_uncovered_are_complementary() -> None:
    rule_labels = labels_from_rules(EXAMPLES, RULES)
    uncovered = uncovered_examples(EXAMPLES, RULES)
    assert [label.example_id for label in rule_labels] == ["e1"]
    assert rule_labels[0].source == "rule"
    assert [ex.example_id for ex in uncovered] == ["e2", "e3"]


def test_stratified_sample_spreads_across_buckets_and_missing_field() -> None:
    examples = [RawExample(f"e{i}", {"channel_type": "news"}, "") for i in range(5)]
    examples += [RawExample("s1", {"channel_type": "sports"}, ""), RawExample("m1", {}, "")]
    sample = stratified_sample(examples, "channel_type", per_bucket=2, seed=0)
    buckets = {ex.metadata.get("channel_type", "(missing)") for ex in sample}
    assert buckets == {"news", "sports", "(missing)"}
    assert len([ex for ex in sample if ex.metadata.get("channel_type") == "news"]) == 2


def test_stratified_sample_is_deterministic() -> None:
    examples = [RawExample(f"e{i}", {"c": "a"}, "") for i in range(10)]
    a = stratified_sample(examples, "c", per_bucket=3, seed=7)
    b = stratified_sample(examples, "c", per_bucket=3, seed=7)
    assert [ex.example_id for ex in a] == [ex.example_id for ex in b]


def test_export_then_load_human_judgments(tmp_path: Path) -> None:
    export_path = tmp_path / "to_judge.jsonl"
    export_for_human_judging(EXAMPLES[1:], export_path)
    rows = export_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2
    assert '"tier": ""' in rows[0]

    # simulate a person filling in one row and leaving the other blank
    filled = rows[0].replace('"tier": ""', '"tier": "large"').replace('"judged_by": ""', '"judged_by": "jai"')
    judged_path = tmp_path / "judged.jsonl"
    judged_path.write_text(filled + "\n" + rows[1], encoding="utf-8")

    by_id = {ex.example_id: ex for ex in EXAMPLES}
    labels = load_human_judgments(judged_path, by_id, known_tiers={"small", "medium", "large"})
    assert len(labels) == 1
    assert labels[0].example_id == "e2"
    assert labels[0].tier == "large"
    assert labels[0].source == "human"
    assert "jai" in labels[0].reason


def test_load_human_judgments_rejects_unknown_example_tier_or_missing_judge(tmp_path: Path) -> None:
    by_id = {ex.example_id: ex for ex in EXAMPLES}
    bad_id = tmp_path / "a.jsonl"
    bad_id.write_text('{"example_id": "nope", "tier": "small", "judged_by": "x"}', encoding="utf-8")
    with pytest.raises(LabelError, match="not in the example pool"):
        load_human_judgments(bad_id, by_id)

    bad_tier = tmp_path / "b.jsonl"
    bad_tier.write_text('{"example_id": "e1", "tier": "giant", "judged_by": "x"}', encoding="utf-8")
    with pytest.raises(LabelError, match="not a declared tier"):
        load_human_judgments(bad_tier, by_id, known_tiers={"small", "medium", "large"})

    no_judge = tmp_path / "c.jsonl"
    no_judge.write_text('{"example_id": "e1", "tier": "small", "judged_by": ""}', encoding="utf-8")
    with pytest.raises(LabelError, match="judged_by"):
        load_human_judgments(no_judge, by_id)


def test_merge_labels_human_overrides_rule_on_the_same_example(tmp_path: Path) -> None:
    rule_labels = labels_from_rules(EXAMPLES, RULES)  # e1 -> small, by rule
    by_id = {ex.example_id: ex for ex in EXAMPLES}
    override_path = tmp_path / "override.jsonl"
    override_path.write_text('{"example_id": "e1", "tier": "large", "judged_by": "jai"}', encoding="utf-8")
    human_labels = load_human_judgments(override_path, by_id)

    merged = merge_labels(rule_labels, human_labels)
    assert len(merged) == 1
    winner = merged[0]
    assert winner.tier == "large"
    assert winner.source == "human"


def test_merge_labels_keeps_rule_labels_untouched_otherwise() -> None:
    rule_labels = labels_from_rules(EXAMPLES, RULES)
    merged = merge_labels(rule_labels, [])
    assert merged == rule_labels
