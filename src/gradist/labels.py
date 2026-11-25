"""Labeled training data: rule-derived seed labels (strategy 1) + human judgments (strategy 2).

Decision (2026-11-16): combine strategy 1 (today's known rules, turned into labels for free)
with strategy 2 (a small, deliberately-sampled human-judged set covering exactly what no rule
reaches). Strategies 3/4 (oracle-by-running, to cross-check 1+2 once they exist) and 6
(production feedback loop, for the long term) are later phases, not built here.

Flow: load_examples -> uncovered_examples (what no rule reaches) -> stratified_sample ->
export_for_human_judging -> (a person fills in "tier") -> load_human_judgments ->
merge_labels(labels_from_rules(...), human_labels).
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from gradist.rules import RuleSet

LabelSource = Literal["rule", "human"]


class LabelError(ValueError):
    """An examples or human-judgment file is malformed, or two labels for one example conflict."""


@dataclass(frozen=True)
class RawExample:
    """One unlabeled example: what the feature builder and classifier will see at training time."""

    example_id: str
    metadata: dict[str, str]
    text: str
    image_path: str | None = None
    """Path to an image file (relative to wherever the caller resolves it), or None. Resolved
    to bytes with `features.load_image_bytes` only when a classifier was given an image_embedder
    — an example with no image is a normal, valid input either way."""


@dataclass(frozen=True)
class TierLabel:
    """One training label, tagged with where it came from.

    `source` matters for evaluation, not only for training: a rule-derived label only ever
    repeats what the rule already encodes (a bias risk), so accuracy must be
    reported separately for rule-derived vs. human-judged examples — pooling them would make a
    classifier that merely learned the rules back look better than it is.
    """

    example_id: str
    metadata: dict[str, str]
    text: str
    tier: str
    source: LabelSource
    reason: str  # which rule matched, or who judged it and when
    image_path: str | None = None


def load_examples(path: str | Path) -> list[RawExample]:
    """JSONL, one `{"example_id": ..., "metadata": {...}, "text": ...}` object per line."""
    examples: list[RawExample] = []
    seen: set[str] = set()
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise LabelError(f"cannot read examples file {path}: {exc}") from exc
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LabelError(f"{path}:{line_no}: not valid JSON: {exc}") from exc
        if not isinstance(row, dict) or not row.get("example_id") or not isinstance(row.get("metadata"), dict):
            raise LabelError(f"{path}:{line_no}: needs 'example_id' and a 'metadata' mapping")
        example_id = str(row["example_id"])
        if example_id in seen:
            raise LabelError(f"{path}:{line_no}: duplicate example_id '{example_id}'")
        seen.add(example_id)
        examples.append(
            RawExample(
                example_id=example_id,
                metadata={str(k): str(v) for k, v in row["metadata"].items()},
                text=str(row.get("text") or ""),
                image_path=(str(row["image_path"]) if row.get("image_path") else None),
            )
        )
    if not examples:
        raise LabelError(f"{path}: no examples found")
    return examples


def labels_from_rules(examples: list[RawExample], rules: RuleSet) -> list[TierLabel]:
    """One label per example a rule actually matches. Examples no rule reaches are skipped —

    that gap is exactly what `uncovered_examples` and the human-judged set should cover.
    """
    labels = []
    for example in examples:
        hit = rules.resolve(example.metadata)
        if hit is not None:
            tier, reason = hit
            labels.append(
                TierLabel(
                    example.example_id, example.metadata, example.text, tier, "rule", reason, example.image_path
                )
            )
    return labels


def uncovered_examples(examples: list[RawExample], rules: RuleSet) -> list[RawExample]:
    """Examples no rule matches: the pool a human-judged sample should be drawn from, so judging

    effort is spent exactly where the rules have nothing to say.
    """
    return [ex for ex in examples if rules.resolve(ex.metadata) is None]


def stratified_sample(
    examples: list[RawExample], by_field: str, per_bucket: int, seed: int
) -> list[RawExample]:
    """Up to `per_bucket` examples per distinct value of `metadata[by_field]`, so a small

    human-judged set still spreads across (for example) every channel type instead of
    over-representing whichever one happens to be most common in `examples`. Examples missing
    `by_field` form their own "(missing)" bucket rather than being silently dropped.
    """
    buckets: dict[str, list[RawExample]] = defaultdict(list)
    for example in examples:
        buckets[example.metadata.get(by_field, "(missing)")].append(example)
    rng = random.Random(seed)
    sample: list[RawExample] = []
    for bucket in buckets.values():
        rng.shuffle(bucket)
        sample.extend(bucket[:per_bucket])
    return sample


def export_for_human_judging(examples: list[RawExample], path: str | Path) -> None:
    """Write a JSONL file for a person to fill in: each row gets an empty "tier" and

    "judged_by" for them to complete, plus the metadata and text (truncated) to judge from.
    Re-read it with `load_human_judgments` once it is filled in.
    """
    lines = []
    for example in examples:
        lines.append(
            json.dumps(
                {
                    "example_id": example.example_id,
                    "metadata": example.metadata,
                    "text": example.text[:500],
                    "image_path": example.image_path,
                    "tier": "",
                    "judged_by": "",
                    "notes": "",
                },
                ensure_ascii=False,
            )
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_human_judgments(
    path: str | Path,
    examples_by_id: dict[str, RawExample],
    known_tiers: set[str] | None = None,
) -> list[TierLabel]:
    """JSONL rows from `export_for_human_judging`, filled in: `example_id`, `tier`, `judged_by`

    required; `notes` optional. A row with an empty `tier` (not yet judged) is skipped, not an
    error. `example_id` must exist in `examples_by_id`, and `tier` must be in `known_tiers` when
    given — both raise rather than silently producing a label nothing can use.
    """
    labels: list[TierLabel] = []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise LabelError(f"cannot read human-judgment file {path}: {exc}") from exc
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LabelError(f"{path}:{line_no}: not valid JSON: {exc}") from exc
        if not isinstance(row, dict) or not row.get("example_id"):
            raise LabelError(f"{path}:{line_no}: needs 'example_id'")
        tier = str(row.get("tier") or "").strip()
        if not tier:
            continue  # not judged yet
        example_id = str(row["example_id"])
        example = examples_by_id.get(example_id)
        if example is None:
            raise LabelError(f"{path}:{line_no}: example_id '{example_id}' is not in the example pool")
        if known_tiers is not None and tier not in known_tiers:
            raise LabelError(f"{path}:{line_no}: '{tier}' is not a declared tier. Known: {sorted(known_tiers)}")
        judged_by = str(row.get("judged_by") or "").strip()
        if not judged_by:
            raise LabelError(f"{path}:{line_no}: a judged row needs 'judged_by'")
        reason = f"judged by {judged_by}"
        if row.get("notes"):
            reason += f" ({row['notes']})"
        labels.append(
            TierLabel(example_id, example.metadata, example.text, tier, "human", reason, example.image_path)
        )
    return labels


def merge_labels(rule_labels: list[TierLabel], human_labels: list[TierLabel]) -> list[TierLabel]:
    """Combine both sources. A human judgment overrides a rule label for the same example_id —

    a person deliberately looked at a case the rule also covered, which is more trustworthy
    evidence than the rule guessing at it, and it is not subject to the rule's bias.
    """
    by_id: dict[str, TierLabel] = {label.example_id: label for label in rule_labels}
    for label in human_labels:
        by_id[label.example_id] = label
    return list(by_id.values())
