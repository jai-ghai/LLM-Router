"""Rule overrides: when these attributes match, use this tier (and optionally force a
selection mode). Most specific rule wins, then highest `priority` breaks a tie between equally
specific rules.

Trimmed port of the earlier R&D/llmrouter project's rules.py (ADR-0013 there): same matching
and precedence, no RuleStore/history (that was for a live-editable service, which does not
exist in this phase), and validation is against tier names from tiers.yaml rather than a model
registry.

A field's value may be a single string or a list of strings (OR match) — e.g. a rule that
fires for `task in {qc_review, audit}` without needing two separate rules.

Each rule may optionally pin an `optimize_for` mode ("cost"/"accuracy"/"balanced"). When set,
it overrides the per-request default — so an operator can mark "qc_review on video" as
always-accuracy without changing every caller (P2 — PLAN-v2.md §5).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

OptimizeFor = Literal["cost", "accuracy", "balanced"]
VALID_OPTIMIZE_FOR: tuple[OptimizeFor, ...] = ("cost", "accuracy", "balanced")


class RuleError(ValueError):
    """A rule is malformed, conflicts with another, or names an unknown tier."""


def canonical_key(value: object) -> str:
    return str(value).strip().lower()


def canonical_value(value: object) -> str:
    """Compare values without case, padding or leading zeros: '01010013' equals '1010013'."""
    text = str(value).strip()
    try:
        return str(int(text))
    except ValueError:
        return text.lower()


def _canonical_values(value: object) -> tuple[str, ...]:
    values = value if isinstance(value, (list, tuple, set)) and not isinstance(value, str) else [value]
    return tuple(sorted({canonical_value(v) for v in values}))


def _normalise(when: Mapping[str, object]) -> dict[str, tuple[str, ...]]:
    return {canonical_key(k): _canonical_values(v) for k, v in when.items()}


@dataclass(frozen=True)
class Rule:
    when: Mapping[str, tuple[str, ...]]
    tier: str
    priority: int = 0
    optimize_for: OptimizeFor | None = None
    """Optional per-rule selection mode. `None` (the default) means "use whatever the caller
    asked for, or the system default"; a value pins the mode regardless of what the caller
    passed in — handy for rules that encode a quality-sensitive task (e.g. qc_review on
    video)."""

    def matches(self, attributes: Mapping[str, str]) -> bool:
        return all(canonical_value(attributes.get(k, "")) in v for k, v in self.when.items())

    def describe(self) -> str:
        return " and ".join(f"{k}={'|'.join(v)}" for k, v in self.when.items())

    @classmethod
    def create(
        cls,
        when: Mapping[str, object],
        tier: str,
        priority: int = 0,
        optimize_for: OptimizeFor | None = None,
    ) -> Rule:
        if optimize_for is not None and optimize_for not in VALID_OPTIMIZE_FOR:
            raise RuleError(
                f"optimize_for must be one of {VALID_OPTIMIZE_FOR!r} or None; got '{optimize_for}'"
            )
        return cls(_normalise(when), tier.strip(), priority, optimize_for)


@dataclass(frozen=True)
class RuleSet:
    rules: tuple[Rule, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        seen: dict[tuple[tuple[str, tuple[str, ...]], ...], str] = {}
        for rule in self.rules:
            key = tuple(sorted(rule.when.items()))
            if key in seen and seen[key] != rule.tier:
                raise RuleError(f"conflicting rules for {rule.describe()}: {seen[key]} and {rule.tier}")
            seen[key] = rule.tier

    def is_empty(self) -> bool:
        return not self.rules

    def validate(self, known_tiers: set[str]) -> None:
        """Every tier a rule names must be one of tiers.yaml's tier names."""
        for rule in self.rules:
            if rule.tier not in known_tiers:
                known = ", ".join(sorted(known_tiers))
                raise RuleError(f"rule {rule.describe()} names unknown tier '{rule.tier}'. Known: {known}")

    def match(self, attributes: Mapping[str, str]) -> Rule | None:
        """The matching rule with the highest (priority, specificity), or None when none
        matches. Specificity (how many conditions) breaks a tie in priority; priority is the
        explicit override when an operator wants a less-specific rule to still win (for
        example, a blanket safety rule over a narrower content-type rule).

        Prefer this over `.resolve()` when the caller needs the rule's `optimize_for` or any
        other rule attribute beyond tier+reason.
        """
        best: Rule | None = None
        for rule in self.rules:
            if not rule.matches(attributes):
                continue
            if best is None or (rule.priority, len(rule.when)) > (best.priority, len(best.when)):
                best = rule
        return best

    def resolve(self, attributes: Mapping[str, str]) -> tuple[str, str] | None:
        """(tier, reason) from the matching rule, or None. Back-compat wrapper over `match()`;
        callers that need `optimize_for` should use `match()` directly.
        """
        rule = self.match(attributes)
        return (rule.tier, f"rule {rule.describe()}") if rule else None

    def coverage(self, attributes: Sequence[Mapping[str, str]]) -> int:
        """How many of these attribute-sets this ruleset would match — used by the UI to preview
        a rule's effect before saving it.
        """
        return sum(1 for attrs in attributes if self.resolve(attrs) is not None)


def rules_from_file(path: str | Path) -> RuleSet:
    """`rules: [{when: {...}, tier: ..., priority: 0}, ...]` from a YAML file. `priority` is
    optional and defaults to 0; a field's value under `when` may be a string or a list.
    """
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuleError(f"cannot read rules file {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("rules"), list):
        raise RuleError(f"{path}: expected a mapping with a 'rules' list")
    rules: list[Rule] = []
    for index, entry in enumerate(data["rules"]):
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("when"), dict)
            or not entry.get("tier")
        ):
            raise RuleError(f"{path}: rules[{index}] needs 'when' (a mapping) and 'tier'")
        when = _normalise(entry["when"])
        if not when:
            raise RuleError(f"{path}: rules[{index}] has an empty 'when'")
        try:
            priority = int(entry.get("priority", 0))
        except (TypeError, ValueError) as exc:
            raise RuleError(f"{path}: rules[{index}] has a non-integer 'priority'") from exc
        optimize_raw = entry.get("optimize_for")
        if optimize_raw is None:
            optimize_for: OptimizeFor | None = None
        else:
            text = str(optimize_raw).strip().lower()
            if text not in VALID_OPTIMIZE_FOR:
                raise RuleError(
                    f"{path}: rules[{index}] has 'optimize_for'='{optimize_raw}'; "
                    f"must be one of {VALID_OPTIMIZE_FOR!r}"
                )
            # Membership check above is enough for mypy to narrow `text` to OptimizeFor
            # (VALID_OPTIMIZE_FOR is a tuple of the Literal values).
            optimize_for = text
        rules.append(Rule(when, str(entry["tier"]).strip(), priority, optimize_for))
    return RuleSet(rules=tuple(rules))
