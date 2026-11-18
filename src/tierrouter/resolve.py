"""The final piece: rule -> classifier -> low-confidence fallback -> a model name.

No model is called anywhere in this package (ARCHITECTURE.md §1). `resolve()` is the one
function a consumer actually needs; everything else in this package exists to build its inputs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from tierrouter.catalog import ModelCatalog, OptimizeFor
from tierrouter.classifier import TierClassifier
from tierrouter.rules import RuleSet, canonical_value
from tierrouter.tiers import TierTable


@dataclass(frozen=True)
class Decision:
    tier: str
    model_name: str
    confidence: float | None
    """1.0 for a rule match (an operator's explicit override). The classifier's own probability
    when it decided. None when neither a rule nor a classifier produced one."""
    reason: str
    rule_outcome: str = "no matching rule"
    classifier_outcome: str | None = None
    """None means "not reached" — a rule matched, so the classifier never ran."""
    fallback_outcome: str | None = None
    """None means "not used". The structured counterpart to `reason`'s prose, one entry per
    precedence step (rule, then classifier, then fallback) — a UI can render this as a trace
    instead of parsing the sentence in `reason`."""
    optimize_for: OptimizeFor = "cost"
    """Which mode actually drove the catalog pick: the request's value, unless a matching rule
    pinned its own — then the rule's wins (so a UI can show the operator what really ran)."""


def validate_setup(rules: RuleSet, tiers: TierTable) -> None:
    """Call once at startup, not per request: every rule's tier must be declared in tiers.yaml."""
    rules.validate(tiers.names)


def resolve(
    metadata: dict[str, str],
    text: str,
    rules: RuleSet,
    tiers: TierTable,
    classifier: TierClassifier | None = None,
    provider: str | None = None,
    min_confidence: float = 0.0,
    image_bytes: bytes | None = None,
    catalog: ModelCatalog | None = None,
    selected_models: set[str] | None = None,
    quality: Mapping[tuple[str, str], float] | None = None,
    quality_field: str | None = None,
    min_quality: float | None = None,
    optimize_for: OptimizeFor = "cost",
) -> Decision:
    """Precedence: a matching rule wins outright, then the classifier, then the fallback tier.

    `min_confidence` is a trust threshold, not the classifier's own concern (ARCHITECTURE.md §1
    keeps "should this prediction be trusted" out of the classifier): a prediction below it is
    replaced by `tiers.low_confidence_fallback`, visibly, with the real confidence still reported
    so a caller can see how low it was. Its default, 0.0, trusts every classifier prediction;
    set it explicitly once there is evidence (PLAN.md §8) for where the cutoff should be.

    `catalog`/`selected_models` are optional: without a catalog, the tier maps to the single
    model `tiers.yaml` names for it (the original design). With one, the cheapest catalog model
    at the decided tier is used instead, restricted to `selected_models` if given (`None` means
    "every model in the catalog is a candidate") — capability is never silently downgraded below
    the tier that was actually decided; see `catalog.py`.

    `quality`/`quality_field`/`min_quality` add a human-reviewed quality floor on top of cost
    (see quality.py) — pass `quality_field` as whichever metadata field a deployment buckets
    reviews by (e.g. "task"); this request's own `metadata[quality_field]` is looked up
    automatically. Omit `min_quality` (the default) to ignore quality entirely and pick on cost
    alone, as before this existed.

    `optimize_for` (P2 — PLAN-v2.md §5) picks which candidate wins at the decided tier: "cost"
    (cheapest, default), "accuracy" (highest human rating), or "balanced" (quality/cost
    heuristic). A matching rule's own `optimize_for` takes precedence over this argument — an
    operator pinning "qc_review on video → accuracy" wins even if a caller forgot to ask for
    it.
    """
    effective_optimize_for: OptimizeFor = optimize_for
    matched_rule = rules.match(metadata)
    if matched_rule is not None:
        tier = matched_rule.tier
        reason = f"rule {matched_rule.describe()}"
        confidence: float | None = 1.0
        rule_outcome = reason
        classifier_outcome: str | None = None
        fallback_outcome: str | None = None
        if matched_rule.optimize_for is not None:
            effective_optimize_for = matched_rule.optimize_for
    elif classifier is not None:
        predicted_tier, predicted_confidence = classifier.predict(metadata, text, image_bytes)
        rule_outcome = "no matching rule"
        if predicted_confidence < min_confidence:
            tier = tiers.low_confidence_fallback
            confidence = predicted_confidence
            reason = (
                f"classifier confidence {predicted_confidence:.2f} is below the "
                f"{min_confidence:.2f} threshold; used the fallback tier '{tier}'"
            )
            classifier_outcome = f"{predicted_tier}, confidence {predicted_confidence:.2f} (below the {min_confidence:.2f} threshold)"
            fallback_outcome = f"used '{tier}'"
        else:
            tier = predicted_tier
            confidence = predicted_confidence
            reason = f"classifier, confidence {predicted_confidence:.2f}"
            classifier_outcome = f"{tier}, confidence {predicted_confidence:.2f}"
            fallback_outcome = "not needed"
    else:
        tier = tiers.low_confidence_fallback
        confidence = None
        reason = f"no rule matched and no classifier is available; used the fallback tier '{tier}'"
        rule_outcome = "no matching rule"
        classifier_outcome = "no classifier configured"
        fallback_outcome = f"used '{tier}'"

    if catalog is not None:
        quality_value = canonical_value(metadata.get(quality_field, "")) if quality_field else None
        entry, note = catalog.pick(
            tier,
            tiers.rank(),
            selected=selected_models,
            provider=provider,
            needs_image=image_bytes is not None,
            quality=quality,
            quality_value=quality_value or None,
            min_quality=min_quality,
            optimize_for=effective_optimize_for,
        )
        model_name = entry.id
        if note:
            reason = f"{reason}; {note}"
    else:
        model_name = tiers.model_for(tier, provider)
    return Decision(
        tier=tier,
        model_name=model_name,
        confidence=confidence,
        reason=reason,
        rule_outcome=rule_outcome,
        classifier_outcome=classifier_outcome,
        fallback_outcome=fallback_outcome,
        optimize_for=effective_optimize_for,
    )
