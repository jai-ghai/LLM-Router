# [Fact-Forcing]
# Callers: gradist.service.app.do_resolve; tests/test_resolve.py; experiments smoke scripts.
# This IS resolve.py — rewritten so catalog no longer skips the tier ladder. Reads Decision
# inputs (metadata/text/image); writes Decision. No secrets.
# User: "i want all models and providers .. generalized .. train better use hugging face
# data .. dynamic and read the content .. don't just set the router for fixed"
"""The final piece: rule -> classifier -> low-confidence fallback -> a model name.

No model is called anywhere in this package (ARCHITECTURE.md §1). `resolve()` is the one
function a consumer actually needs; everything else in this package exists to build its inputs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from gradist.catalog import ModelCatalog, OptimizeFor
from gradist.classifier import TierClassifier
from gradist.qc_risk import QcRiskHead
from gradist.quality_head import TierQualityHead
from gradist.rules import RuleSet, canonical_value
from gradist.tiers import TierTable

# Metadata keys that alone do not count as "content" for the empty-request guard.
_WEAK_META_KEYS = frozenset({"modality", "provider", "optimize_for"})
# `task` names a model pool (or a rule key, already handled by the rule step), not content
# the domain classifier was trained on — a text-only request with just a task is "generic".
_NON_DOMAIN_META_KEYS = _WEAK_META_KEYS | {"task"}


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


def has_routing_signal(
    metadata: dict[str, str],
    text: str,
    image_bytes: bytes | None = None,
) -> bool:
    """True when there is something for rules/classifier to read (text, image, or metadata)."""
    if image_bytes:
        return True
    if text and text.strip():
        return True
    for key, value in metadata.items():
        if key in _WEAK_META_KEYS:
            continue
        if str(value).strip():
            return True
    return False


def has_domain_signal(metadata: dict[str, str], image_bytes: bytes | None = None) -> bool:
    """True when the request carries an image or domain metadata (channel, genre, pipeline, …) —
    what the domain classifier was trained on. Plain text without it goes to the quality head."""
    if image_bytes:
        return True
    return any(str(v).strip() for k, v in metadata.items() if k not in _NON_DOMAIN_META_KEYS)


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
    quality_head: TierQualityHead | None = None,
    qc_risk: QcRiskHead | None = None,
) -> Decision:
    """Precedence: a matching rule wins outright, then the classifier, then the fallback tier.

    The classifier step has two models: requests with an image or domain metadata use the domain
    `classifier`; plain-text requests use `quality_head` when one is loaded (it predicts each
    tier's score and takes the cheapest within the `optimize_for` margin), else the classifier.

    After that, `qc_risk` (when loaded) may raise — never lower — the tier of a tabson-shaped
    sub-story whose predicted chance of needing a QC fix is above its threshold. A matching rule
    is an operator's explicit decision and is never escalated.

    `min_confidence` is a trust threshold, not the classifier's own concern (ARCHITECTURE.md §1
    keeps "should this prediction be trusted" out of the classifier): a prediction below it is
    replaced by `tiers.low_confidence_fallback`, visibly, with the real confidence still reported
    so a caller can see how low it was. Its default, 0.0, trusts every classifier prediction;
    set it explicitly once there is evidence (PLAN.md §8) for where the cutoff should be.

    Without a catalog, the decided tier maps through `tiers.yaml` to one model name per provider.

    With a catalog, the same rule → classifier → fallback ladder still decides the **tier**;
    then `catalog.pick(tier, …)` chooses among all providers/models at that tier (or above),
    filtered by selection / provider / modality / quality / `optimize_for`. Empty requests
    (no text, image, or useful metadata) skip the classifier and use the fallback tier so a
    blank Playground click cannot look like a confident content decision.

    Modality: `metadata["modality"] == "video"|"audio"` (via `canonical_value`) and whether
    `image_bytes` was supplied filter the catalog pool by capability.
    """
    effective_optimize_for: OptimizeFor = optimize_for
    quality_value = canonical_value(metadata.get(quality_field, "")) if quality_field else None
    modality = canonical_value(metadata.get("modality", ""))
    task = canonical_value(metadata.get("task", "")) if metadata.get("task") else ""

    if not has_routing_signal(metadata, text, image_bytes):
        tier = tiers.low_confidence_fallback
        confidence: float | None = 0.0
        reason = (
            f"empty request (no text, image, or metadata); "
            f"used the fallback tier '{tier}'"
        )
        rule_outcome = "no matching rule"
        classifier_outcome = "skipped — empty request"
        fallback_outcome = f"used '{tier}'"
    else:
        matched_rule = rules.match(metadata)
        if matched_rule is not None:
            tier = matched_rule.tier
            reason = f"rule {matched_rule.describe()}"
            confidence = 1.0
            rule_outcome = reason
            classifier_outcome = None
            fallback_outcome = None
            if matched_rule.optimize_for is not None:
                effective_optimize_for = matched_rule.optimize_for
        elif (
            quality_head is not None
            and text.strip()
            and not has_domain_signal(metadata, image_bytes)
            and set(quality_head.tiers) <= tiers.names
        ):
            tier, scores, margin = quality_head.choose(text, effective_optimize_for)
            confidence = None
            score_text = ", ".join(f"{t}={scores[t]:.2f}" for t in quality_head.tiers)
            reason = (
                f"quality head (text only): predicted scores {score_text}; "
                f"cheapest tier within {margin:.2f} of the best ({effective_optimize_for}) is '{tier}'"
            )
            rule_outcome = "no matching rule"
            classifier_outcome = reason
            fallback_outcome = "not needed"
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
                classifier_outcome = (
                    f"{predicted_tier}, confidence {predicted_confidence:.2f} "
                    f"(below the {min_confidence:.2f} threshold)"
                )
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
            reason = (
                f"no rule matched and no classifier is available; "
                f"used the fallback tier '{tier}'"
            )
            rule_outcome = "no matching rule"
            classifier_outcome = "no classifier configured"
            fallback_outcome = f"used '{tier}'"

    if (
        qc_risk is not None
        and qc_risk.threshold is not None
        and rule_outcome == "no matching rule"
        and qc_risk.escalate_to in tiers.names
        and qc_risk.applies_to(metadata, text)
    ):
        ranks = tiers.rank()
        risk = qc_risk.risk(metadata, text)
        if risk >= qc_risk.threshold and ranks[tier] < ranks[qc_risk.escalate_to]:
            reason = (
                f"{reason}; QC risk {risk:.2f} ≥ {qc_risk.threshold:.2f} (riskiest sub-stories) "
                f"→ escalated '{tier}' to '{qc_risk.escalate_to}'"
            )
            tier = qc_risk.escalate_to

    if catalog is not None:
        entry, note = catalog.pick(
            tier,
            tiers.rank(),
            selected=selected_models,
            provider=provider,
            needs_image=image_bytes is not None,
            needs_video=modality == "video",
            needs_audio=modality == "audio",
            quality=quality,
            quality_value=quality_value or None,
            min_quality=min_quality,
            optimize_for=effective_optimize_for,
        )
        pick_reason = (
            f"{reason} → catalog tier='{tier}'"
            + (f", task='{task}'" if task else "")
            + f", optimize_for='{effective_optimize_for}'"
            + f" → '{entry.id}' ({entry.provider})"
        )
        if note:
            pick_reason = f"{pick_reason}; {note}"
        return Decision(
            tier=entry.tier,
            model_name=entry.id,
            confidence=confidence,
            reason=pick_reason,
            rule_outcome=rule_outcome,
            classifier_outcome=classifier_outcome,
            fallback_outcome=fallback_outcome,
            optimize_for=effective_optimize_for,
        )

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
