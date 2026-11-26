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

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field

from gradist.catalog import ModelCatalog, OptimizeFor
from gradist.classifier import TierClassifier
from gradist.feedback import SkillEvidence
from gradist.fit import describe_fit
from gradist.qc_risk import QcRiskHead
from gradist.quality_head import TierQualityHead
from gradist.rules import RuleSet, canonical_value
from gradist.skills import SkillNeeds
from gradist.tiers import TierTable

# Metadata keys that alone do not count as "content" for the empty-request guard.
_WEAK_META_KEYS = frozenset({"modality", "provider", "optimize_for"})
# `task` names a model pool (or a rule key, already handled by the rule step), not content
# the domain classifier was trained on — a text-only request with just a task is "generic".
_NON_DOMAIN_META_KEYS = _WEAK_META_KEYS | {"task"}

# Classifier path: pick the highest tier whose "needs at least this tier" probability reaches
# the mode's share — cost only upgrades when fairly sure, accuracy upgrades on a real chance.
NEED_SHARE: dict[OptimizeFor, float] = {"cost": 0.7, "balanced": 0.5, "accuracy": 0.3}
SHORTLIST_SIZE = 8
MOVED_BY_RATINGS_SHOWN = 5


def tier_by_need(
    proba: Mapping[str, float], ranks: Mapping[str, int], optimize_for: OptimizeFor
) -> tuple[str, dict[str, float]]:
    """(tier, P(needs ≥ tier) per tier) — the highest tier whose share reaches NEED_SHARE."""
    ordered = sorted(ranks, key=lambda t: ranks[t])
    at_least: dict[str, float] = {}
    running = 0.0
    for name in reversed(ordered):
        running += proba.get(name, 0.0)
        at_least[name] = min(running, 1.0)
    share = NEED_SHARE[optimize_for]
    chosen = next((t for t in reversed(ordered) if at_least[t] >= share), ordered[0])
    return chosen, {t: at_least[t] for t in ordered}


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
    tier_scores: dict[str, float] | None = None
    """Per-tier evidence behind the tier: the quality head's predicted scores, or the
    classifier's P(needs at least this tier). None for rules and fallbacks."""
    tier_scores_kind: str | None = None
    """"predicted score" or "P(needs ≥ tier)" — how to read `tier_scores`."""
    candidates: tuple[dict[str, object], ...] = ()
    """Catalog shortlist the model was picked from, best first (at most SHORTLIST_SIZE)."""
    pool_size: int | None = None
    """Catalog models left after the selection/provider/modality filters, all tiers."""
    signals: dict[str, object] = field(default_factory=dict)
    """What the router read from the request (text size, metadata used/ignored, media)."""
    request_id: str | None = None
    """Set by the service; rate the pick later with POST /v1/quality {request_id}."""


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
    provider: str | Collection[str] | None = None,
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
    screen_text: str = "",
    skill_needs: SkillNeeds | None = None,
    learned_skills: Mapping[str, Mapping[str, SkillEvidence]] | None = None,
) -> Decision:
    """Precedence: a matching rule wins outright, then the classifier, then the fallback tier.

    `screen_text` is OCR text read from the image; the classifier reads it only if it was
    trained with screen text.

    `skill_needs` (skills.request_needs) makes the catalog rank the decided tier's models by how
    well they fit those needs instead of a fixed sort; each candidate then carries its fit.
    `learned_skills` (feedback.learn_skills) adjusts those skills by past ratings.

    The classifier step has two models: requests with an image or domain metadata use the domain
    `classifier`; plain-text requests use `quality_head` when one is loaded (it predicts each
    tier's score and takes the cheapest within the `optimize_for` margin), else the classifier.
    The classifier's tier is the highest one whose P(needs at least this tier) reaches
    `NEED_SHARE[optimize_for]`, so the cost/balanced/accuracy dial moves the tier on both paths.

    After that, `qc_risk` (when loaded) may raise — never lower — the tier of a tabson-shaped
    sub-story whose predicted chance of needing a QC fix is above its threshold. A matching rule
    is an operator's explicit decision and is never escalated.

    `min_confidence` is a trust threshold, not the classifier's own concern (ARCHITECTURE.md §1
    keeps "should this prediction be trusted" out of the classifier): a prediction below it is
    replaced by `tiers.low_confidence_fallback`, visibly, with the real confidence still reported
    so a caller can see how low it was. Its default, 0.0, trusts every classifier prediction;
    set it explicitly once there is evidence for where the cutoff should be.

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
    providers = sorted(
        {p.strip() for p in (provider.split(",") if isinstance(provider, str) else provider or ()) if p.strip()}
    )
    signals: dict[str, object] = {
        "text_chars": len(text.strip()),
        "text_words": len(text.split()),
        "metadata": dict(metadata),
        "image_bytes": len(image_bytes) if image_bytes else 0,
        "modality": modality or ("image" if image_bytes else "text"),
        "providers": providers,
        "selection_size": len(selected_models) if selected_models is not None else None,
    }
    if skill_needs is not None:
        signals["skill_needs"] = skill_needs.as_signal()
    tier_scores: dict[str, float] | None = None
    tier_scores_kind: str | None = None

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
            tier_scores, tier_scores_kind = dict(scores), "predicted score"
            signals["path"] = "quality head (plain text)"
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
            evidence = classifier.explain(metadata, text, image_bytes, screen_text)
            proba = evidence.pop("proba")
            signals["classifier_inputs"] = evidence
            predicted_tier = max(proba, key=lambda t: proba[t])
            predicted_confidence = proba[predicted_tier]
            signals["path"] = "domain classifier"
            signals["metadata_used"] = sorted(k for k in metadata if k in classifier.metadata_fields)
            signals["metadata_ignored"] = sorted(
                k for k in metadata if k not in classifier.metadata_fields and k not in _WEAK_META_KEYS
            )
            rule_outcome = "no matching rule"
            dial_tier, at_least = tier_by_need(proba, tiers.rank(), effective_optimize_for)
            tier_scores, tier_scores_kind = at_least, "P(needs ≥ tier)"
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
                tier = dial_tier
                confidence = predicted_confidence
                need_text = ", ".join(f"≥{t} {p:.2f}" for t, p in reversed(at_least.items()))
                reason = (
                    f"classifier, confidence {predicted_confidence:.2f} (most likely '{predicted_tier}'); "
                    f"P(needs {need_text}); {effective_optimize_for} takes the highest tier with "
                    f"P ≥ {NEED_SHARE[effective_optimize_for]:.1f} → '{tier}'"
                )
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
        ranking = catalog.pick_ranked(
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
            needs=skill_needs,
            learned=learned_skills,
        )
        entry, note = ranking.ranked[0], ranking.note
        winner_fit = ranking.fits.get(entry.id)
        moved = [
            {"id": e.id, "rank": rank, "fit": round(ranking.fits[e.id].score, 3), "learned": learned}
            for rank, e in enumerate(ranking.ranked, start=1)
            if e.id in ranking.fits and (learned := ranking.fits[e.id].as_dict()["learned"])
        ]
        if moved:
            signals["moved_by_ratings"] = moved[:MOVED_BY_RATINGS_SHOWN]
        pick_reason = (
            f"{reason} → catalog tier='{tier}'"
            + (f", task='{task}'" if task else "")
            + f", optimize_for='{effective_optimize_for}'"
            + f" → '{entry.id}' ({entry.provider})"
            + (f", {describe_fit(winner_fit)}" if winner_fit else "")
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
            tier_scores=tier_scores,
            tier_scores_kind=tier_scores_kind,
            candidates=tuple(
                {
                    "id": e.id,
                    "provider": e.provider,
                    "tier": e.tier,
                    "cost": e.cost,
                    "capability": e.capability,
                    "capability_source": e.capability_source,
                    **(ranking.fits[e.id].as_dict() if e.id in ranking.fits else {}),
                }
                for e in ranking.ranked[:SHORTLIST_SIZE]
            ),
            pool_size=ranking.pool_size,
            signals=signals,
        )

    model_name = tiers.model_for(tier, providers[0] if providers else None)
    return Decision(
        tier=tier,
        model_name=model_name,
        confidence=confidence,
        reason=reason,
        rule_outcome=rule_outcome,
        classifier_outcome=classifier_outcome,
        fallback_outcome=fallback_outcome,
        optimize_for=effective_optimize_for,
        tier_scores=tier_scores,
        tier_scores_kind=tier_scores_kind,
        signals=signals,
    )
