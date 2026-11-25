"""How well each candidate model fits what the request needs.

    fit = Σ_skill need × model skill  +  q × capability  −  λ × price

`need` comes from skills.request_needs (0–1 per skill). The model's skills come from the catalog,
each scaled to 0–1 across the whole catalog:

- video / audio: supports_video / supports_audio (declared, 1 or 0)
- hindi: the multilingual index
- reasoning / writing: capability ÷ the top capability in the catalog, or the human rating for
  this request's quality value when one exists (1–5 → 0–1)
- long_input: context window on a log scale from 8K (0) to 1M tokens (1)
- fast: cheapness, 1 − price

Price is on a fixed log scale, PRICE_FLOOR (0) to PRICE_CEILING (1) per 1K input tokens and
rising past it, so a price gap weighs the same in a two-model task pool as in the whole catalog.

Estimated values (fit on price and release date, not published) count at ESTIMATE_TRUST.
`q` and `λ` come from optimize_for: cost leans on price, accuracy on capability.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from gradist.skills import SKILLS

if TYPE_CHECKING:
    from gradist.catalog import CatalogEntry

MODE_WEIGHTS: dict[str, tuple[float, float]] = {
    "cost": (0.1, 1.0),
    "balanced": (0.6, 0.6),
    "accuracy": (1.0, 0.2),
}
"""optimize_for → (weight on general capability, weight on price)."""

ESTIMATE_TRUST = 0.85
PRICE_FLOOR = 1e-5
PRICE_CEILING = 0.1
CONTEXT_FLOOR = 8_000
CONTEXT_CEILING = 1_000_000


@dataclass(frozen=True)
class CatalogScale:
    top_capability: float

    @classmethod
    def of(cls, entries: Iterable[CatalogEntry]) -> CatalogScale:
        caps = [e.capability for e in entries if e.capability is not None]
        return cls(top_capability=max(caps, default=1.0) or 1.0)

    @staticmethod
    def price(cost: float) -> float:
        """0 at or below PRICE_FLOOR (free), 1 at PRICE_CEILING, still rising (log) above it."""
        span = math.log10(PRICE_CEILING) - math.log10(PRICE_FLOOR)
        return (math.log10(max(cost, PRICE_FLOOR)) - math.log10(PRICE_FLOOR)) / span


@dataclass(frozen=True)
class SkillValue:
    value: float
    source: str
    """"declared", "measured", "estimated", "rating", "price", "context" or "unknown"."""

    @property
    def trusted(self) -> float:
        return self.value * ESTIMATE_TRUST if self.source == "estimated" else self.value


def model_skills(entry: CatalogEntry, scale: CatalogScale, rating: float | None = None) -> dict[str, SkillValue]:
    if rating is not None:
        quality = SkillValue((rating - 1.0) / 4.0, "rating")
    elif entry.capability is not None:
        quality = SkillValue(min(entry.capability / scale.top_capability, 1.0), entry.capability_source or "measured")
    else:
        quality = SkillValue(0.0, "unknown")
    if entry.context_length:
        span = math.log10(CONTEXT_CEILING) - math.log10(CONTEXT_FLOOR)
        context = (math.log10(entry.context_length) - math.log10(CONTEXT_FLOOR)) / span
        long_input = SkillValue(min(max(context, 0.0), 1.0), "context")
    else:
        long_input = SkillValue(0.0, "unknown")
    hindi = (
        SkillValue(entry.multilingual, entry.multilingual_source or "measured")
        if entry.multilingual is not None
        else SkillValue(0.0, "unknown")
    )
    return {
        "video": SkillValue(1.0 if entry.supports_video else 0.0, "declared"),
        "audio": SkillValue(1.0 if entry.supports_audio else 0.0, "declared"),
        "hindi": hindi,
        "reasoning": quality,
        "long_input": long_input,
        "writing": quality,
        "fast": SkillValue(max(1.0 - scale.price(entry.cost), 0.0), "price"),
    }


@dataclass(frozen=True)
class Fit:
    score: float
    parts: dict[str, float]
    """What each term added: one entry per needed skill, plus "capability" and "price" (negative)."""
    skills: dict[str, SkillValue]

    def as_dict(self) -> dict[str, object]:
        return {
            "fit": round(self.score, 3),
            "fit_parts": {k: round(v, 3) for k, v in self.parts.items() if abs(v) >= 0.0005},
            "skill_sources": {k: v.source for k, v in self.skills.items()},
        }


def fit(
    entry: CatalogEntry,
    needs: Mapping[str, float],
    scale: CatalogScale,
    mode: str,
    rating: float | None = None,
) -> Fit:
    q, lam = MODE_WEIGHTS[mode]
    skills = model_skills(entry, scale, rating)
    parts = {s: needs[s] * skills[s].trusted for s in SKILLS if needs.get(s, 0.0) > 0}
    parts["capability"] = q * skills["reasoning"].trusted
    parts["price"] = -lam * scale.price(entry.cost)
    return Fit(score=sum(parts.values()), parts=parts, skills=skills)


def rank_by_fit(
    candidates: list[CatalogEntry],
    needs: Mapping[str, float],
    scale: CatalogScale,
    mode: str,
    quality: Mapping[tuple[str, str], float] | None = None,
    quality_value: str | None = None,
) -> tuple[list[CatalogEntry], dict[str, Fit], str]:
    """Candidates best fit first (cost, then id, break ties), their fits, and a note on how."""
    fits = {
        e.id: fit(e, needs, scale, mode, quality.get((e.id, quality_value)) if quality and quality_value else None)
        for e in candidates
    }
    ranked = sorted(candidates, key=lambda e: (-fits[e.id].score, e.cost, e.id))
    q, lam = MODE_WEIGHTS[mode]
    needed = [s for s in SKILLS if needs.get(s, 0.0) > 0]
    note = (
        f"ranked by fit to the request: Σ need × skill over {', '.join(needed) or 'no specific skills'}"
        f" + {q:g} × capability − {lam:g} × price ({mode})"
    )
    return ranked, fits, note


def describe_fit(entry_fit: Fit, top: int = 3) -> str:
    """'fit 1.52 (video +1.00, hindi +0.79, price -0.40)' — the biggest terms, for the reason."""
    gains = sorted(((k, v) for k, v in entry_fit.parts.items() if k != "price" and v > 0), key=lambda p: -p[1])[:top]
    text = ", ".join(f"{k} +{v:.2f}" for k, v in gains)
    price = entry_fit.parts.get("price", 0.0)
    if price:
        text = f"{text}, price {price:.2f}" if text else f"price {price:.2f}"
    return f"fit {entry_fit.score:.2f} ({text})"
