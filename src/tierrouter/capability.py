"""Live model capability facts from OpenRouter's public model list — cost, context length, and
image-support, for any model OpenRouter knows about, across providers.

This is metadata only: fetching a models LIST is one GET request against a public catalog page,
not "calling a model" — ARCHITECTURE.md §1's "no model is ever called" is about an inference
request (sending a prompt, getting a completion), which this never does. It is also never called
from `resolve()` or anywhere on a request's hot path — only from the offline refresh script
(`experiments/refresh_catalog.py`), so a request never pays for, or depends on, a network call to
openrouter.ai. If that script is never run, nothing here is ever touched.

What this does NOT give you: which capability TIER (small/medium/large) a model belongs to for
YOUR use case, or whether it is actually any good at YOUR tasks. Both of those stay a human
judgment — see catalog.py (`tier` is set by hand in catalog.yaml) and quality.py (a human rates
real output). A public models list can tell you facts; it cannot tell you what you need.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


class CapabilityError(RuntimeError):
    """The capability source could not be reached or did not return the expected shape."""


@dataclass(frozen=True)
class ModelCapability:
    id: str
    """OpenRouter's own id, e.g. "google/gemini-3-flash-preview" — namespaced by provider,
    unlike catalog.yaml's bare ids (e.g. "gemini-3-flash-preview"). See `match_by_suffix`."""
    provider: str
    context_length: int | None
    supports_image: bool
    cost_per_1k_prompt: float | None
    cost_per_1k_completion: float | None


def fetch_openrouter_capabilities(timeout: float = 10.0) -> list[ModelCapability]:
    try:
        with urllib.request.urlopen(OPENROUTER_MODELS_URL, timeout=timeout) as response:
            data = json.load(response)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise CapabilityError(f"could not reach {OPENROUTER_MODELS_URL}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CapabilityError(f"{OPENROUTER_MODELS_URL} did not return valid JSON: {exc}") from exc
    entries = data.get("data") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise CapabilityError(f"{OPENROUTER_MODELS_URL}: expected a mapping with a 'data' list")
    return [_parse_entry(e) for e in entries if isinstance(e, dict) and e.get("id")]


def _parse_entry(entry: dict[str, object]) -> ModelCapability:
    model_id = str(entry["id"])
    provider = model_id.split("/", 1)[0] if "/" in model_id else ""
    architecture = entry.get("architecture")
    architecture = architecture if isinstance(architecture, dict) else {}
    input_modalities = architecture.get("input_modalities") or []
    pricing = entry.get("pricing")
    pricing = pricing if isinstance(pricing, dict) else {}
    return ModelCapability(
        id=model_id,
        provider=provider,
        context_length=_as_int(entry.get("context_length")),
        supports_image="image" in input_modalities,
        cost_per_1k_prompt=_as_cost(pricing.get("prompt")),
        cost_per_1k_completion=_as_cost(pricing.get("completion")),
    )


def _as_int(value: object) -> int | None:
    if not isinstance(value, (int, float, str)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_cost(value: object) -> float | None:
    """OpenRouter prices per single token, in USD; catalog.yaml's `cost` is "per 1k", so the
    raw value is scaled here rather than at every call site."""
    if not isinstance(value, (int, float, str)):
        return None
    try:
        return float(value) * 1000
    except (TypeError, ValueError):
        return None


def match_by_suffix(catalog_id: str, capabilities: list[ModelCapability]) -> ModelCapability | None:
    """`catalog_id` (e.g. "gemini-3.5-flash", no provider namespace) against OpenRouter's ids
    (e.g. "google/gemini-3.5-flash"): an exact match first, else a case-insensitive match on the
    part after the provider's "/". Returns None rather than guessing when nothing matches —
    the refresh script reports that case rather than silently skipping it.
    """
    lowered = catalog_id.strip().lower()
    for cap in capabilities:
        if cap.id.lower() == lowered:
            return cap
    for cap in capabilities:
        if cap.id.split("/", 1)[-1].lower() == lowered:
            return cap
    return None
