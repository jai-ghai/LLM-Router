"""What a request needs from a model, per skill, on a 0–1 scale.

Two sources, combined by taking the larger value per skill:
- the task's profile in config/task_skills.yaml (operator-editable);
- signals in the request itself: modality, Devanagari in the text or on-screen text, channel
  type, input length and the QC-risk score.

Every non-zero need carries the reasons behind it, so a decision can explain why a model fits.
The skill names match the catalog's model profile: video and audio map to supports_video /
supports_audio, hindi to multilingual, reasoning and writing to capability, long_input to
context_length, and fast to cost.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from gradist.rules import RuleError

SKILLS = ("video", "audio", "hindi", "reasoning", "long_input", "writing", "fast")

SPEECH_CHANNEL_TYPES = {"news", "sports"}
HINDI_LANGUAGE_VALUES = {"hi", "hin", "hindi", "hinglish"}
LONG_INPUT_TOKENS = 32_000
SHORT_REQUEST_WORDS = 150
WORDY_REQUEST_WORDS = 400


@dataclass(frozen=True)
class SkillNeeds:
    needs: dict[str, float]
    reasons: dict[str, list[str]] = field(default_factory=dict)
    task_profile: str | None = None

    def as_signal(self) -> dict[str, object]:
        return {
            "needs": {s: round(self.needs.get(s, 0.0), 2) for s in SKILLS},
            "reasons": {s: r for s, r in self.reasons.items() if r},
            "task_profile": self.task_profile,
        }


def load_task_skills(path: str | Path) -> dict[str, dict[str, float]]:
    """Read `tasks: {task: {skill: 0–1}}`. Unknown skills or out-of-range values are errors."""
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise RuleError(f"cannot read task skills file {path}: {exc}") from exc
    tasks = data.get("tasks") if isinstance(data, dict) else None
    if not isinstance(tasks, dict):
        raise RuleError(f"{path}: expected a mapping with a 'tasks' mapping")
    profiles: dict[str, dict[str, float]] = {}
    for task, needs in tasks.items():
        if not isinstance(needs, dict):
            raise RuleError(f"{path}: task '{task}' must map skills to numbers")
        profile: dict[str, float] = {}
        for skill, value in needs.items():
            if skill not in SKILLS:
                raise RuleError(f"{path}: task '{task}' has unknown skill '{skill}' (known: {', '.join(SKILLS)})")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise RuleError(f"{path}: task '{task}' skill '{skill}' is not a number") from exc
            if not 0.0 <= number <= 1.0:
                raise RuleError(f"{path}: task '{task}' skill '{skill}' must be between 0 and 1")
            profile[skill] = number
        profiles[str(task)] = profile
    return profiles


def devanagari_share(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if "\u0900" <= ch <= "\u097f") / len(letters)


def request_needs(
    metadata: Mapping[str, str],
    text: str,
    *,
    screen_text: str = "",
    has_image: bool = False,
    qc_risk: float | None = None,
    task_skills: Mapping[str, Mapping[str, float]] | None = None,
) -> SkillNeeds:
    needs = dict.fromkeys(SKILLS, 0.0)
    reasons: dict[str, list[str]] = {s: [] for s in SKILLS}

    def need(skill: str, value: float, why: str) -> None:
        if value <= 0:
            return
        needs[skill] = max(needs[skill], min(value, 1.0))
        reasons[skill].append(f"{why} ({value:.1f})")

    task = (metadata.get("task") or "").strip()
    profile = (task_skills or {}).get(task) if task else None
    for skill, value in (profile or {}).items():
        need(skill, value, f"task '{task}'")

    modality = (metadata.get("modality") or "").strip().lower()
    channel_type = (metadata.get("channel_type") or metadata.get("pipeline") or "").strip().lower()
    if modality == "video":
        need("video", 1.0, "modality=video")
        if channel_type in SPEECH_CHANNEL_TYPES:
            need("audio", 0.7, f"{channel_type} clip: the speech carries the story")
    elif has_image:
        need("video", 0.3, "a still image")

    if (metadata.get("language") or "").strip().lower() in HINDI_LANGUAGE_VALUES:
        need("hindi", 0.9, f"language={metadata['language']}")
    for label, source in (("text", text), ("on-screen text", screen_text)):
        share = devanagari_share(source)
        if share >= 0.2:
            need("hindi", 1.0, f"{share:.0%} of the {label} is Devanagari")
        elif share > 0:
            need("hindi", 0.6, f"some Devanagari in the {label}")

    if qc_risk is not None:
        need("reasoning", qc_risk, "QC-risk score: sub-stories like this often need fixes")
    words = len(text.split())
    if words >= WORDY_REQUEST_WORDS:
        need("reasoning", 0.5, f"{words} words of input")

    tokens = (len(text) + len(screen_text)) / 4
    if tokens >= LONG_INPUT_TOKENS / 8:
        need("long_input", tokens / LONG_INPUT_TOKENS, f"about {tokens:,.0f} tokens of input")

    if modality != "video" and not has_image and 0 < words < SHORT_REQUEST_WORDS:
        need("fast", 0.5, f"short text-only request ({words} words)")

    return SkillNeeds(needs=needs, reasons=reasons, task_profile=task if profile is not None else None)
