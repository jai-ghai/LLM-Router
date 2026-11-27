"""What a request needs from a model, per skill, on a 0–1 scale.

Three sources, combined by taking the larger value per skill:
- the task's profile in config/task_skills.yaml (operator-editable);
- signals in the request itself: modality, the language metadata, Indian scripts in the text or
  on-screen text, channel type, input length and the QC-risk score;
- for a video, its profile (video_profile.py): speech share, spoken language, length, content type.

Every non-zero need carries the reasons behind it, so a decision can explain why a model fits.

`difficulty` (0–1) is how hard this particular request looks, separate from which skills it needs:
the largest of the QC-risk score, input length, how many figures and conditions the request text
holds (never the on-screen text: clocks and channel codes are not figures to keep straight), a
low-resource language and the clip's content type; a low score for an easy clip or a short
text-only request when nothing harder applies. None when no signal applies (neutral).
The skill names match the catalog's model profile: video and audio map to supports_video /
supports_audio, hindi to multilingual, reasoning and writing to capability, long_input to
context_length, and fast to cost.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from gradist.rules import RuleError

if TYPE_CHECKING:
    from gradist.video_profile import VideoProfile

SKILLS = ("video", "audio", "hindi", "reasoning", "long_input", "writing", "fast")

SPEECH_CHANNEL_TYPES = {"news", "sports"}
DIALOGUE_CHANNEL_TYPES = {"entertainment", "gec", "drama", "reality"}
INDIAN_LANGUAGES: dict[str, tuple[str, bool]] = {
    "hi": ("Hindi", False),
    "bn": ("Bengali", False),
    "mr": ("Marathi", False),
    "ta": ("Tamil", False),
    "te": ("Telugu", False),
    "gu": ("Gujarati", False),
    "kn": ("Kannada", False),
    "ml": ("Malayalam", False),
    "ur": ("Urdu", False),
    "pa": ("Punjabi", True),
    "or": ("Odia", True),
    "as": ("Assamese", True),
    "ne": ("Nepali", True),
}
"""Language code → (name, low-resource). Low-resource languages make a request harder: speech
recognition and spelling in them vary far more between models than Hindi does."""
LANGUAGE_ALIASES = {
    **{code: code for code in INDIAN_LANGUAGES},
    **{name.lower(): code for code, (name, _) in INDIAN_LANGUAGES.items()},
    "hin": "hi",
    "hinglish": "hi",
    "bangla": "bn",
    "oriya": "or",
    "odiya": "or",
    "asamiya": "as",
    "panjabi": "pa",
    "gurmukhi": "pa",
}
INDIAN_SCRIPTS = {
    "Devanagari": ("\u0900", "\u097f"),
    "Bengali-Assamese": ("\u0980", "\u09ff"),
    "Gurmukhi": ("\u0a00", "\u0a7f"),
    "Gujarati": ("\u0a80", "\u0aff"),
    "Odia": ("\u0b00", "\u0b7f"),
    "Tamil": ("\u0b80", "\u0bff"),
    "Telugu": ("\u0c00", "\u0c7f"),
    "Kannada": ("\u0c80", "\u0cff"),
    "Malayalam": ("\u0d00", "\u0d7f"),
}
LOW_RESOURCE_DIFFICULTY = 0.6
SPEECH_FOR_AUDIO = 0.15
"""A clip with at least this share of speech needs a model that hears it."""
SINGING_FOR_AUDIO = 0.15
HARD_CONTENT = {
    "music": (0.8, "sung lyrics and music over the voice"),
    "reality": (0.7, "several people talking over each other"),
    "news": (0.5, "dense on-screen text over speech"),
    "ads": (0.4, "fast cuts between short spots"),
}
EASY_CONTENT = {
    "talk": (0.3, "one or two people speaking, few cuts"),
    "drama": (0.35, "scripted dialogue, one scene at a time"),
}
"""video_profile content type → (difficulty, why). Easy ones apply only when nothing harder does."""
VIDEO_TOKENS_PER_SECOND = 300
"""Frames at 1 fps plus the audio track, as multimodal APIs count a video (≈ 260 + 32 per second)."""
LONG_INPUT_TOKENS = 32_000
SHORT_REQUEST_WORDS = 150
WORDY_REQUEST_WORDS = 400
EASY_DIFFICULTY = 0.25
MANY_FIGURES = 5
MANY_CONDITIONS = 3
FIGURE = re.compile(r"\d[\d,.:/%]*")
CONDITION = re.compile(
    r"\b(?:if|unless|except|only|but|however|must not|do not)\b|अगर|यदि|लेकिन|परंतु|सिवाय|केवल|नहीं",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SkillNeeds:
    needs: dict[str, float]
    reasons: dict[str, list[str]] = field(default_factory=dict)
    task_profile: str | None = None
    input_tokens: int = 0
    """Rough size of text + on-screen text (characters ÷ 4); models with a smaller window are dropped."""
    difficulty: float | None = None
    difficulty_reasons: list[str] = field(default_factory=list)

    def as_signal(self) -> dict[str, object]:
        return {
            "needs": {s: round(self.needs.get(s, 0.0), 2) for s in SKILLS},
            "reasons": {s: r for s, r in self.reasons.items() if r},
            "task_profile": self.task_profile,
            "input_tokens": self.input_tokens,
            "difficulty": None if self.difficulty is None else round(self.difficulty, 2),
            "difficulty_reasons": self.difficulty_reasons,
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
    return script_shares(text).get("Devanagari", 0.0)


def script_shares(text: str) -> dict[str, float]:
    """Share of the letters in each Indian script (INDIAN_SCRIPTS); scripts not present are omitted."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return {}
    counts = {
        name: sum(1 for ch in letters if low <= ch <= high) for name, (low, high) in INDIAN_SCRIPTS.items()
    }
    return {name: n / len(letters) for name, n in counts.items() if n}


def indian_language(value: str) -> str | None:
    """'Tamil', 'ta', 'oriya' … → the INDIAN_LANGUAGES code; None for anything else."""
    return LANGUAGE_ALIASES.get(value.strip().lower())


def request_needs(
    metadata: Mapping[str, str],
    text: str,
    *,
    screen_text: str = "",
    has_image: bool = False,
    qc_risk: float | None = None,
    task_skills: Mapping[str, Mapping[str, float]] | None = None,
    video: VideoProfile | None = None,
) -> SkillNeeds:
    """`video` (video_profile.VideoProfile) adds what is in the clip: how much of it is speech,
    the spoken language, its length and how hard its content type is."""
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
        elif channel_type in DIALOGUE_CHANNEL_TYPES:
            need("audio", 0.6, f"{channel_type} clip: the dialogue carries the story")
    elif has_image:
        need("video", 0.3, "a still image")

    extra_difficulty: list[tuple[float, str]] = []
    language = indian_language(metadata.get("language") or "")
    if language is not None:
        name, low_resource = INDIAN_LANGUAGES[language]
        need("hindi", 0.9 if language == "hi" else 1.0, f"language={metadata['language']}")
        if low_resource:
            extra_difficulty.append((LOW_RESOURCE_DIFFICULTY, f"{name} is a low-resource language"))
    for label, source in (("text", text), ("on-screen text", screen_text)):
        shares = script_shares(source)
        if not shares:
            continue
        script, share = max(shares.items(), key=lambda kv: kv[1])
        total = sum(shares.values())
        if total >= 0.2:
            need("hindi", 1.0, f"{share:.0%} of the {label} is {script}")
        else:
            need("hindi", 0.6, f"some {script} in the {label}")

    easy_difficulty: list[tuple[float, str]] = []
    if video is not None and modality == "video":
        if not video.has_audio:
            needs["audio"] = 0.0
            reasons["audio"] = ["the clip has no audio track"]
        else:
            speech = video.speech_share or 0.0
            if speech >= SPEECH_FOR_AUDIO:
                need("audio", 0.5 + 0.5 * speech, f"speech in {speech:.0%} of the clip")
            if video.sounds.get("singing", 0.0) >= SINGING_FOR_AUDIO:
                need("audio", 1.0, f"singing in {video.sounds['singing']:.0%} of the sampled audio")
        if video.language in INDIAN_LANGUAGES:
            name, low_resource = INDIAN_LANGUAGES[video.language]
            need("hindi", 0.9 if video.language == "hi" else 1.0, f"spoken {name} (Whisper {video.language_confidence:.2f})")
            if low_resource:
                extra_difficulty.append((LOW_RESOURCE_DIFFICULTY, f"spoken {name} is a low-resource language"))
        content = video.content_type
        # On-screen text density alone cannot tell a bulletin from a serial (both run 5–8 boxes
        # a frame), so a news channel's clip read as talk or drama stays news.
        if channel_type == "news" and content in EASY_CONTENT:
            content = "news"
        if content in HARD_CONTENT:
            level, why = HARD_CONTENT[content]
            extra_difficulty.append((level, f"{content} clip: {why}"))
        elif content in EASY_CONTENT:
            level, why = EASY_CONTENT[content]
            easy_difficulty.append((level, f"{content} clip: {why}"))

    duration = _seconds(metadata.get("duration_s")) or (video.duration_s if video is not None else 0.0)
    video_tokens = duration * VIDEO_TOKENS_PER_SECOND if modality == "video" and duration else 0.0

    if qc_risk is not None:
        need("reasoning", qc_risk, "QC-risk score: sub-stories like this often need fixes")
    words = len(text.split())
    if words >= WORDY_REQUEST_WORDS:
        need("reasoning", 0.5, f"{words} words of input")

    tokens = (len(text) + len(screen_text)) / 4 + video_tokens
    if tokens >= LONG_INPUT_TOKENS / 8:
        why = f"{duration / 60:.0f} min of video" if video_tokens else "input"
        need("long_input", tokens / LONG_INPUT_TOKENS, f"about {tokens:,.0f} tokens of {why}")

    short_text_only = modality != "video" and not has_image and 0 < words < SHORT_REQUEST_WORDS
    if short_text_only:
        need("fast", 0.5, f"short text-only request ({words} words)")

    # Only the request's own text: clocks, channel codes and prices read off the screen are not
    # figures the model has to keep straight.
    difficulty, difficulty_reasons = _difficulty(text, words, qc_risk, short_text_only, extra_difficulty, easy_difficulty)
    return SkillNeeds(
        needs=needs,
        reasons=reasons,
        task_profile=task if profile is not None else None,
        input_tokens=round(tokens),
        difficulty=difficulty,
        difficulty_reasons=difficulty_reasons,
    )


def _seconds(value: str | None) -> float:
    try:
        seconds = float(value or 0)
    except ValueError:
        return 0.0
    return seconds if seconds > 0 else 0.0


def _difficulty(
    text: str,
    words: int,
    qc_risk: float | None,
    short_text_only: bool,
    extra: list[tuple[float, str]] | None = None,
    easy: list[tuple[float, str]] | None = None,
) -> tuple[float | None, list[str]]:
    """The largest signal, or — when there is none — the easy one (an easy clip, a short request)."""
    signals: list[tuple[float, str]] = list(extra or [])
    if qc_risk is not None:
        signals.append((qc_risk, "QC-risk score"))
    if words >= WORDY_REQUEST_WORDS:
        signals.append((min(words / (2 * WORDY_REQUEST_WORDS), 1.0), f"{words} words of input"))
    figures = len(FIGURE.findall(text))
    if figures >= MANY_FIGURES:
        signals.append((min(0.4 + 0.05 * figures, 1.0), f"{figures} figures to keep straight"))
    conditions = len(CONDITION.findall(text))
    if conditions >= MANY_CONDITIONS:
        signals.append((min(0.3 + 0.1 * conditions, 1.0), f"{conditions} conditions or exceptions"))
    if not signals:
        signals.extend(easy or [])
    if short_text_only and not signals:
        signals.append((EASY_DIFFICULTY, f"short text-only request ({words} words)"))
    if not signals:
        return None, []
    return max(v for v, _ in signals), [f"{why} ({v:.2f})" for v, why in sorted(signals, reverse=True)]
