"""What is in a video — read once per file, so the router can tell one clip from another.

Request text and metadata say what the job is; they say nothing about the clip itself. A video
profile adds that, from local tools only (no model API is called):

- audio: how much of the clip is speech (Silero VAD), the spoken language (Whisper), and how much
  sounds like music, singing, laughter or a crowd (an AudioSet tagger);
- picture: scene changes, black and frozen stretches (OpenCV, a frame every SAMPLE_EVERY_S);
- screen: how much text the frames carry (EasyOCR's detector, when OCR is on);
- a content type read off those signals by `classify_content`.

skills.request_needs turns a profile into needs and difficulty. Profiles are cached by the
file's SHA-256, so five requests about one clip read it once.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PROFILE_VERSION = 1
SAMPLE_RATE = 16_000
SAMPLE_EVERY_S = 2.0
LANGUAGE_WINDOWS = 3
LANGUAGE_WINDOW_S = 30
SOUND_WINDOWS = 6
SOUND_WINDOW_S = 10
MIN_LANGUAGE_CONFIDENCE = 0.5
SCENE_CHANGE = 0.5
"""Histogram correlation between samples below this counts as a scene change."""
BLACK_LEVEL = 16
FROZEN_DIFF = 0.8
TEXT_FRAMES = (0.1, 0.35, 0.6, 0.85)
WHISPER_SIZE = "small"
SOUND_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"

SOUND_GROUPS: dict[str, frozenset[str]] = {
    "speech": frozenset({
        "Speech", "Male speech, man speaking", "Female speech, woman speaking", "Child speech, kid speaking",
        "Conversation", "Narration, monologue", "Speech synthesizer",
    }),
    "music": frozenset({
        "Music", "Background music", "Pop music", "Music of Asia", "Music of Bollywood", "Carnatic music",
        "Theme music", "Soundtrack music", "Musical instrument", "Television", "Jingle (music)",
    }),
    "singing": frozenset({"Singing", "Choir", "Song", "Vocal music", "Male singing", "Female singing", "Chant"}),
    "laughter": frozenset({"Laughter", "Giggle", "Chuckle, chortle", "Belly laugh", "Snicker", "Baby laughter"}),
    "crowd": frozenset({"Crowd", "Chatter", "Hubbub, speech noise, speech babble", "Cheering", "Applause", "Clapping"}),
}
CONTENT_TYPES = ("news", "music", "reality", "talk", "drama", "ads", "unknown")


@dataclass(frozen=True)
class VideoProfile:
    id: str
    duration_s: float
    has_audio: bool
    speech_share: float | None = None
    speech_turns_per_min: float | None = None
    language: str | None = None
    """Whisper language code (e.g. "ta"), None when unsure or silent."""
    language_confidence: float | None = None
    sounds: dict[str, float] = field(default_factory=dict)
    """SOUND_GROUPS → mean share across the sampled windows (0–1)."""
    scene_changes_per_min: float | None = None
    black_share: float | None = None
    frozen_share: float | None = None
    text_boxes_per_frame: float | None = None
    content_type: str = "unknown"
    content_reasons: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    version: int = PROFILE_VERSION

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoProfile:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def classify_content(
    *,
    has_audio: bool,
    speech_share: float | None,
    sounds: dict[str, float],
    scene_changes_per_min: float | None,
    text_boxes_per_frame: float | None,
) -> tuple[str, list[str]]:
    """The content type these signals point to, and why. First matching rule wins."""
    speech = speech_share or 0.0
    music = sounds.get("music", 0.0)
    singing = sounds.get("singing", 0.0)
    lively = sounds.get("laughter", 0.0) + sounds.get("crowd", 0.0)
    cuts = scene_changes_per_min or 0.0
    text = text_boxes_per_frame
    if not has_audio:
        return "unknown", ["no audio track"]
    if singing >= 0.15 or (music >= 0.5 and speech < 0.35):
        return "music", [f"singing {singing:.0%}, music {music:.0%}, speech {speech:.0%}"]
    if lively >= 0.08:
        return "reality", [f"laughter and crowd {lively:.0%}: several people at once"]
    if text is not None and text >= 8 and speech >= 0.4:
        return "news", [f"{text:.0f} text boxes per frame with speech {speech:.0%}: tickers over a voice"]
    if cuts >= 12 and music >= 0.3:
        return "ads", [f"{cuts:.0f} scene changes a minute over music {music:.0%}"]
    if speech >= 0.5 and cuts < 5.5 and music < 0.5:
        return "talk", [f"speech {speech:.0%}, only {cuts:.1f} scene changes a minute"]
    if speech >= 0.3:
        return "drama", [f"speech {speech:.0%} with music {music:.0%}"]
    return "unknown", [f"speech {speech:.0%}, music {music:.0%}: no clear pattern"]


def group_sounds(windows: Sequence[Sequence[tuple[str, float]]]) -> dict[str, float]:
    """Per window, each group takes its best label score; groups are then averaged over windows."""
    if not windows:
        return {}
    totals = dict.fromkeys(SOUND_GROUPS, 0.0)
    for labels in windows:
        for group, names in SOUND_GROUPS.items():
            totals[group] += max((score for label, score in labels if label in names), default=0.0)
    return {g: round(v / len(windows), 3) for g, v in totals.items()}


def picture_stats(samples: Sequence[np.ndarray], every_s: float) -> dict[str, float]:
    """Scene changes a minute and black / frozen shares from grey frames sampled `every_s` apart."""
    import cv2

    if len(samples) < 2:
        return {"scene_changes_per_min": 0.0, "black_share": 0.0, "frozen_share": 0.0}
    hists = []
    for grey in samples:
        hist = cv2.calcHist([grey], [0], None, [32], [0, 256])
        hists.append(cv2.normalize(hist, hist).flatten())
    changes = sum(
        1 for a, b in itertools.pairwise(hists) if cv2.compareHist(a, b, cv2.HISTCMP_CORREL) < SCENE_CHANGE
    )
    frozen = sum(
        1
        for a, b in itertools.pairwise(samples)
        if float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)))) < FROZEN_DIFF
    )
    black = sum(1 for grey in samples if float(np.mean(grey)) < BLACK_LEVEL)
    minutes = len(samples) * every_s / 60
    return {
        "scene_changes_per_min": round(changes / minutes, 2),
        "black_share": round(black / len(samples), 3),
        "frozen_share": round(frozen / (len(samples) - 1), 3),
    }


LanguageDetector = Callable[[np.ndarray], tuple[str, float, list[tuple[str, float]]]]
SoundTagger = Callable[[np.ndarray], list[tuple[str, float]]]
TextCounter = Callable[[bytes], int]


class VideoProfiler:
    """Builds and caches VideoProfiles. Whisper and the sound tagger load on first use; tests pass
    stand-ins through `language_detector` / `sound_tagger`."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        language_detector: LanguageDetector | None = None,
        sound_tagger: SoundTagger | None = None,
        text_counter: TextCounter | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self._language_detector = language_detector
        self._sound_tagger = sound_tagger
        self.text_counter = text_counter
        self._lock = threading.Lock()

    def load(self, profile_id: str) -> VideoProfile | None:
        path = self._path(profile_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return VideoProfile.from_dict(data) if data.get("version") == PROFILE_VERSION else None

    def profile(self, video: Path) -> VideoProfile:
        """The cached profile for this file's contents, building it when there is none."""
        profile_id = file_id(video)
        cached = self.load(profile_id)
        if cached is not None:
            return cached
        with self._lock:
            cached = self.load(profile_id)
            if cached is not None:
                return cached
            built = self._build(video, profile_id)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._path(profile_id).write_text(json.dumps(built.as_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
            return built

    def _path(self, profile_id: str) -> Path:
        if not profile_id.isalnum():
            raise ValueError(f"not a profile id: {profile_id!r}")
        return self.cache_dir / f"{profile_id}.json"

    def _build(self, video: Path, profile_id: str) -> VideoProfile:
        timings: dict[str, float] = {}

        def timed(name: str, started: float) -> None:
            timings[name] = round(time.time() - started, 2)

        started = time.time()
        audio = read_audio(video)
        timed("audio", started)
        picture_started = time.time()
        greys, stills, duration = sample_frames(video, SAMPLE_EVERY_S, TEXT_FRAMES)
        picture = picture_stats(greys, SAMPLE_EVERY_S)
        timed("picture", picture_started)
        if audio is not None and len(audio):
            duration = max(duration, len(audio) / SAMPLE_RATE)

        speech_share = turns = language = confidence = None
        sounds: dict[str, float] = {}
        if audio is not None and len(audio) >= SAMPLE_RATE:
            step = time.time()
            segments = speech_segments(audio)
            speech_share = round(sum(e - s for s, e in segments) / len(audio), 3)
            turns = round(len(segments) / max(len(audio) / SAMPLE_RATE / 60, 1e-6), 2)
            timed("speech", step)
            step = time.time()
            language, confidence = self._language(audio, segments)
            timed("language", step)
            step = time.time()
            sounds = group_sounds([self._tagger()(w) for w in spread_windows(audio, SOUND_WINDOWS, SOUND_WINDOW_S)])
            timed("sounds", step)

        text_boxes = None
        if self.text_counter is not None and stills:
            step = time.time()
            text_boxes = round(sum(self.text_counter(s) for s in stills) / len(stills), 2)
            timed("screen", step)

        content, reasons = classify_content(
            has_audio=audio is not None,
            speech_share=speech_share,
            sounds=sounds,
            scene_changes_per_min=picture["scene_changes_per_min"],
            text_boxes_per_frame=text_boxes,
        )
        return VideoProfile(
            id=profile_id,
            duration_s=round(duration, 2),
            has_audio=audio is not None,
            speech_share=speech_share,
            speech_turns_per_min=turns,
            language=language,
            language_confidence=confidence,
            sounds=sounds,
            scene_changes_per_min=picture["scene_changes_per_min"],
            black_share=picture["black_share"],
            frozen_share=picture["frozen_share"],
            text_boxes_per_frame=text_boxes,
            content_type=content,
            content_reasons=reasons,
            timings=timings,
        )

    def _language(self, audio: np.ndarray, segments: list[tuple[int, int]]) -> tuple[str | None, float | None]:
        """Whisper on LANGUAGE_WINDOWS stretches of speech; probabilities averaged across them."""
        speech = np.concatenate([audio[s:e] for s, e in segments]) if segments else audio
        if len(speech) < 5 * SAMPLE_RATE:
            return None, None
        totals: dict[str, float] = {}
        windows = spread_windows(speech, LANGUAGE_WINDOWS, LANGUAGE_WINDOW_S)
        for window in windows:
            _, _, probs = self._detector()(window)
            for code, p in probs:
                totals[code] = totals.get(code, 0.0) + p / len(windows)
        best = max(totals, key=lambda c: totals[c])
        confidence = round(totals[best], 3)
        return (best if confidence >= MIN_LANGUAGE_CONFIDENCE else None), confidence

    def _detector(self) -> LanguageDetector:
        if self._language_detector is None:
            from faster_whisper import WhisperModel

            model = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
            self._language_detector = model.detect_language
        return self._language_detector

    def _tagger(self) -> SoundTagger:
        if self._sound_tagger is None:
            from transformers import pipeline

            classify = pipeline("audio-classification", model=SOUND_MODEL, device=-1)

            def tag(window: np.ndarray) -> list[tuple[str, float]]:
                out = classify({"raw": window.astype(np.float32), "sampling_rate": SAMPLE_RATE}, top_k=10)
                return [(str(o["label"]), float(o["score"])) for o in out]

            self._sound_tagger = tag
        return self._sound_tagger


def file_id(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_audio(video: Path) -> np.ndarray | None:
    """Mono 16 kHz float32, or None when the file has no audio track."""
    import av

    with av.open(str(video)) as container:
        if not container.streams.audio:
            return None
    from faster_whisper.audio import decode_audio

    return np.asarray(decode_audio(str(video), sampling_rate=SAMPLE_RATE), dtype=np.float32)


def speech_segments(audio: np.ndarray) -> list[tuple[int, int]]:
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    return [(s["start"], s["end"]) for s in get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=300))]


def spread_windows(audio: np.ndarray, count: int, seconds: int) -> list[np.ndarray]:
    """`count` windows of `seconds`, centred at evenly spaced points; fewer when the audio is short."""
    size = seconds * SAMPLE_RATE
    if len(audio) <= size:
        return [audio]
    centres = [int(len(audio) * (i + 0.5) / count) for i in range(count)]
    starts = sorted({min(max(c - size // 2, 0), len(audio) - size) for c in centres})
    return [audio[s : s + size] for s in starts]


def sample_frames(video: Path, every_s: float, still_ratios: Sequence[float]) -> tuple[list[np.ndarray], list[bytes], float]:
    """Small grey frames every `every_s`, JPEG stills at `still_ratios` of the clip, and its length."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total / fps if fps else 0.0
        greys: list[np.ndarray] = []
        step = max(round(fps * every_s), 1)
        for index in range(0, max(total, 1), step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                break
            greys.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36)))
        stills: list[bytes] = []
        for ratio in still_ratios:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * ratio))
            ok, frame = cap.read()
            if ok:
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    stills.append(buf.tobytes())
        return greys, stills, duration
    finally:
        cap.release()
