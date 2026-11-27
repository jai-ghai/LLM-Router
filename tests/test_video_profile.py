from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gradist import video_profile as vp
from gradist.video_profile import (
    PROFILE_VERSION,
    SAMPLE_RATE,
    VideoProfile,
    VideoProfiler,
    classify_content,
    group_sounds,
    spread_windows,
)


def classify(**signals: object) -> str:
    base: dict[str, object] = {
        "has_audio": True,
        "speech_share": 0.0,
        "sounds": {},
        "scene_changes_per_min": 0.0,
        "text_boxes_per_frame": None,
    }
    return classify_content(**{**base, **signals})[0]  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("signals", "expected"),
    [
        ({"has_audio": False, "speech_share": 0.9}, "unknown"),
        ({"speech_share": 0.3, "sounds": {"singing": 0.2}}, "music"),
        ({"speech_share": 0.2, "sounds": {"music": 0.6}}, "music"),
        ({"speech_share": 0.7, "sounds": {"laughter": 0.05, "crowd": 0.05}}, "reality"),
        ({"speech_share": 0.8, "sounds": {"music": 0.4}, "text_boxes_per_frame": 12.0}, "news"),
        ({"speech_share": 0.4, "sounds": {"music": 0.4}, "scene_changes_per_min": 15.0}, "ads"),
        ({"speech_share": 0.9, "sounds": {"music": 0.05}, "scene_changes_per_min": 2.0}, "talk"),
        ({"speech_share": 0.6, "sounds": {"music": 0.45}, "scene_changes_per_min": 1.3}, "talk"),
        ({"speech_share": 0.7, "sounds": {"music": 0.3}, "scene_changes_per_min": 6.0}, "drama"),
        ({"speech_share": 0.1, "sounds": {"music": 0.2}}, "unknown"),
    ],
)
def test_classify_content_rules(signals: dict[str, object], expected: str) -> None:
    assert classify(**signals) == expected


def test_classify_content_explains_itself() -> None:
    content, reasons = classify_content(
        has_audio=True, speech_share=0.8, sounds={}, scene_changes_per_min=1.0, text_boxes_per_frame=None
    )
    assert content == "talk"
    assert reasons and "speech 80%" in reasons[0]


def test_group_sounds_takes_the_best_label_per_window_then_averages() -> None:
    windows = [
        [("Speech", 0.6), ("Male speech, man speaking", 0.8), ("Music", 0.1)],
        [("Music", 0.5), ("Singing", 0.4)],
    ]
    groups = group_sounds(windows)
    assert groups["speech"] == 0.4
    assert groups["music"] == 0.3
    assert groups["singing"] == 0.2
    assert groups["laughter"] == 0.0
    assert group_sounds([]) == {}


def test_spread_windows_covers_short_and_long_audio() -> None:
    short = np.zeros(SAMPLE_RATE * 5, dtype=np.float32)
    assert [len(w) for w in spread_windows(short, 3, 10)] == [len(short)]
    long = np.zeros(SAMPLE_RATE * 100, dtype=np.float32)
    windows = spread_windows(long, 3, 10)
    assert len(windows) == 3
    assert all(len(w) == SAMPLE_RATE * 10 for w in windows)


def test_picture_stats_counts_cuts_black_and_frozen_frames() -> None:
    pytest.importorskip("cv2")
    dark = np.zeros((32, 32), dtype=np.uint8)
    bright = np.full((32, 32), 200, dtype=np.uint8)
    stats = vp.picture_stats([dark, dark, bright, bright, dark, dark], every_s=2.0)
    assert stats["scene_changes_per_min"] == 10.0
    assert stats["black_share"] == pytest.approx(4 / 6, abs=1e-3)
    assert stats["frozen_share"] == 0.6
    assert vp.picture_stats([dark], every_s=2.0)["scene_changes_per_min"] == 0.0


def test_sample_frames_reads_a_real_video(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("no MJPG encoder in this OpenCV build")
    for i in range(60):
        writer.write(np.full((48, 64, 3), 0 if i < 30 else 220, dtype=np.uint8))
    writer.release()
    greys, stills, duration = vp.sample_frames(path, 2.0, (0.1, 0.9))
    assert duration == pytest.approx(6.0)
    assert len(greys) == 3
    assert len(stills) == 2 and all(s[:2] == b"\xff\xd8" for s in stills)


def test_profile_round_trips_and_ignores_unknown_keys() -> None:
    profile = VideoProfile(id="abc", duration_s=60.0, has_audio=True, language="ta", sounds={"music": 0.2})
    data = {**profile.as_dict(), "added_later": 1}
    assert VideoProfile.from_dict(data) == profile


@pytest.fixture
def stub_decoders(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """30 s of 'audio' whose middle 20 s is speech, and a static picture."""
    calls = {"audio": 0}
    audio = np.zeros(SAMPLE_RATE * 30, dtype=np.float32)

    def read_audio(_: Path) -> np.ndarray:
        calls["audio"] += 1
        return audio

    grey = np.full((16, 16), 120, dtype=np.uint8)
    monkeypatch.setattr(vp, "read_audio", read_audio)
    monkeypatch.setattr(vp, "speech_segments", lambda _a: [(SAMPLE_RATE * 5, SAMPLE_RATE * 25)])
    monkeypatch.setattr(vp, "sample_frames", lambda _p, _e, _t: ([grey] * 15, [b"jpg"] * 4, 30.0))
    monkeypatch.setattr(
        vp,
        "picture_stats",
        lambda _s, _e: {"scene_changes_per_min": 1.0, "black_share": 0.0, "frozen_share": 1.0},
    )
    return calls


def test_profiler_builds_and_caches_a_profile(tmp_path: Path, stub_decoders: dict[str, int]) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")
    profiler = VideoProfiler(
        tmp_path / "profiles",
        language_detector=lambda _w: ("ta", 0.9, [("ta", 0.9), ("hi", 0.1)]),
        sound_tagger=lambda _w: [("Speech", 0.9), ("Music", 0.05)],
        text_counter=lambda _jpg: 3,
    )
    profile = profiler.profile(video)
    assert profile.id == vp.file_id(video)
    assert profile.duration_s == 30.0
    assert profile.speech_share == pytest.approx(0.667, abs=1e-3)
    assert profile.language == "ta"
    assert profile.language_confidence == 0.9
    assert profile.sounds["speech"] == 0.9
    assert profile.text_boxes_per_frame == 3.0
    assert profile.content_type == "talk"

    saved = json.loads((tmp_path / "profiles" / f"{profile.id}.json").read_text(encoding="utf-8"))
    assert saved["version"] == PROFILE_VERSION
    assert profiler.profile(video) == profile
    assert stub_decoders["audio"] == 1
    assert profiler.load(profile.id) == profile


def test_an_unsure_language_is_left_out(tmp_path: Path, stub_decoders: dict[str, int]) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    profiler = VideoProfiler(
        tmp_path,
        language_detector=lambda _w: ("hi", 0.4, [("hi", 0.4), ("gu", 0.35)]),
        sound_tagger=lambda _w: [],
    )
    profile = profiler.profile(video)
    assert profile.language is None
    assert profile.language_confidence == 0.4


def test_a_silent_clip_has_no_audio_signals(
    tmp_path: Path, stub_decoders: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vp, "read_audio", lambda _p: None)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    profile = VideoProfiler(tmp_path).profile(video)
    assert profile.has_audio is False
    assert profile.speech_share is None
    assert profile.content_type == "unknown"


def test_load_refuses_odd_ids_and_old_versions(tmp_path: Path) -> None:
    profiler = VideoProfiler(tmp_path)
    with pytest.raises(ValueError):
        profiler.load("../secrets")
    (tmp_path / "abc.json").write_text(json.dumps({"id": "abc", "duration_s": 1, "has_audio": False, "version": 0}))
    assert profiler.load("abc") is None
    assert profiler.load("missing") is None
