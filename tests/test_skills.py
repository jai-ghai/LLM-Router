from __future__ import annotations

from pathlib import Path

import pytest

from gradist.rules import RuleError
from gradist.skills import SKILLS, load_task_skills, request_needs

REPO_TASK_SKILLS = Path(__file__).resolve().parents[1] / "config" / "task_skills.yaml"


def test_a_hindi_news_clip_needs_video_hearing_and_hindi() -> None:
    needs = request_needs(
        {"task": "video_processing", "modality": "video", "channel_type": "news"},
        "",
        screen_text="वेटलिफ्टर बिंदियारानी देवी ने जीता ब्रॉन्ज | 2026-07-28 07:40",
        has_image=True,
        task_skills={"video_processing": {"video": 1.0, "audio": 0.8, "long_input": 0.8}},
    )
    assert needs.needs["video"] == 1.0
    assert needs.needs["audio"] == 0.8
    assert needs.needs["hindi"] == 1.0
    assert needs.needs["fast"] == 0.0
    assert needs.task_profile == "video_processing"
    assert any("Devanagari" in r for r in needs.reasons["hindi"])


def test_a_short_english_similarity_check_needs_speed_not_video() -> None:
    needs = request_needs(
        {"task": "headline_similarity"},
        "Are these the same story? A: Rates held. B: RBI keeps repo rate unchanged.",
        task_skills={"headline_similarity": {"fast": 1.0}},
    ).as_signal()
    assert needs["needs"] == {**dict.fromkeys(SKILLS, 0.0), "fast": 1.0}


def test_qc_risk_and_long_input_raise_their_needs_without_a_task_profile() -> None:
    needs = request_needs({"channel_name": "Aaj Tak"}, "word " * 6000, qc_risk=0.83)
    assert needs.needs["reasoning"] == pytest.approx(0.83)
    assert 0 < needs.needs["long_input"] < 1
    assert needs.task_profile is None


def test_difficulty_is_low_for_a_short_plain_request_and_high_for_figures_and_conditions() -> None:
    easy = request_needs({}, "Is this headline the same story as that one?")
    assert easy.difficulty == 0.25
    assert easy.difficulty_reasons == ["short text-only request (9 words) (0.25)"]

    hard = request_needs(
        {},
        "Pass only if the score is 3/5 or more, unless the clip is under 30 s; "
        "however reject it if 2 of the 4 frames or 10% of the audio are missing.",
    )
    assert hard.difficulty is not None and hard.difficulty >= 0.6
    assert any("figures" in r for r in hard.difficulty_reasons)
    assert any("conditions" in r for r in hard.difficulty_reasons)


def test_difficulty_follows_qc_risk_and_is_neutral_without_any_signal() -> None:
    assert request_needs({}, "x", qc_risk=0.8).difficulty == 0.8
    assert request_needs({"modality": "video"}, "").difficulty is None
    assert request_needs({}, "").as_signal()["difficulty"] is None


def test_the_shipped_task_skills_file_is_valid() -> None:
    profiles = load_task_skills(REPO_TASK_SKILLS)
    assert profiles["v3_single"]["video"] == 1.0
    assert profiles["headline_similarity"]["fast"] == 1.0


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("tasks: {t: {vision: 0.5}}", "unknown skill"),
        ("tasks: {t: {hindi: 1.5}}", "between 0 and 1"),
        ("nothing: here", "'tasks' mapping"),
    ],
)
def test_bad_task_skills_files_are_rejected(tmp_path: Path, body: str, message: str) -> None:
    path = tmp_path / "task_skills.yaml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(RuleError, match=message):
        load_task_skills(path)
