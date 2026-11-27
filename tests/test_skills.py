from __future__ import annotations

from pathlib import Path

import pytest

from gradist.rules import RuleError
from gradist.skills import SKILLS, indian_language, load_task_skills, request_needs, script_shares
from gradist.video_profile import VideoProfile

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


VIDEO = {"modality": "video"}


def clip(**fields: object) -> VideoProfile:
    return VideoProfile(**{"id": "c", "duration_s": 0.0, "has_audio": True, **fields})  # type: ignore[arg-type]


@pytest.mark.parametrize(("value", "code"), [("Tamil", "ta"), ("hi", "hi"), (" Oriya ", "or"), ("English", None)])
def test_indian_language_names_and_codes(value: str, code: str | None) -> None:
    assert indian_language(value) == code


def test_script_shares_names_each_indian_script() -> None:
    shares = script_shares("தமிழ் news")
    assert set(shares) == {"Tamil"}
    assert script_shares("12:40") == {}


def test_language_metadata_raises_the_indian_language_need() -> None:
    assert request_needs({"language": "Hindi"}, "summarise").needs["hindi"] == 0.9
    assert request_needs({"language": "Tamil"}, "summarise").needs["hindi"] == 1.0
    assert request_needs({"language": "English"}, "summarise").needs["hindi"] == 0.0


def test_a_low_resource_language_makes_the_request_harder() -> None:
    needs = request_needs({**VIDEO, "language": "Odia"}, "summarise the clip")
    assert needs.difficulty == 0.6
    assert "Odia" in needs.difficulty_reasons[0]


def test_any_indian_script_on_screen_counts() -> None:
    needs = request_needs(VIDEO, "log the scenes", screen_text="ಕನ್ನಡ ಸುದ್ದಿ")
    assert needs.needs["hindi"] == 1.0
    assert any("Kannada" in r for r in needs.reasons["hindi"])


def test_figures_on_screen_do_not_make_a_request_hard() -> None:
    screen = "12:40 | 2026-11-26 | Sensex 81,200 | Nifty 24,900 | ₹ 450 | 7 PM"
    assert request_needs(VIDEO, "log the scenes", screen_text=screen).difficulty is None
    assert request_needs({}, f"reconcile these: {screen}").difficulty is not None


def test_a_clip_without_audio_needs_no_hearing() -> None:
    needs = request_needs({**VIDEO, "channel_type": "news"}, "", video=clip(has_audio=False))
    assert needs.needs["audio"] == 0.0
    assert needs.reasons["audio"] == ["the clip has no audio track"]


def test_speech_and_singing_set_the_audio_need() -> None:
    assert request_needs(VIDEO, "", video=clip(speech_share=0.8)).needs["audio"] == pytest.approx(0.9)
    assert request_needs(VIDEO, "", video=clip(speech_share=0.05)).needs["audio"] == 0.0
    sung = request_needs(VIDEO, "", video=clip(speech_share=0.2, sounds={"singing": 0.3}, content_type="music"))
    assert sung.needs["audio"] == 1.0
    assert sung.difficulty == 0.8


def test_spoken_language_from_the_profile() -> None:
    needs = request_needs(VIDEO, "", video=clip(language="as", language_confidence=0.8))
    assert needs.needs["hindi"] == 1.0
    assert needs.difficulty == 0.6


def test_an_easy_clip_only_counts_when_nothing_harder_does() -> None:
    talk = clip(speech_share=0.9, content_type="talk")
    assert request_needs(VIDEO, "", video=talk).difficulty == 0.3
    assert request_needs(VIDEO, "", video=talk, qc_risk=0.7).difficulty == 0.7


def test_a_news_channel_clip_read_as_drama_stays_news() -> None:
    drama = clip(speech_share=0.8, content_type="drama")
    assert request_needs({**VIDEO, "channel_type": "news"}, "", video=drama).difficulty == 0.5
    assert request_needs({**VIDEO, "channel_type": "gec"}, "", video=drama).difficulty == 0.35


def test_an_easy_clip_is_flagged_only_when_nothing_makes_it_harder() -> None:
    talk = clip(speech_share=0.9, content_type="talk", language="hi", language_confidence=0.9, duration_s=600.0)
    assert request_needs(VIDEO, "", video=talk).easy_clip == "talk clip: one or two people speaking, few cuts; clear Hindi speech"
    unsure = clip(speech_share=0.9, content_type="talk", language=None, language_confidence=0.5, duration_s=600.0)
    assert request_needs(VIDEO, "", video=unsure).easy_clip is None
    assert request_needs(VIDEO, "", video=talk, qc_risk=0.6).easy_clip is None
    assert request_needs({**VIDEO, "channel_type": "news"}, "", video=talk).easy_clip is None
    assert request_needs({**VIDEO, "language": "Odia"}, "", video=talk).easy_clip is None
    long = clip(speech_share=0.9, content_type="talk", language="hi", language_confidence=0.9, duration_s=3600.0)
    assert request_needs(VIDEO, "", video=long).easy_clip is None


def test_the_profile_is_ignored_for_non_video_requests() -> None:
    needs = request_needs({}, "hello there", video=clip(speech_share=0.9, language="ta"))
    assert needs.needs["audio"] == 0.0
    assert needs.needs["hindi"] == 0.0


def test_clip_length_becomes_video_tokens() -> None:
    long = request_needs({**VIDEO, "duration_s": "600"}, "")
    assert long.input_tokens == 180_000
    assert long.needs["long_input"] == 1.0
    assert any("10 min of video" in r for r in long.reasons["long_input"])
    minute = request_needs(VIDEO, "", video=clip(duration_s=60.0))
    assert minute.needs["long_input"] == pytest.approx(18_000 / 32_000)
    assert request_needs({**VIDEO, "duration_s": "soon"}, "").input_tokens == 0
