from __future__ import annotations

import json
import urllib.error
from io import BytesIO
from typing import Self
from unittest.mock import patch

import pytest

from gradist.capability import (
    CapabilityError,
    ModelCapability,
    fetch_openrouter_capabilities,
    match_by_suffix,
)

SAMPLE_RESPONSE = {
    "data": [
        {
            "id": "google/gemini-3.5-flash",
            "context_length": 1048576,
            "architecture": {"input_modalities": ["text", "image"]},
            "pricing": {"prompt": "0.0000005", "completion": "0.0000015"},
        },
        {
            "id": "openai/text-only-model",
            "context_length": 8192,
            "architecture": {"input_modalities": ["text"]},
            "pricing": {"prompt": "0.000001", "completion": "0.000002"},
        },
        # missing an id -> must be skipped, not crash
        {"context_length": 1000},
    ]
}


class _FakeResponse:
    def __init__(self, payload: object) -> None:
        self._buffer = BytesIO(json.dumps(payload).encode("utf-8"))

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._buffer.read()


def test_fetch_parses_pricing_context_and_modality() -> None:
    with patch("gradist.capability.urllib.request.urlopen", return_value=_FakeResponse(SAMPLE_RESPONSE)):
        capabilities = fetch_openrouter_capabilities()
    assert len(capabilities) == 2  # the id-less entry is skipped
    flash = next(c for c in capabilities if c.id == "google/gemini-3.5-flash")
    assert flash.provider == "google"
    assert flash.context_length == 1048576
    assert flash.supports_image is True
    assert flash.cost_per_1k_prompt == pytest.approx(0.0005)
    text_only = next(c for c in capabilities if c.id == "openai/text-only-model")
    assert text_only.supports_image is False


def test_fetch_raises_a_capability_error_on_network_failure() -> None:
    with patch(
        "gradist.capability.urllib.request.urlopen",
        side_effect=urllib.error.URLError("no network"),
    ), pytest.raises(CapabilityError, match="could not reach"):
        fetch_openrouter_capabilities()


def test_fetch_raises_on_the_wrong_shape() -> None:
    with (
        patch("gradist.capability.urllib.request.urlopen", return_value=_FakeResponse({"oops": []})),
        pytest.raises(CapabilityError, match="expected a mapping"),
    ):
        fetch_openrouter_capabilities()


def test_match_by_suffix_exact_then_suffix_then_none() -> None:
    caps = [
        ModelCapability("google/gemini-3.5-flash", "google", None, True, None, None),
        ModelCapability("anthropic/claude-haiku-5-5", "anthropic", None, True, None, None),
    ]
    assert match_by_suffix("gemini-3.5-flash", caps) is caps[0]
    assert match_by_suffix("GEMINI-3.5-FLASH", caps) is caps[0]  # case-insensitive
    assert match_by_suffix("google/gemini-3.5-flash", caps) is caps[0]  # exact id also matches
    assert match_by_suffix("does-not-exist", caps) is None


# -----------------------------------------------------------------------------------------
# P3 — video + audio modality parsing (PLAN-v2.md §6.4).
# -----------------------------------------------------------------------------------------


def test_parses_video_and_audio_from_input_modalities() -> None:
    payload = {
        "data": [
            {
                "id": "google/gemini-video",
                "context_length": 128000,
                "architecture": {"input_modalities": ["text", "image", "video"]},
                "pricing": {"prompt": "0.000001", "completion": "0.000002"},
            },
            {
                "id": "openai/gpt-audio",
                "context_length": 32000,
                "architecture": {"input_modalities": ["text", "AUDIO"]},  # case-insensitive
                "pricing": {"prompt": "0.000001", "completion": "0.000002"},
            },
        ]
    }
    with patch("gradist.capability.urllib.request.urlopen", return_value=_FakeResponse(payload)):
        caps = fetch_openrouter_capabilities()
    video = next(c for c in caps if c.id == "google/gemini-video")
    assert video.supports_image is True
    assert video.supports_video is True
    assert video.supports_audio is False
    audio = next(c for c in caps if c.id == "openai/gpt-audio")
    assert audio.supports_image is False
    assert audio.supports_video is False
    assert audio.supports_audio is True


def test_absent_modalities_default_to_false() -> None:
    # No architecture block at all, and an empty modalities list — both must default
    # video/audio (and image) to False rather than crashing the fetch.
    payload = {
        "data": [
            {"id": "plain/no-arch", "context_length": 1000, "pricing": {}},
            {
                "id": "plain/empty-modalities",
                "architecture": {"input_modalities": []},
                "pricing": {},
            },
            {
                "id": "plain/bad-modalities",
                "architecture": {"input_modalities": "not-a-list"},
                "pricing": {},
            },
        ]
    }
    with patch("gradist.capability.urllib.request.urlopen", return_value=_FakeResponse(payload)):
        caps = fetch_openrouter_capabilities()
    assert len(caps) == 3
    for cap in caps:
        assert cap.supports_image is False
        assert cap.supports_video is False
        assert cap.supports_audio is False
