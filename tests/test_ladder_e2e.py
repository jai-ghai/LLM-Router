# [Fact-Forcing preamble]
# Behaviour under test: ladder resolve precedence rule → classifier → fallback, including
# thumbnail/image bytes. Modules: `gradist.resolve.resolve`, `gradist.service.app` train/resolve.
# Existing coverage: `tests/test_resolve.py` (unit ladder + catalog), `tests/test_service.py`
# (`test_train_and_resolve_end_to_end_without_images`, image-refuse cases), `tests/test_classifier.py`.
# Gap: no dedicated E2E that clears rules after train, asserts classifier_outcome / fallback_outcome,
# and trains+resolves WITH thumbnails (FakeImageEmbedder). User instruction: "we only tested on
# rules .. i wanted to test through classifier and fallback please test this .. end to end all
# the functionality is proper .. use generic data .. check with thumbnails all tests .. use skill
# if you want .. ecc test skill"
"""End-to-end ladder coverage: rule → classifier → fallback, including thumbnails.

Catalog-first skips this ladder; these tests use NO catalog so precedence is real.
Image plumbing uses FakeImageEmbedder (ECC sandbox/fast path) — Dinov2 stays behind
GRADIST_TEST_REAL_IMAGE_MODEL in test_dinov2_embedder.py.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from gradist.classifier import TierClassifier
from gradist.features import HashingEmbedder, MetadataEncoder
from gradist.labels import TierLabel
from gradist.ocr import ScreenLine, ScreenReader, merge_frame_lines
from gradist.resolve import resolve
from gradist.rules import Rule, RuleSet
from gradist.service.app import AppState, create_app
from gradist.service.auth import SessionConfig, hash_password
from gradist.service.store import Store
from gradist.service.users_store import UsersStore
from gradist.tiers import TierSpec, TierTable

TIERS = TierTable(
    tiers=(
        TierSpec("small", "cheap", {"gemini": "g-small", "anthropic": "a-small"}),
        TierSpec("medium", "default", {"gemini": "g-medium", "anthropic": "a-medium"}),
        TierSpec("large", "hard", {"gemini": "g-large", "anthropic": "a-large"}),
    ),
    default_provider="gemini",
    low_confidence_fallback="medium",
)

NEWS = [
    "parliament passed the budget bill today",
    "the minister announced a snap election",
    "court delivered the verdict this morning",
]
SPORTS = [
    "goal scored in the final match overtime",
    "the striker won the tournament cup",
    "wicket fell after a century stand",
]


class FakeImageEmbedder:
    """Deterministic fake: embedding encodes byte-length so red vs blue thumbnails differ."""

    width = 4

    def __init__(self) -> None:
        self.calls: list[bytes] = []

    def embed(self, images: list[bytes]) -> np.ndarray:
        self.calls.extend(images)
        return np.array([[float(len(img)), 0.5, 1.5, 2.5] for img in images], dtype=np.float64)


def _tiny_png(tag: bytes) -> bytes:
    header = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde"
    )
    return header + tag


RED_THUMB = _tiny_png(b"RED" * 40)
BLUE_THUMB = _tiny_png(b"BLU" * 10)


def _encoder() -> MetadataEncoder:
    return MetadataEncoder({"channel_type": ["news", "sports"], "task": ["headline", "qc_review"]})


def _trained_text_classifier() -> TierClassifier:
    clf = TierClassifier(_encoder(), HashingEmbedder())
    labels = [
        TierLabel(f"n{i}", {"channel_type": "news", "task": "headline"}, t, "small", "human", "")
        for i, t in enumerate(NEWS)
    ]
    labels += [
        TierLabel(f"s{i}", {"channel_type": "sports", "task": "qc_review"}, t, "large", "human", "")
        for i, t in enumerate(SPORTS)
    ]
    clf.fit(labels)
    return clf


def test_unit_classifier_path_when_no_rule_matches() -> None:
    decision = resolve(
        {"channel_type": "news", "task": "headline"},
        NEWS[0],
        RuleSet(),
        TIERS,
        classifier=_trained_text_classifier(),
    )
    assert decision.tier == "small"
    assert decision.model_name == "g-small"
    assert decision.rule_outcome == "no matching rule"
    assert decision.classifier_outcome is not None
    assert decision.classifier_outcome.startswith("small, confidence")
    assert decision.fallback_outcome == "not needed"
    assert "classifier, confidence" in decision.reason


def test_unit_fallback_when_no_rule_and_no_classifier() -> None:
    decision = resolve({"channel_type": "news"}, "anything generic", RuleSet(), TIERS)
    assert decision.tier == "medium"
    assert decision.model_name == "g-medium"
    assert decision.confidence is None
    assert decision.rule_outcome == "no matching rule"
    assert decision.classifier_outcome == "no classifier configured"
    assert decision.fallback_outcome == "used 'medium'"
    assert "no classifier is available" in decision.reason


def test_unit_low_confidence_fallback_still_reports_true_confidence() -> None:
    decision = resolve(
        {"channel_type": "news", "task": "headline"},
        NEWS[0],
        RuleSet(),
        TIERS,
        classifier=_trained_text_classifier(),
        min_confidence=0.999,
    )
    assert decision.tier == "medium"
    assert decision.model_name == "g-medium"
    assert decision.confidence is not None and decision.confidence < 0.999
    assert "below the 1.00 threshold" in decision.reason
    assert decision.fallback_outcome == "used 'medium'"
    assert decision.classifier_outcome is not None and "below the" in decision.classifier_outcome


def test_unit_rule_still_beats_classifier() -> None:
    rules = RuleSet(rules=(Rule.create({"channel_type": "sports"}, "small"),))
    decision = resolve(
        {"channel_type": "sports", "task": "qc_review"},
        SPORTS[0],
        rules,
        TIERS,
        classifier=_trained_text_classifier(),
    )
    assert decision.tier == "small"
    assert decision.confidence == 1.0
    assert decision.classifier_outcome is None
    assert decision.fallback_outcome is None


def test_unit_classifier_with_thumbnail_uses_image_bytes(tmp_path: Path) -> None:
    embedder = FakeImageEmbedder()
    clf = TierClassifier(_encoder(), HashingEmbedder(n_features=64), embedder)

    red_path = tmp_path / "red.png"
    blue_path = tmp_path / "blue.png"
    red_path.write_bytes(RED_THUMB)
    blue_path.write_bytes(BLUE_THUMB)

    labels = [
        TierLabel(
            f"red-{i}",
            {"channel_type": "news", "task": "headline"},
            "thumbnail of a busy promo slate",
            "large",
            "human",
            "",
            str(red_path),
        )
        for i in range(3)
    ] + [
        TierLabel(
            f"blue-{i}",
            {"channel_type": "news", "task": "headline"},
            "thumbnail of a plain slate",
            "small",
            "human",
            "",
            str(blue_path),
        )
        for i in range(3)
    ]
    clf.fit(labels)
    assert clf.uses_images is True

    large_decision = resolve(
        {"channel_type": "news", "task": "headline"},
        "thumbnail of a slate",
        RuleSet(),
        TIERS,
        classifier=clf,
        image_bytes=RED_THUMB,
    )
    small_decision = resolve(
        {"channel_type": "news", "task": "headline"},
        "thumbnail of a slate",
        RuleSet(),
        TIERS,
        classifier=clf,
        image_bytes=BLUE_THUMB,
    )
    assert large_decision.rule_outcome == "no matching rule"
    assert small_decision.rule_outcome == "no matching rule"
    assert large_decision.fallback_outcome == "not needed"
    assert small_decision.fallback_outcome == "not needed"
    assert large_decision.tier == "large"
    assert large_decision.model_name == "g-large"
    assert small_decision.tier == "small"
    assert small_decision.model_name == "g-small"
    assert embedder.calls

    inputs = large_decision.signals["classifier_inputs"]
    assert isinstance(inputs, dict)
    assert inputs["image_used"] is True
    assert inputs["image_dims"] == 4
    assert inputs["text_dims"] == 64
    assert inputs["metadata_buckets"] == {"channel_type": "news", "task": "headline"}
    assert inputs["without_image"]["large"] < large_decision.tier_scores["large"]


def test_explain_matches_predict_proba_and_embeds_the_image_once() -> None:
    embedder = FakeImageEmbedder()
    clf = TierClassifier(_encoder(), HashingEmbedder(n_features=64), embedder)
    clf.fit(
        [
            TierLabel("a", {"channel_type": "news"}, NEWS[0], "small", "human", ""),
            TierLabel("b", {"channel_type": "sports"}, SPORTS[0], "large", "human", ""),
        ]
    )
    meta = {"channel_type": "sports", "not_a_field": "x"}
    report = clf.explain(meta, SPORTS[1], RED_THUMB)
    assert embedder.calls.count(RED_THUMB) == 1
    assert report["proba"] == pytest.approx(clf.predict_proba(meta, SPORTS[1], RED_THUMB))
    assert report["metadata_buckets"] == {"channel_type": "sports"}
    assert set(report) >= {"without_image", "without_text", "without_metadata"}


def _ladder_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    fake = FakeImageEmbedder()

    def _fake_image_embedder(self: AppState) -> FakeImageEmbedder:  # type: ignore[override]
        return fake

    monkeypatch.setattr(AppState, "image_embedder", _fake_image_embedder)

    store = Store(tmp_path / "data")
    users = UsersStore(path=store.root / "users.jsonl")
    users.add("tester", hash_password("tester-pw"), "admin")
    app = create_app(
        TIERS,
        store,
        users,
        SessionConfig(secret="ladder-e2e-secret"),
        catalog=None,
    )
    client = TestClient(app)
    assert client.post("/v1/auth/login", json={"username": "tester", "password": "tester-pw"}).status_code == 200
    return client


def test_service_fallback_before_any_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _ladder_client(tmp_path, monkeypatch)
    status = client.get("/v1/status").json()
    assert status["trained"] is False
    assert status["catalog_configured"] is False

    decision = client.post(
        "/v1/resolve",
        data={"metadata": json.dumps({"channel_type": "news", "task": "headline"}), "text": "generic request"},
    ).json()
    assert decision["tier"] == "medium"
    assert decision["model_name"] == "g-medium"
    assert decision["rule_outcome"] == "no matching rule"
    assert decision["classifier_outcome"] == "no classifier configured"
    assert decision["fallback_outcome"] == "used 'medium'"


def test_service_classifier_and_low_confidence_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _ladder_client(tmp_path, monkeypatch)

    client.put(
        "/v1/rules",
        json={
            "rules": [
                {"when": {"channel_type": "news"}, "tier": "small"},
                {"when": {"channel_type": "sports"}, "tier": "large"},
            ]
        },
    )
    for text in NEWS:
        client.post(
            "/v1/examples",
            data={"metadata": json.dumps({"channel_type": "news", "task": "headline"}), "text": text},
        )
    for text in SPORTS:
        client.post(
            "/v1/examples",
            data={"metadata": json.dumps({"channel_type": "sports", "task": "qc_review"}), "text": text},
        )

    train = client.post("/v1/train").json()
    assert train["n_labels"] == 6
    assert train["uses_images"] is False
    assert client.get("/v1/status").json()["trained"] is True

    # Clear rules so resolve must use the classifier.
    client.put("/v1/rules", json={"rules": []})

    hit = client.post(
        "/v1/resolve",
        data={
            "metadata": json.dumps({"channel_type": "news", "task": "headline"}),
            "text": NEWS[0],
        },
    ).json()
    assert hit["tier"] == "small"
    assert hit["model_name"] == "g-small"
    assert hit["rule_outcome"] == "no matching rule"
    assert "classifier" in hit["reason"]
    assert hit["fallback_outcome"] == "not needed"

    low = client.post(
        "/v1/resolve",
        data={
            "metadata": json.dumps({"channel_type": "news", "task": "headline"}),
            "text": NEWS[0],
            "min_confidence": "0.999",
        },
    ).json()
    assert low["tier"] == "medium"
    assert low["model_name"] == "g-medium"
    assert "below the" in low["reason"]
    assert low["fallback_outcome"] == "used 'medium'"


def test_service_rule_overrides_trained_classifier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _ladder_client(tmp_path, monkeypatch)
    client.put(
        "/v1/rules",
        json={"rules": [{"when": {"channel_type": "sports"}, "tier": "large"}]},
    )
    for text in SPORTS:
        client.post(
            "/v1/examples",
            data={"metadata": json.dumps({"channel_type": "sports"}), "text": text},
        )
    assert client.post("/v1/train").status_code == 200

    client.put(
        "/v1/rules",
        json={"rules": [{"when": {"channel_type": "sports"}, "tier": "small"}]},
    )
    decision = client.post(
        "/v1/resolve",
        data={"metadata": json.dumps({"channel_type": "sports"}), "text": SPORTS[0]},
    ).json()
    assert decision["tier"] == "small"
    assert decision["model_name"] == "g-small"
    assert decision["reason"] == "rule channel_type=sports"
    assert decision["classifier_outcome"] is None


def test_service_thumbnail_refused_when_classifier_trained_without_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _ladder_client(tmp_path, monkeypatch)
    client.put("/v1/rules", json={"rules": [{"when": {"channel_type": "news"}, "tier": "small"}]})
    client.post(
        "/v1/examples",
        data={"metadata": json.dumps({"channel_type": "news"}), "text": NEWS[0]},
    )
    assert client.post("/v1/train").json()["uses_images"] is False

    client.put("/v1/rules", json={"rules": []})
    refused = client.post(
        "/v1/resolve",
        data={"metadata": json.dumps({"channel_type": "news"}), "text": "x"},
        files={"image": ("thumb.png", io.BytesIO(RED_THUMB), "image/png")},
    )
    assert refused.status_code == 422
    assert "not trained with any image" in refused.json()["detail"]


def test_service_thumbnail_accepted_when_classifier_trained_with_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _ladder_client(tmp_path, monkeypatch)

    for i, text in enumerate(NEWS[:2]):
        added = client.post(
            "/v1/examples",
            data={"metadata": json.dumps({"channel_type": "news", "task": "headline"}), "text": text},
            files={"image": (f"red-{i}.png", io.BytesIO(RED_THUMB), "image/png")},
        ).json()
        assert added["has_image"] is True
        assert (
            client.post(
                "/v1/judgments",
                json={"example_id": added["example_id"], "tier": "large", "notes": "busy thumb"},
            ).status_code
            == 200
        )

    for i, text in enumerate(SPORTS[:2]):
        added = client.post(
            "/v1/examples",
            data={"metadata": json.dumps({"channel_type": "sports", "task": "qc_review"}), "text": text},
            files={"image": (f"blue-{i}.png", io.BytesIO(BLUE_THUMB), "image/png")},
        ).json()
        assert (
            client.post(
                "/v1/judgments",
                json={"example_id": added["example_id"], "tier": "small", "notes": "plain thumb"},
            ).status_code
            == 200
        )

    train = client.post("/v1/train").json()
    assert train["uses_images"] is True
    assert train["n_human_labels"] == 4
    assert client.get("/v1/status").json()["uses_images"] is True

    with_thumb = client.post(
        "/v1/resolve",
        data={
            "metadata": json.dumps({"channel_type": "news", "task": "headline"}),
            "text": "generic promo frame",
        },
        files={"image": ("query.png", io.BytesIO(RED_THUMB), "image/png")},
    )
    assert with_thumb.status_code == 200, with_thumb.text
    body = with_thumb.json()
    assert body["rule_outcome"] == "no matching rule"
    assert body["classifier_outcome"] is not None
    assert body["model_name"] in {"g-small", "g-medium", "g-large"}
    assert body["tier"] == "large"
    assert body["fallback_outcome"] == "not needed"

    no_thumb = client.post(
        "/v1/resolve",
        data={
            "metadata": json.dumps({"channel_type": "sports", "task": "qc_review"}),
            "text": SPORTS[0],
        },
    )
    assert no_thumb.status_code == 200
    assert no_thumb.json()["rule_outcome"] == "no matching rule"


class FakeScreenReader:
    engine = "fake-ocr"

    def read(self, image_bytes: bytes) -> list[ScreenLine]:
        return [ScreenLine("BREAKING NEWS", 0.9)] if image_bytes == RED_THUMB else [ScreenLine("SCORE 2-1", 0.8)]

    def save_cache(self) -> None:
        pass


def test_service_reads_screen_text_when_ocr_is_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _ladder_client(tmp_path, monkeypatch)
    reader = FakeScreenReader()
    monkeypatch.setattr(AppState, "screen_reader", lambda self: reader)
    for tier, thumb, meta in (
        ("large", RED_THUMB, {"channel_type": "news", "channel_code": "101", "channel_name": "Alpha"}),
        ("small", BLUE_THUMB, {"channel_type": "sports", "channel_code": "202", "channel_name": "Beta"}),
    ):
        for i in range(2):
            added = client.post(
                "/v1/examples",
                data={"metadata": json.dumps(meta), "text": f"clip {i}"},
                files={"image": ("f.png", io.BytesIO(thumb), "image/png")},
            ).json()
            client.post("/v1/judgments", json={"example_id": added["example_id"], "tier": tier, "notes": ""})
    assert client.post("/v1/train").status_code == 200
    meta = json.loads((tmp_path / "data" / "classifier.meta.json").read_text(encoding="utf-8"))
    assert meta["reads_screen_text"] is True

    body = client.post(
        "/v1/resolve",
        data={"metadata": json.dumps({"channel_type": "news"}), "text": "check this clip"},
        files={"image": ("q.png", io.BytesIO(RED_THUMB), "image/png")},
    ).json()
    assert body["signals"]["screen_text"] == {
        "engine": "fake-ocr",
        "lines": [{"text": "BREAKING NEWS", "confidence": 0.9}],
    }
    inputs = body["signals"]["classifier_inputs"]
    assert inputs["screen_text_used"] is True
    assert set(inputs["without_screen_text"]) == {"small", "large"}

    channels = client.get("/v1/channels").json()["channels"]
    assert channels["101"] == {"channel_name": "Alpha", "channel_type": "news"}


def test_merge_frame_lines_keeps_each_line_once_at_its_best_confidence() -> None:
    merged = merge_frame_lines(
        [
            [ScreenLine("BREAKING NEWS", 0.6), ScreenLine("Aaj Tak", 0.9)],
            [ScreenLine("breaking  news", 0.8), ScreenLine("SCORE 2-1", 0.7)],
        ]
    )
    assert merged == [ScreenLine("BREAKING NEWS", 0.8), ScreenLine("Aaj Tak", 0.9), ScreenLine("SCORE 2-1", 0.7)]


def test_screen_reader_reads_a_frame_once_when_asked_concurrently() -> None:
    import threading
    import time

    reader = ScreenReader.__new__(ScreenReader)
    reader._lock = threading.Lock()
    reader._cache = {}
    reader._cache_path = None
    calls: list[bytes] = []

    def slow_run(image_bytes: bytes) -> list[ScreenLine]:
        calls.append(image_bytes)
        time.sleep(0.2)
        return [ScreenLine("TICKER", 0.9)]

    reader._run = slow_run  # type: ignore[method-assign]
    results: list[list[ScreenLine]] = []
    threads = [threading.Thread(target=lambda: results.append(reader.read(b"frame"))) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1
    assert results == [[ScreenLine("TICKER", 0.9)]] * 2


class CountingScreenReader(FakeScreenReader):
    def __init__(self) -> None:
        self.cache: dict[bytes, list[ScreenLine]] = {}
        self.runs = 0

    def read(self, image_bytes: bytes) -> list[ScreenLine]:
        if image_bytes not in self.cache:
            self.runs += 1
            self.cache[image_bytes] = super().read(image_bytes)
        return self.cache[image_bytes]


def _png_parts(field: str, images: tuple[bytes, ...]) -> list[tuple[str, tuple[str, io.BytesIO, str]]]:
    return [(field, (f"f{i}.png", io.BytesIO(b), "image/png")) for i, b in enumerate(images)]


def test_prefetched_frames_are_not_read_again_at_resolve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _ladder_client(tmp_path, monkeypatch)
    reader = CountingScreenReader()
    monkeypatch.setattr(AppState, "screen_reader", lambda self: reader)

    pre = client.post("/v1/ocr", files=_png_parts("frames", (RED_THUMB, BLUE_THUMB, RED_THUMB))).json()
    assert pre["frames_read"] == 3
    assert [ln["text"] for ln in pre["lines"]] == ["BREAKING NEWS", "SCORE 2-1"]
    assert reader.runs == 2

    body = client.post(
        "/v1/resolve",
        data={"metadata": json.dumps({"channel_type": "news", "modality": "video"}), "text": "clip"},
        files=_png_parts("image", (RED_THUMB,)) + _png_parts("ocr_frames", (RED_THUMB, BLUE_THUMB)),
    ).json()
    screen = body["signals"]["screen_text"]
    assert screen["frames_read"] == 2
    assert [ln["text"] for ln in screen["lines"]] == ["BREAKING NEWS", "SCORE 2-1"]
    assert reader.runs == 2

    assert client.post("/v1/ocr", files=_png_parts("frames", (RED_THUMB,) * 9)).status_code == 422


def test_service_fallback_with_unused_thumbnail_before_train(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _ladder_client(tmp_path, monkeypatch)
    response = client.post(
        "/v1/resolve",
        data={"metadata": json.dumps({"task": "anything"}), "text": "hello"},
        files={"image": ("t.png", io.BytesIO(BLUE_THUMB), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["tier"] == "medium"
    assert body["fallback_outcome"] == "used 'medium'"
    assert body["classifier_outcome"] == "no classifier configured"
