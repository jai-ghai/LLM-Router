# [Fact-Forcing preamble]
# Behaviour and module under test: FastAPI app from `src/tierrouter/service/app.py`
# (routes /v1/rules, /v1/examples, /v1/judgments, /v1/train, /v1/resolve, /v1/catalog,
# /v1/quality, /v1/history, plus new /v1/auth/* + /v1/users* in P0).
# Existing test files for it: this file. No other test file targets `app.py`.
# User's verbatim instruction: "create auth or required things because maybe user have
# different workflow and they can create difference judgments" — PLAN-v2.md §4 (P0 — Auth,
# Level B). "tests_too" chosen in follow-up: tests included with every PR.
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tierrouter.catalog import CatalogEntry, ModelCatalog
from tierrouter.service.app import create_app
from tierrouter.service.auth import SessionConfig, hash_password
from tierrouter.service.store import Store, guess_image_ext
from tierrouter.service.users_store import UsersStore
from tierrouter.tiers import TierSpec, TierTable

TIERS = TierTable(
    tiers=(
        TierSpec("small", "", {"gemini": "g-small"}),
        TierSpec("medium", "", {"gemini": "g-medium"}),
        TierSpec("large", "", {"gemini": "g-large"}),
    ),
    default_provider="gemini",
    low_confidence_fallback="medium",
)


def _authed_client(
    tmp_path: Path,
    *,
    role: str = "admin",
    catalog: ModelCatalog | None = None,
    quality_field: str | None = None,
    allow_read_without_auth: bool = False,
) -> TestClient:
    """A TestClient with a session cookie for a freshly-created `tester` user.

    Cost-12 bcrypt is fine per fixture (one hash). If this test suite grows into the
    hundreds of fixtures, swap `hash_password` for a stubbed scheme here.
    """
    store = Store(tmp_path / "data")
    users = UsersStore(path=store.root / "users.jsonl")
    users.add("tester", hash_password("tester-pw"), role)  # type: ignore[arg-type]
    app = create_app(
        TIERS,
        store,
        users,
        SessionConfig(secret="test-secret-does-not-matter-as-long-as-it-is-set"),
        catalog=catalog,
        quality_field=quality_field,
        allow_read_without_auth=allow_read_without_auth,
    )
    client = TestClient(app)
    login = client.post("/v1/auth/login", json={"username": "tester", "password": "tester-pw"})
    assert login.status_code == 200, login.text
    return client


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return _authed_client(tmp_path)


CATALOG = ModelCatalog(
    entries=(
        CatalogEntry("cheap-small", "gemini", "small", cost=1, label="Cheap Small"),
        CatalogEntry("pricey-small", "gemini", "small", cost=5, label="Pricey Small"),
    )
)


@pytest.fixture
def catalog_client(tmp_path: Path) -> TestClient:
    return _authed_client(tmp_path, catalog=CATALOG)


@pytest.fixture
def quality_client(tmp_path: Path) -> TestClient:
    return _authed_client(tmp_path, catalog=CATALOG, quality_field="task")


def test_guess_image_ext() -> None:
    assert guess_image_ext("photo.JPG", None) == ".jpg"
    assert guess_image_ext(None, "image/png") == ".png"
    assert guess_image_ext("file.txt", "image/webp") == ".webp"
    assert guess_image_ext(None, None) == ".png"


def test_healthz_and_tiers(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    tiers = client.get("/v1/tiers").json()
    assert [t["name"] for t in tiers["tiers"]] == ["small", "medium", "large"]
    assert tiers["low_confidence_fallback"] == "medium"


def test_rules_round_trip_and_validation(client: TestClient) -> None:
    assert client.get("/v1/rules").json() == {"rules": []}
    good = {"rules": [{"when": {"task": "news"}, "tier": "small"}]}
    response = client.put("/v1/rules", json=good)
    assert response.status_code == 200
    expected = [{"when": {"task": "news"}, "tier": "small", "priority": 0}]
    assert client.get("/v1/rules").json()["rules"] == expected

    bad = {"rules": [{"when": {"task": "news"}, "tier": "giant"}]}
    rejected = client.put("/v1/rules", json=bad)
    assert rejected.status_code == 422
    assert client.get("/v1/rules").json()["rules"] == expected  # unchanged


def test_rules_or_match_and_priority(client: TestClient) -> None:
    """A field may list several accepted values (OR), and `priority` breaks a tie between
    equally-specific rules — the two gaps flagged as making the old rules engine 'too basic'."""
    client.put(
        "/v1/rules",
        json={
            "rules": [
                {"when": {"task": ["qc_review", "audit"]}, "tier": "large"},
                {"when": {"task": "qc_review"}, "tier": "small", "priority": 5},
            ]
        },
    )
    decision = client.post(
        "/v1/resolve",
        data={"metadata": json.dumps({"task": "qc_review"}), "text": ""},
    ).json()
    # Both rules match; equally specific (one condition each), so the higher-priority rule wins.
    assert decision["tier"] == "small"
    decision2 = client.post(
        "/v1/resolve",
        data={"metadata": json.dumps({"task": "audit"}), "text": ""},
    ).json()
    assert decision2["tier"] == "large"


def test_add_example_without_image_and_coverage(client: TestClient) -> None:
    client.put("/v1/rules", json={"rules": [{"when": {"task": "video_analysis"}, "tier": "medium"}]})
    added = client.post(
        "/v1/examples",
        data={"metadata": json.dumps({"task": "video_analysis"}), "text": "a clip"},
    ).json()
    assert added["covered_by_rule"] is True
    assert added["has_image"] is False

    listed = client.get("/v1/examples").json()["examples"]
    assert len(listed) == 1
    assert listed[0]["example_id"] == added["example_id"]


def test_add_example_with_image_stores_the_file(client: TestClient, tmp_path: Path) -> None:
    files = {"image": ("pic.png", io.BytesIO(b"not-a-real-png-but-bytes"), "image/png")}
    added = client.post(
        "/v1/examples",
        data={"metadata": json.dumps({"task": "social_media_clip"}), "text": "pick a moment"},
        files=files,
    ).json()
    assert added["has_image"] is True
    assert added["covered_by_rule"] is False


def test_bad_metadata_json_is_rejected(client: TestClient) -> None:
    response = client.post("/v1/examples", data={"metadata": "not json", "text": "x"})
    assert response.status_code == 422


def test_uncovered_examples_and_judgments(client: TestClient) -> None:
    client.put("/v1/rules", json={"rules": [{"when": {"task": "video_analysis"}, "tier": "medium"}]})
    client.post("/v1/examples", data={"metadata": json.dumps({"task": "video_analysis"}), "text": "a"})
    uncovered_one = client.post("/v1/examples", data={"metadata": json.dumps({"task": "new_task"}), "text": "b"}).json()

    uncovered = client.get("/v1/examples/uncovered").json()["examples"]
    assert [e["example_id"] for e in uncovered] == [uncovered_one["example_id"]]

    bad_tier = client.post(
        "/v1/judgments", json={"example_id": uncovered_one["example_id"], "tier": "giant", "judged_by": "jai"}
    )
    assert bad_tier.status_code == 422

    bad_id = client.post("/v1/judgments", json={"example_id": "nope", "tier": "small", "judged_by": "jai"})
    assert bad_id.status_code == 422

    ok = client.post(
        "/v1/judgments", json={"example_id": uncovered_one["example_id"], "tier": "small", "judged_by": "jai"}
    )
    assert ok.status_code == 200


def test_train_needs_examples_then_labels(client: TestClient) -> None:
    assert client.post("/v1/train").status_code == 422  # no examples at all

    client.post("/v1/examples", data={"metadata": json.dumps({"task": "x"}), "text": "no rule, no judgment"})
    assert client.post("/v1/train").status_code == 422  # examples exist, but zero labels


def test_train_and_resolve_end_to_end_without_images(client: TestClient) -> None:
    client.put(
        "/v1/rules",
        json={
            "rules": [
                {"when": {"task": "video_analysis"}, "tier": "medium"},
                {"when": {"task": "qc_review"}, "tier": "large"},
            ]
        },
    )
    for task, text in [("video_analysis", "watch the clip"), ("qc_review", "check the counter")]:
        client.post("/v1/examples", data={"metadata": json.dumps({"task": task}), "text": text})

    result = client.post("/v1/train").json()
    assert result["n_labels"] == 2
    assert result["uses_images"] is False
    assert result["report"]["accuracy"] == 1.0

    status = client.get("/v1/status").json()
    assert status == {"trained": True, "n_labels_last_trained_on": 2, "uses_images": False}

    rule_hit = client.post(
        "/v1/resolve", data={"metadata": json.dumps({"task": "qc_review"}), "text": "x"}
    ).json()
    assert rule_hit["tier"] == "large"
    assert rule_hit["reason"] == "rule task=qc_review"

    classifier_hit = client.post(
        "/v1/resolve", data={"metadata": json.dumps({"task": "unseen_task"}), "text": "watch the clip"}
    ).json()
    assert "classifier" in classifier_hit["reason"]

    # trained WITHOUT any image example, so an image at resolve time cannot be used
    files = {"image": ("pic.png", io.BytesIO(b"bytes"), "image/png")}
    refused = client.post(
        "/v1/resolve", data={"metadata": json.dumps({"task": "unseen_task"}), "text": "x"}, files=files
    )
    assert refused.status_code == 422
    assert "not trained with any image" in refused.json()["detail"]


def test_resolve_with_image_is_refused_before_a_training_run(client: TestClient) -> None:
    files = {"image": ("pic.png", io.BytesIO(b"bytes"), "image/png")}
    response = client.post("/v1/resolve", data={"metadata": "{}", "text": "x"}, files=files)
    assert response.status_code == 200  # no classifier yet -> falls back, image is simply unused


def test_catalog_endpoint_is_empty_when_none_is_configured(client: TestClient) -> None:
    assert client.get("/v1/catalog").json() == {"catalog": []}


def test_catalog_endpoint_lists_entries(catalog_client: TestClient) -> None:
    body = catalog_client.get("/v1/catalog").json()
    ids = {e["id"] for e in body["catalog"]}
    assert ids == {"cheap-small", "pricey-small"}


def test_resolve_uses_the_catalog_when_one_is_configured(catalog_client: TestClient) -> None:
    catalog_client.put("/v1/rules", json={"rules": [{"when": {"task": "news"}, "tier": "small"}]})
    decision = catalog_client.post(
        "/v1/resolve", data={"metadata": json.dumps({"task": "news"}), "text": ""}
    ).json()
    assert decision["model_name"] == "cheap-small"  # cheapest, not g-small from tiers.yaml


def test_resolve_honours_selected_models(catalog_client: TestClient) -> None:
    catalog_client.put("/v1/rules", json={"rules": [{"when": {"task": "news"}, "tier": "small"}]})
    decision = catalog_client.post(
        "/v1/resolve",
        data={
            "metadata": json.dumps({"task": "news"}),
            "text": "",
            "selected_models": json.dumps(["pricey-small"]),
        },
    ).json()
    assert decision["model_name"] == "pricey-small"


def test_resolve_calls_are_recorded_in_history(client: TestClient) -> None:
    assert client.get("/v1/history").json() == {"history": []}
    client.put("/v1/rules", json={"rules": [{"when": {"task": "news"}, "tier": "small"}]})
    client.post("/v1/resolve", data={"metadata": json.dumps({"task": "news"}), "text": "a story"})
    history = client.get("/v1/history").json()["history"]
    assert len(history) == 1
    entry = history[0]
    assert entry["tier"] == "small"
    assert entry["metadata"] == {"task": "news"}
    assert entry["reason"] == "rule task=news"
    assert "timestamp" in entry


def test_quality_rating_round_trip_and_validation(client: TestClient) -> None:
    assert client.get("/v1/quality").json() == {"field": None, "ratings": []}
    # P0: client-supplied `judged_by` is ignored; the server records the session user
    # (`tester`, set up in the fixture). Covered as its own regression in test_auth.py; here
    # we just assert the stored field matches the session, not the request body.
    good = client.post(
        "/v1/quality",
        json={"model_id": "cheap-small", "metadata": {"task": "news"}, "rating": 4, "judged_by": "alex"},
    )
    assert good.status_code == 200
    ratings = client.get("/v1/quality").json()["ratings"]
    assert ratings == [{"model_id": "cheap-small", "metadata": {"task": "news"}, "rating": 4, "judged_by": "tester"}]

    bad = client.post(
        "/v1/quality",
        json={"model_id": "cheap-small", "metadata": {"task": "news"}, "rating": 9, "judged_by": "alex"},
    )
    assert bad.status_code == 422


def test_quality_endpoint_aggregates_when_a_field_is_configured(quality_client: TestClient) -> None:
    quality_client.post(
        "/v1/quality", json={"model_id": "cheap-small", "metadata": {"task": "news"}, "rating": 2, "judged_by": "a"}
    )
    quality_client.post(
        "/v1/quality", json={"model_id": "cheap-small", "metadata": {"task": "news"}, "rating": 4, "judged_by": "b"}
    )
    body = quality_client.get("/v1/quality").json()
    assert body["field"] == "task"
    assert body["scores"] == [{"model_id": "cheap-small", "value": "news", "average": 3.0}]


def test_resolve_min_quality_excludes_a_low_scoring_reviewed_model(quality_client: TestClient) -> None:
    quality_client.put("/v1/rules", json={"rules": [{"when": {"task": "news"}, "tier": "small"}]})
    # cheap-small is cheaper but rated 1.0 for "news"; pricey-small is unreviewed.
    quality_client.post(
        "/v1/quality", json={"model_id": "cheap-small", "metadata": {"task": "news"}, "rating": 1, "judged_by": "a"}
    )
    decision = quality_client.post(
        "/v1/resolve",
        data={"metadata": json.dumps({"task": "news"}), "text": "", "min_quality": "3.0"},
    ).json()
    assert decision["model_name"] == "pricey-small"  # unreviewed is never excluded; cheap-small was


def test_resolve_min_quality_has_no_effect_without_quality_field(catalog_client: TestClient) -> None:
    catalog_client.put("/v1/rules", json={"rules": [{"when": {"task": "news"}, "tier": "small"}]})
    catalog_client.post(
        "/v1/quality", json={"model_id": "cheap-small", "metadata": {"task": "news"}, "rating": 1, "judged_by": "a"}
    )
    decision = catalog_client.post(
        "/v1/resolve",
        data={"metadata": json.dumps({"task": "news"}), "text": "", "min_quality": "3.0"},
    ).json()
    assert decision["model_name"] == "cheap-small"  # no TIERROUTER_QUALITY_FIELD configured -> ignored
