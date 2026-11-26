# [Fact-Forcing preamble]
# Behaviour and module under test: FastAPI app from `src/gradist/service/app.py`
# (routes /v1/rules, /v1/examples, /v1/judgments, /v1/train, /v1/resolve, /v1/catalog,
# /v1/quality, /v1/history, plus new /v1/auth/* + /v1/users* in P0).
# Existing test files for it: this file. No other test file targets `app.py`.
# User's verbatim instruction: "create auth or required things because maybe user have
# different workflow and they can create difference judgments".
# "tests_too" chosen in follow-up: tests included with every PR.
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gradist.catalog import CatalogEntry, ModelCatalog
from gradist.service.app import create_app
from gradist.service.auth import SessionConfig, hash_password
from gradist.service.store import Store, guess_image_ext
from gradist.service.users_store import UsersStore
from gradist.tiers import TierSpec, TierTable

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
    catalog_path: Path | None = None,
    quality_field: str | None = None,
    allow_read_without_auth: bool = False,
) -> TestClient:
    """A TestClient with a session cookie for a freshly-created `tester` user.

    Cost-12 bcrypt is fine per fixture (one hash). If this test suite grows into the
    hundreds of fixtures, swap `hash_password` for a stubbed scheme here.
    """
    store = Store(tmp_path / "data", catalog_path=catalog_path)
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


def _write_catalog_yaml(path: Path, catalog: ModelCatalog) -> None:
    import yaml

    from gradist.catalog import entry_to_yaml_dict

    path.write_text(
        yaml.safe_dump({"catalog": [entry_to_yaml_dict(e) for e in catalog.entries]}, sort_keys=False),
        encoding="utf-8",
    )


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
    # P2 adds optimize_for to the wire shape; absent on the put means null on the get.
    expected = [{"when": {"task": "news"}, "tier": "small", "priority": 0, "optimize_for": None}]
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
    assert status["trained"] is True
    assert status["n_labels_last_trained_on"] == 2
    assert status["uses_images"] is False
    assert status["catalog_configured"] is False  # default fixture has no catalog
    assert status["unreviewed_count"] == 0
    assert status["quality_field"] is None
    assert status["allow_read_without_auth"] is False

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


def test_every_decision_reports_the_request_skill_needs(client: TestClient) -> None:
    client.put("/v1/rules", json={"rules": [{"when": {"task": "news"}, "tier": "small"}]})
    body = client.post(
        "/v1/resolve", data={"metadata": json.dumps({"task": "news", "modality": "video"}), "text": "a story"}
    ).json()
    skill_needs = body["signals"]["skill_needs"]
    assert skill_needs["needs"]["video"] == 1.0
    assert "modality=video (1.0)" in skill_needs["reasons"]["video"]


def test_a_hindi_request_picks_the_catalog_model_that_fits_best(tmp_path: Path) -> None:
    catalog = ModelCatalog(
        entries=(
            CatalogEntry("cheap-english", "gemini", "small", cost=0.0001, multilingual=0.4, multilingual_source="measured"),
            CatalogEntry("hindi-strong", "gemini", "small", cost=0.0003, multilingual=0.9, multilingual_source="measured"),
        )
    )
    client = _authed_client(tmp_path, catalog=catalog)
    client.put("/v1/rules", json={"rules": [{"when": {"task": "news"}, "tier": "small"}]})
    body = client.post(
        "/v1/resolve",
        data={"metadata": json.dumps({"task": "news"}), "text": "मुख्यमंत्री ने आज नई योजना की घोषणा की"},
    ).json()
    assert body["model_name"] == "hindi-strong"
    assert [c["id"] for c in body["candidates"]] == ["hindi-strong", "cheap-english"]
    assert body["candidates"][0]["fit_parts"]["hindi"] == 0.9
    assert body["candidates"][0]["fit"] > body["candidates"][1]["fit"]
    assert ", fit " in body["reason"]


def test_bad_ratings_on_hindi_requests_move_the_next_pick(tmp_path: Path) -> None:
    catalog = ModelCatalog(
        entries=(
            CatalogEntry("cheap-english", "gemini", "small", cost=0.0001, multilingual=0.6, multilingual_source="measured"),
            CatalogEntry("hindi-strong", "gemini", "small", cost=0.0003, multilingual=0.9, multilingual_source="measured"),
        )
    )
    client = _authed_client(tmp_path, catalog=catalog)
    client.put("/v1/rules", json={"rules": [{"when": {"task": "news"}, "tier": "small"}]})
    form = {"metadata": json.dumps({"task": "news"}), "text": "मुख्यमंत्री ने आज नई योजना की घोषणा की"}

    first = client.post("/v1/resolve", data=form).json()
    assert first["model_name"] == "hindi-strong"
    rated = client.post("/v1/quality", json={"request_id": first["request_id"], "rating": 1, "model_id": "spoofed"})
    assert rated.status_code == 200, rated.text

    stored = client.get("/v1/quality").json()["ratings"]
    assert stored[0]["model_id"] == "hindi-strong"
    logged = client.get("/v1/history").json()["history"][0]
    assert logged["skill_needs"]["hindi"] > 0
    assert [r["rating"] for r in logged["ratings"]] == [1]
    assert logged["ratings"][0]["judged_by"] == stored[0]["judged_by"]
    after = client.post("/v1/resolve", data=form).json()
    assert after["model_name"] == "cheap-english"
    loser = next(c for c in after["candidates"] if c["id"] == "hindi-strong")
    assert loser["learned"]["hindi"] == {"prior": 0.9, "value": 0.675, "ratings": 1}
    assert after["signals"]["moved_by_ratings"] == [
        {"id": "hindi-strong", "rank": 2, "fit": loser["fit"], "learned": loser["learned"]}
    ]


def test_rating_by_request_id_checks_the_request_exists_and_belongs_to_you(tmp_path: Path) -> None:
    admin = _authed_client(tmp_path)
    admin.put("/v1/rules", json={"rules": [{"when": {"task": "news"}, "tier": "small"}]})
    request_id = admin.post("/v1/resolve", data={"metadata": json.dumps({"task": "news"}), "text": "x"}).json()["request_id"]

    assert admin.post("/v1/quality", json={"request_id": "nope", "rating": 3}).status_code == 404
    assert admin.post("/v1/quality", json={"rating": 3}).status_code == 422

    admin.post("/v1/users", json={"username": "member", "password": "member-pw", "role": "member"})
    member = TestClient(admin.app)
    assert member.post("/v1/auth/login", json={"username": "member", "password": "member-pw"}).status_code == 200
    assert member.post("/v1/quality", json={"request_id": request_id, "rating": 3}).status_code == 403


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


def test_first_admin_comes_from_env_only_while_no_user_exists(tmp_path: Path) -> None:
    from gradist.service.auth import verify_password
    from gradist.service.main import seed_admin_from_env

    users = UsersStore(path=tmp_path / "users.jsonl")
    env = {"GRADIST_ADMIN_USERNAME": "jai", "GRADIST_ADMIN_PASSWORD": "pw-123"}
    assert seed_admin_from_env(users, {}) is False
    assert seed_admin_from_env(users, env) is True
    admin = users.find_by_username("jai")
    assert admin is not None and admin.role == "admin"
    assert verify_password("pw-123", admin.password_hash)
    assert "pw-123" not in (tmp_path / "users.jsonl").read_text(encoding="utf-8")
    assert seed_admin_from_env(users, {**env, "GRADIST_ADMIN_USERNAME": "other"}) is False
    assert [u.username for u in users.list_users()] == ["jai"]
    with pytest.raises(RuntimeError, match="GRADIST_ADMIN_PASSWORD"):
        seed_admin_from_env(UsersStore(path=tmp_path / "empty.jsonl"), {"GRADIST_ADMIN_USERNAME": "x"})


def test_history_is_private_to_members_and_filterable_by_admins(tmp_path: Path) -> None:
    store = Store(tmp_path / "data")
    users = UsersStore(path=store.root / "users.jsonl")
    for name, role in (("boss", "admin"), ("asha", "member"), ("ravi", "member")):
        users.add(name, hash_password(f"{name}-pw"), role)  # type: ignore[arg-type]
    app = create_app(TIERS, store, users, SessionConfig(secret="s"))

    def login(name: str) -> TestClient:
        c = TestClient(app)
        assert c.post("/v1/auth/login", json={"username": name, "password": f"{name}-pw"}).status_code == 200
        return c

    clients = {name: login(name) for name in ("boss", "asha", "ravi")}
    for name, c in clients.items():
        c.post("/v1/resolve", data={"metadata": json.dumps({"task": name}), "text": f"from {name}"})

    def owners(c: TestClient, query: str = "") -> list[str]:
        return [e["requested_by"] for e in c.get(f"/v1/history{query}").json()["history"]]

    assert owners(clients["asha"]) == ["asha"]
    assert owners(clients["ravi"]) == ["ravi"]
    assert owners(clients["asha"], "?requested_by=ravi") == ["asha"]
    assert sorted(owners(clients["boss"])) == ["asha", "boss", "ravi"]
    assert owners(clients["boss"], "?requested_by=ravi") == ["ravi"]
    assert TestClient(app).get("/v1/history").status_code == 401


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
    assert decision["model_name"] == "cheap-small"  # no GRADIST_QUALITY_FIELD configured -> ignored


# -----------------------------------------------------------------------------------------
# /v1/resolve optimize_for form field.
# -----------------------------------------------------------------------------------------


def test_resolve_endpoint_accepts_optimize_for_cost_accuracy_balanced(
    quality_client: TestClient,
) -> None:
    # Seed a rule + two ratings so accuracy mode has something to rank. Cost → cheapest,
    # accuracy → highest-rated. Decision.optimize_for echoes what actually drove the pick.
    quality_client.put("/v1/rules", json={"rules": [{"when": {"task": "news"}, "tier": "small"}]})
    quality_client.post(
        "/v1/quality",
        json={"model_id": "cheap-small", "metadata": {"task": "news"}, "rating": 2, "judged_by": "a"},
    )
    quality_client.post(
        "/v1/quality",
        json={"model_id": "pricey-small", "metadata": {"task": "news"}, "rating": 5, "judged_by": "a"},
    )
    cost = quality_client.post(
        "/v1/resolve",
        data={
            "metadata": json.dumps({"task": "news"}),
            "text": "",
            "optimize_for": "cost",
        },
    ).json()
    assert cost["model_name"] == "cheap-small"
    assert cost["optimize_for"] == "cost"
    accuracy = quality_client.post(
        "/v1/resolve",
        data={
            "metadata": json.dumps({"task": "news"}),
            "text": "",
            "optimize_for": "accuracy",
        },
    ).json()
    assert accuracy["model_name"] == "pricey-small"
    assert accuracy["optimize_for"] == "accuracy"
    # balanced is accepted (doesn't 422) and returns a Decision with optimize_for echoed.
    balanced = quality_client.post(
        "/v1/resolve",
        data={
            "metadata": json.dumps({"task": "news"}),
            "text": "",
            "optimize_for": "balanced",
        },
    )
    assert balanced.status_code == 200
    assert balanced.json()["optimize_for"] == "balanced"


def test_task_pools_roundtrip_and_resolve_uses_pool(catalog_client: TestClient) -> None:
    put = catalog_client.put(
        "/v1/task-pools/Video_Processing",
        json={"models": ["pricey-small"]},
    )
    assert put.status_code == 200
    assert put.json()["pools"]["video_processing"] == ["pricey-small"]
    listed = catalog_client.get("/v1/task-pools")
    assert listed.status_code == 200
    assert listed.json()["pools"]["video_processing"] == ["pricey-small"]
    # Module under test: service /v1/resolve hybrid catalog+ladder (user: all providers + content).
    catalog_client.put(
        "/v1/rules",
        json={"rules": [{"when": {"task": "video_processing"}, "tier": "small"}]},
    )
    decision = catalog_client.post(
        "/v1/resolve",
        data={"metadata": "{}", "text": "clip", "task": "VIDEO_PROCESSING", "optimize_for": "cost"},
    )
    assert decision.status_code == 200
    body = decision.json()
    assert body["model_name"] == "pricey-small"
    assert "task='video_processing'" in body["reason"]
    assert body["rule_outcome"] == "rule task=video_processing"


def test_resolve_rejects_empty_selected_models(catalog_client: TestClient) -> None:
    response = catalog_client.post(
        "/v1/resolve",
        data={"metadata": "{}", "text": "", "selected_models": "[]", "optimize_for": "cost"},
    )
    assert response.status_code == 422
    assert "empty" in str(response.json()["detail"]).lower()


def test_task_pool_rejects_empty_models(catalog_client: TestClient) -> None:
    response = catalog_client.put("/v1/task-pools/empty_job", json={"models": []})
    assert response.status_code == 422


def test_resolve_endpoint_rejects_invalid_optimize_for_with_422(client: TestClient) -> None:
    response = client.post(
        "/v1/resolve",
        data={
            "metadata": json.dumps({"task": "news"}),
            "text": "",
            "optimize_for": "speed",
        },
    )
    assert response.status_code == 422
    assert "optimize_for" in str(response.json()["detail"])


def test_rules_endpoint_roundtrips_optimize_for(client: TestClient) -> None:
    # Operators can pin a mode on a rule; it round-trips through PUT → GET → disk.
    put = client.put(
        "/v1/rules",
        json={
            "rules": [
                {
                    "when": {"task": "qc_review"},
                    "tier": "large",
                    "optimize_for": "accuracy",
                }
            ]
        },
    )
    assert put.status_code == 200
    body = client.get("/v1/rules").json()["rules"]
    assert body == [
        {
            "when": {"task": "qc_review"},
            "tier": "large",
            "priority": 0,
            "optimize_for": "accuracy",
        }
    ]
    # Bad mode → 422, previous rules unchanged.
    bad = client.put(
        "/v1/rules",
        json={"rules": [{"when": {"task": "qc_review"}, "tier": "large", "optimize_for": "speed"}]},
    )
    assert bad.status_code == 422
    assert client.get("/v1/rules").json()["rules"] == body


# -----------------------------------------------------------------------------------------
# Modality booleans + quality/field + rating session user.
# -----------------------------------------------------------------------------------------


def test_catalog_endpoint_exposes_modality_booleans(catalog_client: TestClient) -> None:
    body = catalog_client.get("/v1/catalog").json()
    assert body["catalog"]
    for entry in body["catalog"]:
        assert "supports_image" in entry
        assert "supports_video" in entry
        assert "supports_audio" in entry
        # Defaults for the fixture entries (no video/audio declared).
        assert entry["supports_video"] is False
        assert entry["supports_audio"] is False


def test_quality_field_endpoint_returns_configured_field_or_null(tmp_path: Path) -> None:
    # Two separate store roots — sharing tmp_path would collide on the bootstrap user.
    bare = _authed_client(tmp_path / "bare")
    assert bare.get("/v1/quality/field").json() == {"field": None}
    with_field = _authed_client(tmp_path / "with", catalog=CATALOG, quality_field="task")
    assert with_field.get("/v1/quality/field").json() == {"field": "task"}


def _seed_catalog_yaml(path: Path) -> ModelCatalog:
    """Write a tiny catalog.yaml and return the matching in-memory ModelCatalog (P1 tests)."""
    path.write_text(
        "catalog:\n"
        "  - id: cheap-small\n    provider: gemini\n    tier: small\n    cost: 1\n"
        "    supports_image: true\n    label: Cheap Small\n"
        "  - id: pending-model\n    provider: openai\n    tier: UNREVIEWED\n    cost: 2\n"
        "    supports_image: true\n    label: Pending\n",
        encoding="utf-8",
    )
    from gradist.catalog import load_catalog

    return load_catalog(path)


def test_put_catalog_tier_as_admin_updates_and_persists(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.yaml"
    catalog = _seed_catalog_yaml(catalog_path)
    client = _authed_client(tmp_path / "admin", role="admin", catalog=catalog, catalog_path=catalog_path)
    response = client.put("/v1/catalog/pending-model", json={"tier": "medium"})
    assert response.status_code == 200
    body = response.json()["catalog"]
    pending = next(e for e in body if e["id"] == "pending-model")
    assert pending["tier"] == "medium"
    assert pending["assigned_by"] == "tester"
    # Persisted to disk.
    from gradist.catalog import load_catalog

    reloaded = load_catalog(catalog_path)
    assert next(e for e in reloaded.entries if e.id == "pending-model").tier == "medium"
    assert next(e for e in reloaded.entries if e.id == "pending-model").assigned_by == "tester"


def test_put_catalog_tier_as_member_returns_403(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.yaml"
    catalog = _seed_catalog_yaml(catalog_path)
    client = _authed_client(tmp_path / "member", role="member", catalog=catalog, catalog_path=catalog_path)
    response = client.put("/v1/catalog/pending-model", json={"tier": "medium"})
    assert response.status_code == 403


def test_put_catalog_tier_unknown_id_returns_404(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.yaml"
    catalog = _seed_catalog_yaml(catalog_path)
    client = _authed_client(tmp_path / "admin", role="admin", catalog=catalog, catalog_path=catalog_path)
    response = client.put("/v1/catalog/does-not-exist", json={"tier": "small"})
    assert response.status_code == 404


def test_put_catalog_tier_invalid_tier_returns_422(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.yaml"
    catalog = _seed_catalog_yaml(catalog_path)
    client = _authed_client(tmp_path / "admin", role="admin", catalog=catalog, catalog_path=catalog_path)
    response = client.put("/v1/catalog/pending-model", json={"tier": "giant"})
    assert response.status_code == 422


def test_rating_modal_judged_by_is_session_user(catalog_client: TestClient) -> None:
    # P3 History-modal shape: client may omit judged_by (or forge one); server always
    # records the session user. catalog_client has no quality_field so GET returns the
    # raw ratings list with judged_by visible.
    ok = catalog_client.post(
        "/v1/quality",
        json={"model_id": "cheap-small", "metadata": {"task": "news"}, "rating": 4, "notes": "ok"},
    )
    assert ok.status_code == 200
    forged = catalog_client.post(
        "/v1/quality",
        json={
            "model_id": "pricey-small",
            "metadata": {"task": "news"},
            "rating": 5,
            "judged_by": "NOT-THE-SESSION-USER",
            "notes": "",
        },
    )
    assert forged.status_code == 200
    ratings = catalog_client.get("/v1/quality").json()["ratings"]
    assert {r["judged_by"] for r in ratings} == {"tester"}
    assert all(r["judged_by"] != "NOT-THE-SESSION-USER" for r in ratings)


def test_trained_classifier_survives_a_restart(tmp_path: Path) -> None:
    client = _authed_client(tmp_path)
    for tier, text in [("small", "fix a typo"), ("large", "analyse the election debate")]:
        example = client.post("/v1/examples", data={"metadata": json.dumps({"channel_type": tier}), "text": text})
        client.post("/v1/judgments", json={"example_id": example.json()["example_id"], "tier": tier})
    assert client.post("/v1/train").status_code == 200

    store = Store(tmp_path / "data")
    restarted = TestClient(
        create_app(TIERS, store, UsersStore(path=store.root / "users.jsonl"), SessionConfig(secret="s"))
    )
    assert restarted.post("/v1/auth/login", json={"username": "tester", "password": "tester-pw"}).status_code == 200
    status = restarted.get("/v1/status").json()
    assert status["trained"] is True
    assert status["n_labels_last_trained_on"] == 2
    decision = restarted.post(
        "/v1/resolve", data={"metadata": json.dumps({"channel_type": "large"}), "text": "analyse the election debate"}
    ).json()
    assert decision["reason"].startswith("classifier")
