# [Fact-Forcing preamble]
#
# Behaviour and module under test: `src/tierrouter/service/auth.py` (functions
# `hash_password`, `verify_password`, `login_user`, `require_user`, `require_admin`,
# `optional_user`) and the three `/v1/auth/*` routes plus session-enforcement on
# `/v1/rules` and `/v1/quality` defined in `src/tierrouter/service/app.py`.
#
# Existing test files for it: none — this file is new in P0.
#
# The user's verbatim instruction: "create auth or required things because maybe user have
# different workflow and they can create difference judgments" — PLAN-v2.md §4.

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tierrouter.service.app import create_app
from tierrouter.service.auth import AuthError, SessionConfig, hash_password, verify_password
from tierrouter.service.store import Store
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


def _app(tmp_path: Path, *, allow_read_without_auth: bool = False) -> tuple[TestClient, UsersStore]:
    store = Store(tmp_path / "data")
    users = UsersStore(path=store.root / "users.jsonl")
    app = create_app(
        TIERS,
        store,
        users,
        SessionConfig(secret="test-secret"),
        allow_read_without_auth=allow_read_without_auth,
    )
    return TestClient(app), users


# ---- unit-level ----------------------------------------------------------------------


def test_hash_password_rejects_empty() -> None:
    with pytest.raises(AuthError):
        hash_password("")


def test_hash_password_rejects_over_72_bytes() -> None:
    """bcrypt's hard limit — enforced in auth.py rather than silently truncating, because
    truncation would make two different passwords with the same first 72 bytes equivalent."""
    with pytest.raises(AuthError):
        hash_password("x" * 73)


def test_verify_password_rejects_wrong_input() -> None:
    hashed = hash_password("correct-horse")
    assert verify_password("correct-horse", hashed) is True
    assert verify_password("wrong", hashed) is False
    assert verify_password("", hashed) is False
    # A garbage stored hash must not crash verify_password.
    assert verify_password("anything", "not-a-real-bcrypt-hash") is False


# ---- HTTP: login / logout / me -------------------------------------------------------


def test_service_refuses_login_before_bootstrap(tmp_path: Path) -> None:
    client, _ = _app(tmp_path)
    resp = client.post("/v1/auth/login", json={"username": "any", "password": "any"})
    assert resp.status_code == 503, resp.text
    assert "no users exist yet" in resp.json()["detail"]


def test_login_with_correct_password_sets_session_cookie(tmp_path: Path) -> None:
    client, users = _app(tmp_path)
    users.add("admin", hash_password("super-secret"), "admin")
    resp = client.post("/v1/auth/login", json={"username": "admin", "password": "super-secret"})
    assert resp.status_code == 200
    assert resp.json()["user"]["username"] == "admin"
    # Follow-up /v1/auth/me now succeeds — session cookie was set.
    me = client.get("/v1/auth/me")
    assert me.status_code == 200
    assert me.json()["user"]["username"] == "admin"


def test_login_with_wrong_password_returns_401(tmp_path: Path) -> None:
    client, users = _app(tmp_path)
    users.add("admin", hash_password("super-secret"), "admin")
    resp = client.post("/v1/auth/login", json={"username": "admin", "password": "nope"})
    assert resp.status_code == 401
    assert resp.json()["detail"] == "invalid credentials"


def test_login_with_unknown_username_returns_same_401(tmp_path: Path) -> None:
    """Identical error message as wrong-password — don't disclose whether the username exists."""
    client, users = _app(tmp_path)
    users.add("admin", hash_password("super-secret"), "admin")
    resp = client.post("/v1/auth/login", json={"username": "ghost", "password": "anything"})
    assert resp.status_code == 401
    assert resp.json()["detail"] == "invalid credentials"


def test_login_with_disabled_user_returns_401(tmp_path: Path) -> None:
    client, users = _app(tmp_path)
    second = users.add("other", hash_password("pw"), "member")
    users.add("admin", hash_password("pw"), "admin")  # keep an active admin
    users.set_disabled(second.id, True)
    resp = client.post("/v1/auth/login", json={"username": "other", "password": "pw"})
    assert resp.status_code == 401
    assert "disabled" in resp.json()["detail"]


def test_logout_clears_session(tmp_path: Path) -> None:
    client, users = _app(tmp_path)
    users.add("admin", hash_password("pw"), "admin")
    client.post("/v1/auth/login", json={"username": "admin", "password": "pw"})
    assert client.get("/v1/auth/me").status_code == 200
    client.post("/v1/auth/logout")
    assert client.get("/v1/auth/me").status_code == 401


def test_me_returns_401_without_session(tmp_path: Path) -> None:
    client, _ = _app(tmp_path)
    assert client.get("/v1/auth/me").status_code == 401


# ---- authorisation on mutating + read routes -----------------------------------------


def test_put_rules_without_session_returns_401(tmp_path: Path) -> None:
    client, _ = _app(tmp_path)
    resp = client.put("/v1/rules", json={"rules": []})
    assert resp.status_code == 401


def test_post_quality_without_session_returns_401(tmp_path: Path) -> None:
    client, _ = _app(tmp_path)
    resp = client.post(
        "/v1/quality",
        json={"model_id": "m", "metadata": {}, "rating": 3, "judged_by": "x"},
    )
    assert resp.status_code == 401


def test_post_quality_ignores_client_supplied_judged_by_and_uses_session(tmp_path: Path) -> None:
    """Even if a client POSTs a `judged_by` field, the server overrides it with the session
    user's username. Regression guard against impersonation via form fields."""
    client, users = _app(tmp_path)
    users.add("real-user", hash_password("pw"), "member")
    client.post("/v1/auth/login", json={"username": "real-user", "password": "pw"})
    client.post(
        "/v1/quality",
        json={
            "model_id": "some-model",
            "metadata": {"task": "video_qc"},
            "rating": 4,
            "judged_by": "ADMIN-IMPERSONATION",  # must be ignored
        },
    )
    listed = client.get("/v1/quality").json()["ratings"]
    assert len(listed) == 1
    assert listed[0]["judged_by"] == "real-user"  # NOT "ADMIN-IMPERSONATION"


def test_get_rules_requires_auth_by_default(tmp_path: Path) -> None:
    client, _ = _app(tmp_path)
    assert client.get("/v1/rules").status_code == 401


def test_get_rules_allowed_when_relaxed_env_set(tmp_path: Path) -> None:
    client, _ = _app(tmp_path, allow_read_without_auth=True)
    assert client.get("/v1/rules").status_code == 200


def test_healthz_is_always_public(tmp_path: Path) -> None:
    client, _ = _app(tmp_path)
    assert client.get("/healthz").json() == {"status": "ok"}
