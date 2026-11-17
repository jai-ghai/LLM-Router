# [Fact-Forcing preamble]
#
# Behaviour and module under test: `UsersStore` methods (`add`, `set_disabled`,
# `set_password_hash`, `find_by_username`, `find_by_id`, `is_empty`) in
# `src/tierrouter/service/users_store.py`, and the three `/v1/users*` admin routes in
# `src/tierrouter/service/app.py`.
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
from tierrouter.service.auth import SessionConfig, hash_password
from tierrouter.service.store import Store
from tierrouter.service.users_store import UserError, UsersStore
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


def _app_with_user(tmp_path: Path, *, role: str = "admin") -> tuple[TestClient, UsersStore]:
    store = Store(tmp_path / "data")
    users = UsersStore(path=store.root / "users.jsonl")
    users.add("admin", hash_password("pw"), role)  # type: ignore[arg-type]
    app = create_app(TIERS, store, users, SessionConfig(secret="test-secret"))
    client = TestClient(app)
    login = client.post("/v1/auth/login", json={"username": "admin", "password": "pw"})
    assert login.status_code == 200
    return client, users


# ---- UsersStore unit behaviour -------------------------------------------------------


def test_add_rejects_duplicate_username(tmp_path: Path) -> None:
    users = UsersStore(path=tmp_path / "users.jsonl")
    users.add("admin", hash_password("pw"), "admin")
    with pytest.raises(UserError):
        users.add("admin", hash_password("pw2"), "member")


def test_add_rejects_blank_username(tmp_path: Path) -> None:
    users = UsersStore(path=tmp_path / "users.jsonl")
    with pytest.raises(UserError):
        users.add("", hash_password("pw"), "admin")


def test_add_rejects_unknown_role(tmp_path: Path) -> None:
    users = UsersStore(path=tmp_path / "users.jsonl")
    with pytest.raises(UserError):
        users.add("a", hash_password("pw"), "root")  # type: ignore[arg-type]


def test_disable_last_admin_is_rejected(tmp_path: Path) -> None:
    users = UsersStore(path=tmp_path / "users.jsonl")
    only = users.add("admin", hash_password("pw"), "admin")
    with pytest.raises(UserError):
        users.set_disabled(only.id, True)
    # Still active.
    found = users.find_by_username("admin")
    assert found is not None
    assert found.disabled is False


def test_disable_one_admin_when_another_exists(tmp_path: Path) -> None:
    users = UsersStore(path=tmp_path / "users.jsonl")
    a = users.add("a", hash_password("pw"), "admin")
    users.add("b", hash_password("pw"), "admin")
    updated = users.set_disabled(a.id, True)
    assert updated.disabled is True


def test_store_round_trips_through_file(tmp_path: Path) -> None:
    path = tmp_path / "users.jsonl"
    users_one = UsersStore(path=path)
    created = users_one.add("jai", hash_password("pw"), "admin", notes="first")
    users_two = UsersStore(path=path)  # fresh instance reads same records back
    found = users_two.find_by_id(created.id)
    assert found is not None
    assert found.username == "jai"
    assert found.role == "admin"
    assert found.notes == "first"


# ---- HTTP: /v1/users (admin) ---------------------------------------------------------


def test_only_admin_can_list_users(tmp_path: Path) -> None:
    client, _ = _app_with_user(tmp_path, role="admin")
    client.post("/v1/users", json={"username": "mem", "password": "pw", "role": "member"})
    client.post("/v1/auth/logout")
    client.post("/v1/auth/login", json={"username": "mem", "password": "pw"})
    assert client.get("/v1/users").status_code == 403


def test_only_admin_can_create_users(tmp_path: Path) -> None:
    client, _ = _app_with_user(tmp_path, role="admin")
    created = client.post("/v1/users", json={"username": "newbie", "password": "pw", "role": "member"})
    assert created.status_code == 200
    assert created.json()["user"]["username"] == "newbie"
    client.post("/v1/auth/logout")
    client.post("/v1/auth/login", json={"username": "newbie", "password": "pw"})
    forbidden = client.post("/v1/users", json={"username": "x", "password": "pw", "role": "member"})
    assert forbidden.status_code == 403


def test_create_user_rejects_bad_role(tmp_path: Path) -> None:
    client, _ = _app_with_user(tmp_path, role="admin")
    bad = client.post("/v1/users", json={"username": "x", "password": "pw", "role": "root"})
    assert bad.status_code == 422


def test_create_user_rejects_duplicate_username(tmp_path: Path) -> None:
    client, _ = _app_with_user(tmp_path, role="admin")
    dup = client.post("/v1/users", json={"username": "admin", "password": "pw", "role": "member"})
    assert dup.status_code == 422


def test_put_user_toggles_disabled(tmp_path: Path) -> None:
    client, _ = _app_with_user(tmp_path, role="admin")
    created = client.post("/v1/users", json={"username": "m", "password": "pw", "role": "member"}).json()
    user_id = created["user"]["id"]
    disabled = client.put(f"/v1/users/{user_id}", json={"disabled": True})
    assert disabled.status_code == 200
    assert disabled.json()["user"]["disabled"] is True
    enabled = client.put(f"/v1/users/{user_id}", json={"disabled": False})
    assert enabled.status_code == 200
    assert enabled.json()["user"]["disabled"] is False


def test_put_user_unknown_id_returns_404(tmp_path: Path) -> None:
    client, _ = _app_with_user(tmp_path, role="admin")
    resp = client.put("/v1/users/usr-ghost", json={"disabled": True})
    assert resp.status_code == 404


def test_disable_last_admin_via_api_returns_422(tmp_path: Path) -> None:
    client, users = _app_with_user(tmp_path, role="admin")
    only = users.find_by_username("admin")
    assert only is not None
    resp = client.put(f"/v1/users/{only.id}", json={"disabled": True})
    assert resp.status_code == 422
    assert "last active admin" in resp.json()["detail"]
