# [Fact-Forcing preamble]
# Callers: `src/tierrouter/service/auth.py` (functions `login_user`, `require_user`),
#          `src/tierrouter/service/app.py` (routes `/v1/auth/*`, `/v1/users*`),
#          `src/tierrouter/service/bootstrap.py` (CLI `main` function).
# Duplicates: no existing file manages users. `service/store.py` manages examples/judgments/
# rules/history/quality but not users — confirmed by reading service/store.py end-to-end.
# Schema written: JSONL, one record per line: {id, username, password_hash, role,
# created_at, disabled, notes}. Fully rewritten on each mutation (users are tiny; rewrite is
# O(users), not O(requests)). Schema read: same JSONL on service start / each login lookup.
# No secrets stored in plaintext: password_hash is the bcrypt output from passlib (see
# auth.py). TIERROUTER_SESSION_SECRET is never persisted in this file.
# Driven by PLAN-v2.md §4 (P0 — Auth, Level B). Operator's verbatim ask: "create auth or
# required things because maybe user have different workflow and they can create difference
# judgments".

"""Append-only-with-rewrite users store, file-backed as users.jsonl.

See the preamble above for callers and schema. This docstring covers behaviour only.

Tiny file by construction (one row per operator), not hot-path. Loaded lazily and cached;
mutations rewrite the whole file. Not thread-safe — the service is single-worker uvicorn
(see service/main.py); if a future deployment adds workers, wrap the write methods with a
lock here, not at every call site.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

Role = Literal["admin", "member"]
_VALID_ROLES: tuple[Role, ...] = ("admin", "member")


class UserError(ValueError):
    """A user record is malformed or a mutation would break a constraint (unique username,
    admin remains after a disable, etc.). Validated here, not just at the API boundary, so a
    bad write can never quietly land in users.jsonl."""


@dataclass(frozen=True)
class User:
    id: str
    username: str
    password_hash: str
    role: Role
    created_at: str
    disabled: bool = False
    notes: str = ""

    def to_record(self) -> dict[str, object]:
        return {
            "id": self.id,
            "username": self.username,
            "password_hash": self.password_hash,
            "role": self.role,
            "created_at": self.created_at,
            "disabled": self.disabled,
            "notes": self.notes,
        }

    def public(self) -> dict[str, object]:
        """What the API is allowed to return — never the password hash."""
        return {
            "id": self.id,
            "username": self.username,
            "role": self.role,
            "created_at": self.created_at,
            "disabled": self.disabled,
            "notes": self.notes,
        }


@dataclass
class UsersStore:
    path: Path
    _cache: list[User] = field(default_factory=list, init=False)
    _loaded: bool = field(default=False, init=False)

    def _load(self) -> None:
        if self._loaded:
            return
        self._cache = []
        if self.path.exists() and self.path.stat().st_size > 0:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                raw = json.loads(line)
                try:
                    self._cache.append(_user_from_record(raw))
                except (KeyError, UserError):
                    # A malformed row should never crash every later read — same discipline
                    # as Store.list_quality_ratings.
                    continue
        self._loaded = True

    def _rewrite(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(u.to_record(), ensure_ascii=False) for u in self._cache]
        self.path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    # ---- reads ---------------------------------------------------------------------------

    def list_users(self) -> list[User]:
        self._load()
        return list(self._cache)

    def find_by_username(self, username: str) -> User | None:
        self._load()
        normalised = username.strip().lower()
        for user in self._cache:
            if user.username.lower() == normalised:
                return user
        return None

    def find_by_id(self, user_id: str) -> User | None:
        self._load()
        for user in self._cache:
            if user.id == user_id:
                return user
        return None

    def is_empty(self) -> bool:
        """True when no user has ever been created. The service refuses to accept logins in
        this state (see auth.py)."""
        self._load()
        return not self._cache

    # ---- writes --------------------------------------------------------------------------

    def add(
        self, username: str, password_hash: str, role: Role, *, notes: str = ""
    ) -> User:
        username = username.strip()
        if not username:
            raise UserError("username is required")
        if role not in _VALID_ROLES:
            raise UserError(f"role must be one of {_VALID_ROLES}, got {role!r}")
        self._load()
        if self.find_by_username(username) is not None:
            raise UserError(f"username '{username}' is already taken")
        user = User(
            id=f"usr-{uuid.uuid4().hex[:12]}",
            username=username,
            password_hash=password_hash,
            role=role,
            created_at=datetime.now(UTC).isoformat(),
            disabled=False,
            notes=notes,
        )
        self._cache.append(user)
        self._rewrite()
        return user

    def set_disabled(self, user_id: str, disabled: bool) -> User:
        self._load()
        for idx, user in enumerate(self._cache):
            if user.id == user_id:
                if disabled and user.role == "admin" and self._count_active_admins() <= 1:
                    raise UserError(
                        "cannot disable the last active admin — leaves the system with no one "
                        "able to re-enable users"
                    )
                updated = User(
                    id=user.id,
                    username=user.username,
                    password_hash=user.password_hash,
                    role=user.role,
                    created_at=user.created_at,
                    disabled=disabled,
                    notes=user.notes,
                )
                self._cache[idx] = updated
                self._rewrite()
                return updated
        raise UserError(f"no user with id '{user_id}'")

    def set_password_hash(self, user_id: str, password_hash: str) -> User:
        """Used by `bootstrap --reset`. Not exposed as a web endpoint in this phase — there is
        no password-reset flow with email; admins reset from the shell."""
        self._load()
        for idx, user in enumerate(self._cache):
            if user.id == user_id:
                updated = User(
                    id=user.id,
                    username=user.username,
                    password_hash=password_hash,
                    role=user.role,
                    created_at=user.created_at,
                    disabled=user.disabled,
                    notes=user.notes,
                )
                self._cache[idx] = updated
                self._rewrite()
                return updated
        raise UserError(f"no user with id '{user_id}'")

    def _count_active_admins(self) -> int:
        return sum(1 for u in self._cache if u.role == "admin" and not u.disabled)


def _user_from_record(raw: dict[str, object]) -> User:
    try:
        role = raw["role"]
    except KeyError as exc:
        raise UserError(f"users record missing 'role': {raw}") from exc
    if role not in _VALID_ROLES:
        raise UserError(f"users record has unknown role {role!r}: {raw}")
    return User(
        id=str(raw["id"]),
        username=str(raw["username"]),
        password_hash=str(raw["password_hash"]),
        role=role,
        created_at=str(raw["created_at"]),
        disabled=bool(raw.get("disabled", False)),
        notes=str(raw.get("notes") or ""),
    )
