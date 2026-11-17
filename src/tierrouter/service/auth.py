# [Fact-Forcing preamble]
#
# The file(s) and line(s) that will call it:
#   - `src/tierrouter/service/app.py` — `SessionConfig` is used by create_app's
#     SessionMiddleware wire-up; `hash_password` + `verify_password` by the `/v1/auth/login`
#     route; `require_user` + `require_admin` as `Depends(...)` on every mutating route;
#     `AuthError` is caught by `/v1/auth/login` and translated to HTTP 401.
#   - `src/tierrouter/service/bootstrap.py` — `hash_password` is called to write the first
#     admin's password hash into users.jsonl.
#
# No existing file serves the same purpose: confirmed by reading every .py under
# `src/tierrouter/` end-to-end. `service/store.py` manages examples/judgments/rules/history/
# quality but has no notion of users or passwords.
#
# The data schemas it reads or writes:
#   - Reads the User record (from `users_store.py`): {id, username, password_hash, role,
#     created_at, disabled, notes}.
#   - Reads `request.session["user_id"]` (string) set by `SessionMiddleware` via the login
#     route.
#   - Writes nothing directly; mutations happen in `users_store.py`.
#
# No secrets stored in plaintext: bcrypt hashes only (`passlib.context.CryptContext` with
# `schemes=["bcrypt"]`). The session signing secret comes from the
# TIERROUTER_SESSION_SECRET env var and is never persisted by this module.
#
# The user's verbatim instruction driving this file: "create auth or required things because
# maybe user have different workflow and they can create difference judgments" — see
# PLAN-v2.md §4 (P0 — Auth, Level B).

"""Password hashing, session loader, and FastAPI dependencies.

The session cookie is signed by `starlette.middleware.sessions.SessionMiddleware`; we only
read `request.session["user_id"]` here. If the cookie is missing or the referenced user is
gone/disabled, `require_user` raises 401 — never a 500.

bcrypt is the chosen hash — used directly (not via passlib). passlib 1.7.4 has a known
incompatibility with bcrypt>=4; one scheme and one small dep is clearer.

bcrypt has a 72-byte input limit. We enforce it explicitly before hashing — rejecting rather
than silently truncating — because a silent truncation would make two different passwords
with the same first 72 bytes compare equal.
"""

from __future__ import annotations

from dataclasses import dataclass

import bcrypt
from fastapi import Depends, HTTPException, Request, status

from tierrouter.service.users_store import User, UsersStore

_MAX_PASSWORD_BYTES = 72


class AuthError(Exception):
    """Login failed — wrong password, no such user, or user disabled. Caller translates to
    HTTP 401; we don't raise HTTPException directly so this module stays usable from CLI
    bootstrap too."""


def hash_password(plain: str) -> str:
    if not plain:
        raise AuthError("password must not be empty")
    encoded = plain.encode("utf-8")
    if len(encoded) > _MAX_PASSWORD_BYTES:
        raise AuthError(
            f"password must be at most {_MAX_PASSWORD_BYTES} bytes once UTF-8 encoded "
            "(bcrypt's hard limit). Choose a shorter one."
        )
    return bcrypt.hashpw(encoded, bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    encoded = plain.encode("utf-8")
    if len(encoded) > _MAX_PASSWORD_BYTES:
        return False
    try:
        return bcrypt.checkpw(encoded, hashed.encode("utf-8"))
    except (ValueError, TypeError):
        # A malformed stored hash should log in as "wrong password", never crash.
        return False


@dataclass(frozen=True)
class SessionConfig:
    """Pulled out so `service/main.py` can enforce the env-var-required rule in one place
    and tests can inject a known secret."""

    secret: str
    cookie_name: str = "tierrouter_session"
    max_age_seconds: int = 60 * 60 * 24 * 7  # one week; shorter would be more paranoid


def login_user(users: UsersStore, username: str, password: str) -> User:
    """Return the user on success, raise AuthError on any failure. All failure messages are
    deliberately identical (`"invalid credentials"`) to avoid disclosing which part was
    wrong; `disabled` is the one exception so an operator knows to contact the admin."""
    user = users.find_by_username(username)
    if user is None:
        raise AuthError("invalid credentials")
    if not verify_password(password, user.password_hash):
        raise AuthError("invalid credentials")
    if user.disabled:
        raise AuthError("account disabled; contact an admin")
    return user


def _current_user(request: Request) -> User | None:
    users: UsersStore | None = getattr(request.app.state, "users", None)
    if users is None:
        return None
    user_id = request.session.get("user_id") if hasattr(request, "session") else None
    if not isinstance(user_id, str) or not user_id:
        return None
    user = users.find_by_id(user_id)
    if user is None or user.disabled:
        # A session for a user that was deleted/disabled is equivalent to no session. The
        # client will get a 401 on the next protected call and bounce to /login.
        return None
    return user


def require_user(request: Request) -> User:
    user = _current_user(request)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="authentication required"
        )
    return user


def require_admin(user: User = Depends(require_user)) -> User:
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="admin role required"
        )
    return user


def optional_user(request: Request) -> User | None:
    """For read endpoints that may be relaxed via `TIERROUTER_ALLOW_READ_WITHOUT_AUTH=1`:
    returns the current user if there is one, else None, never raises. The route decides
    what to do with None."""
    return _current_user(request)
