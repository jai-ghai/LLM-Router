# [Fact-Forcing preamble]
#
# The file(s) and line(s) that will call it: nothing in the package imports this file. It is
# a CLI entry point, run as `python -m gradist.service.bootstrap ...`.
#
# No existing file serves the same purpose: confirmed by Glob src/gradist/**/*.py and
# experiments/**/*.py. No admin-creation script exists.
#
# The data schemas it reads or writes: writes a User record (via
# `users_store.UsersStore.add` or `set_password_hash`) — same schema as users_store.py.
# No new schema.
#
# No secrets stored in plaintext: the password is read via `getpass.getpass` (never
# echoed), hashed via `auth.hash_password` (bcrypt), and only the hash lands on disk.
# GRADIST_SESSION_SECRET is not touched here.
#
# The user's verbatim instruction driving this file: "create auth or required things because
# maybe user have different workflow and they can create difference judgments" — PLAN-v2.md
# §4.1 names `python -m gradist.service.bootstrap --username <name>` as the gateway for
# creating the first admin before the service accepts any login.

"""CLI: create the first admin user, or reset an existing user's password.

Usage:
    python -m gradist.service.bootstrap --username <name> [--role admin|member]
    python -m gradist.service.bootstrap --reset <username>

Reads GRADIST_DATA (default `./data`) for the users.jsonl location; mirrors the service's
own env-var convention from service/main.py.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

from gradist.service.auth import AuthError, hash_password
from gradist.service.users_store import UserError, UsersStore


def _prompt_password(confirm: bool = True) -> str:
    """Read a password without echoing. If `confirm` is set, read it twice and compare —
    typos on the first-admin password are unrecoverable otherwise."""
    password = getpass.getpass("Password: ")
    if confirm:
        again = getpass.getpass("Confirm:  ")
        if password != again:
            raise AuthError("passwords did not match")
    return password


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=None, help="store directory (default: GRADIST_DATA or ./data)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--username", help="create a new user")
    group.add_argument("--reset", metavar="USERNAME", help="reset an existing user's password")
    parser.add_argument("--role", choices=("admin", "member"), default="admin",
                        help="role for a newly created user (default: admin)")
    parser.add_argument("--notes", default="", help="free-form notes attached to the user")
    args = parser.parse_args(argv)

    data_dir = args.data or Path(os.environ.get("GRADIST_DATA") or "data")
    users_path = data_dir / "users.jsonl"
    users = UsersStore(path=users_path)

    try:
        if args.reset:
            user = users.find_by_username(args.reset)
            if user is None:
                print(f"no user named '{args.reset}'", file=sys.stderr)
                return 1
            password = _prompt_password(confirm=True)
            users.set_password_hash(user.id, hash_password(password))
            print(f"reset password for '{user.username}' ({user.id})")
            return 0

        # --username: create
        if users.find_by_username(args.username) is not None:
            print(f"username '{args.username}' already exists", file=sys.stderr)
            return 1
        if args.role != "admin" and users.is_empty():
            print(
                "first user must be an admin (so someone can create more users); "
                "re-run without --role=member",
                file=sys.stderr,
            )
            return 1
        password = _prompt_password(confirm=True)
        created = users.add(args.username, hash_password(password), args.role, notes=args.notes)
        print(f"created user '{created.username}' ({created.id}) with role '{created.role}'")
        print(f"users.jsonl: {users_path}")
        return 0
    except (AuthError, UserError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
