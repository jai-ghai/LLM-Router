# [Fact-Forcing preamble]
# Callers: run manually (`python experiments/inspect_tabson_db.py`) and from
# `experiments/train_from_tabson_db.py` (next) — no production import path yet.
# Existing file search: no inspect_tabson_db / train_from_tabson_db under experiments/
# (Glob experiments/*.py). Purpose: read-only MySQL explore of gemini_call_telemetry.
# Data shape (synthetic): {called_at_utc, pipeline_name, purpose_name, model_name, tier_name,
# latency_ms} — dates ISO UTC from DB. User instruction: "okay ..i am sharing the database
# details - check in env - the read only access in your router and train model and complete
# router end to end .."
"""Read-only inspection of tabsonsdb for Gradist training sources."""

from __future__ import annotations

from pathlib import Path

import pymysql

ROOT = Path(__file__).resolve().parents[1]


def _load_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or raw.startswith("//") or "=" not in raw:
            continue
        key, _, value = raw.partition("=")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def connect() -> pymysql.connections.Connection:
    env = _load_env(ROOT / ".env")
    return pymysql.connect(
        host=env["MYSQL_HOST"],
        user=env["MYSQL_USER"],
        password=env["MYSQL_PASSWORD"],
        database=env["MYSQL_DATABASE"],
        port=int(env.get("MYSQL_PORT", "3306") or "3306"),
        connect_timeout=20,
        read_timeout=120,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def main() -> None:
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT USER() AS db_user, CURRENT_USER() AS db_current, VERSION() AS db_version")
    print("session:", cur.fetchone())

    cur.execute("SHOW TABLES")
    tables = [next(iter(row.values())) for row in cur.fetchall()]
    interesting = [t for t in tables if any(k in t.lower() for k in ("gemini", "telemetry", "model", "infer"))]
    print("interesting tables:", interesting)

    for table in interesting:
        cur.execute(f"SHOW COLUMNS FROM `{table}`")
        cols = [r["Field"] for r in cur.fetchall()]
        cur.execute(f"SELECT COUNT(*) AS n FROM `{table}`")
        n = cur.fetchone()["n"]
        print(f"\n== {table} ({n} rows) ==")
        print(" cols:", ", ".join(cols[:50]))

    for view in ("v_gemini_call", "gemini_call_telemetry"):
        if view not in tables:
            continue
        print(f"\n-- sample from {view} --")
        try:
            cur.execute(f"SELECT * FROM `{view}` ORDER BY 1 DESC LIMIT 3")
            for row in cur.fetchall():
                keys = list(row.keys())[:14]
                print({k: row[k] for k in keys})
        except Exception as exc:  # noqa: BLE001
            print("sample failed:", exc)

    conn.close()


if __name__ == "__main__":
    main()
