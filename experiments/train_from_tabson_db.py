# [Fact-Forcing preamble]
# Callers: CLI only — `python experiments/train_from_tabson_db.py` (no in-repo import yet).
# Existing files: experiments/inspect_tabson_db.py (schema), experiments/train_classifier.py
# (BARC path) — neither reads tabsonsdb MySQL. This file is the Tabson→Gradist trainer.
# Writes data/examples.jsonl {example_id, metadata, text, image_path},
# data/judgments.jsonl {example_id, tier, judged_by, notes},
# data/rules.yaml {rules:[{when,tier,priority}]}, data/classifier.joblib,
# config/tiers.tabson.yaml. Reads v_gemini_call (called_at_utc, pipeline, purpose, model,
# channel_code, status). User: "okay ..i am sharing the database details - check in env -
# the read only access in your router and train model and complete router end to end .."
"""Read-only pull from tabsonsdb → Gradist ladder training set + fit classifier."""

from __future__ import annotations

import json
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import pymysql
import yaml

from gradist.classifier import TierClassifier, evaluate_classifier
from gradist.features import HashingEmbedder, MetadataEncoder
from gradist.labels import TierLabel
from gradist.resolve import resolve
from gradist.rules import Rule, RuleSet
from gradist.tiers import load_tiers

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT_TIERS = ROOT / "config" / "tiers.tabson.yaml"

# Observed in tabsonsdb.gemini_model + tabson-python-inferencing defaults.
# flash-lite helpers → small; 3.5-flash-lite mid; flash-preview = STAGE1/2 default → large.
MODEL_TO_TIER: dict[str, str] = {
    "gemini-3.1-flash-lite": "small",
    "gemini-3.5-flash-lite": "medium",
    "gemini-2.5-flash-lite": "small",
    "gemini-3-flash-preview": "large",
    "gemini-2.5-flash": "medium",
    "gemini-3.5-flash": "large",
    "gemini-3.1-pro-preview": "large",
    "gemini-3-pro-preview": "large",
}

# Code defaults from tabson-python-inferencing (orchestrator / client / agentic_qc).
# Used as Gradist rules when telemetry is skewed toward one model.
PURPOSE_CODE_DEFAULTS: dict[str, str] = {
    # STAGE1 / STAGE2 / MODEL_ID default → gemini-3-flash-preview
    "v3_single": "large",
    "v3_retry": "large",
    "v3_cont": "large",
    "window_v3": "large",
    "single_segment": "large",
    "playout_observation": "medium",
    "script_repair": "large",
    # Helper / enrichment paths → flash-lite (gemini_helper / HEADLINE_*)
    "headline_similarity": "small",
    "headline_category": "small",
    "hinglish_batch": "small",
    "spell_batch": "small",
    "carry_over_similarity": "small",
    "context_cache_create": "large",
    # QC / RAG paths default QC_MODEL_ID=gemini-3.5-flash → large
    "qc_corrections": "large",
    "merge_pass": "large",
    "merge_synthesize": "large",
    "rag_extract": "large",
    "rag_adjudicate": "large",
    "rag_extract_fallback": "large",
    "rag_adjudicate_fallback": "large",
}


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
        read_timeout=300,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def fetch_dims(cur: pymysql.cursors.DictCursor) -> None:
    cur.execute("SELECT model_id, model_name FROM gemini_model ORDER BY model_id")
    print("models:", cur.fetchall())
    cur.execute("SELECT purpose_id, purpose_name FROM gemini_call_purpose ORDER BY purpose_id")
    print("purposes:", [r["purpose_name"] for r in cur.fetchall()])
    cur.execute("SELECT pipeline_id, pipeline_name FROM gemini_pipeline ORDER BY pipeline_id")
    print("pipelines:", cur.fetchall())


def fetch_aggregates(cur: pymysql.cursors.DictCursor, window: int = 80000) -> list[dict]:
    """Recent successful calls via call_id window (status in DB is 'success', not 'ok')."""
    sql = """
        SELECT pipeline, purpose, model, channel_code, COUNT(*) AS n
        FROM v_gemini_call
        WHERE call_id > (SELECT MAX(call_id) - %s FROM v_gemini_call)
          AND status = 'success'
          AND model IS NOT NULL
          AND purpose IS NOT NULL
        GROUP BY pipeline, purpose, model, channel_code
        ORDER BY n DESC
    """
    cur.execute(sql, (window,))
    rows = list(cur.fetchall())
    print(f"aggregate groups from last ~{window} calls: {len(rows)}")
    return rows


def tier_for_model(model: str) -> str | None:
    name = (model or "").strip().lower()
    if name in MODEL_TO_TIER:
        return MODEL_TO_TIER[name]
    if "flash-lite" in name or name.endswith("-lite"):
        return "small"
    if "pro" in name:
        return "large"
    if "flash" in name:
        return "medium"
    return None


def build_training(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict], dict[str, str]]:
    bucket: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        pipe = str(row["pipeline"] or "unknown")
        purpose = str(row["purpose"] or "unknown")
        channel = str(row["channel_code"] or "unknown")
        model = str(row["model"] or "")
        n = int(row["n"] or 0)
        if not model or n <= 0:
            continue
        bucket[(pipe, purpose, channel)][model] += n

    examples: list[dict] = []
    judgments: list[dict] = []
    purpose_tier_votes: dict[str, Counter[str]] = defaultdict(Counter)

    for (pipe, purpose, channel), models in bucket.items():
        model, votes = models.most_common(1)[0]
        tier = tier_for_model(model)
        if tier is None:
            continue
        example_id = f"ex-{uuid.uuid4().hex[:12]}"
        text = f"{pipe} {purpose} channel {channel}"
        metadata = {
            "pipeline": pipe,
            "task": purpose,
            "purpose": purpose,
            "channel_code": channel,
            "channel_type": pipe,
        }
        examples.append({"example_id": example_id, "metadata": metadata, "text": text, "image_path": None})
        judgments.append(
            {
                "example_id": example_id,
                "tier": tier,
                "judged_by": "tabson-telemetry",
                "notes": f"majority model={model} votes={votes}",
            }
        )
        purpose_tier_votes[purpose][tier] += votes

    rules: list[dict] = []
    purpose_majority: dict[str, str] = {}
    # Prefer inferencing-code defaults; fall back to telemetry majority when strong.
    for purpose, tier in sorted(PURPOSE_CODE_DEFAULTS.items()):
        purpose_majority[purpose] = tier
        rules.append(
            {
                "when": {"task": purpose},
                "tier": tier,
                "priority": 10,
                "optimize_for": "cost" if tier == "small" else "balanced",
            }
        )
    for purpose, counts in sorted(purpose_tier_votes.items()):
        tier, votes = counts.most_common(1)[0]
        total = sum(counts.values())
        purpose_majority.setdefault(purpose, tier)
        if purpose in PURPOSE_CODE_DEFAULTS:
            continue
        if total >= 20 and votes / total >= 0.70:
            rules.append(
                {
                    "when": {"task": purpose},
                    "tier": tier,
                    "priority": 5,
                    "optimize_for": "cost" if tier == "small" else "balanced",
                }
            )
    return examples, judgments, rules, purpose_majority


def write_tiers_yaml(models_seen: set[str]) -> None:
    def pick(candidates: list[str], fallback: str) -> str:
        for c in candidates:
            if c in models_seen:
                return c
        return fallback

    # Match live gemini_model dim + inferencing defaults.
    small = pick(["gemini-3.1-flash-lite"], "gemini-3.1-flash-lite")
    medium = pick(["gemini-3.5-flash-lite", "gemini-3-flash-preview"], "gemini-3.5-flash-lite")
    large = pick(["gemini-3-flash-preview", "gemini-3.5-flash"], "gemini-3-flash-preview")
    doc = {
        "tiers": [
            {
                "name": "small",
                "description": "Lite Gemini — cheap / high-volume Tabson stages.",
                "models": {"google": small, "gemini": small},
            },
            {
                "name": "medium",
                "description": "Default flash — Tabson STAGE1/STAGE2 default class.",
                "models": {"google": medium, "gemini": medium},
            },
            {
                "name": "large",
                "description": "Stronger flash / pro — harder QC-style work.",
                "models": {"google": large, "gemini": large},
            },
        ],
        "default_provider": "google",
        "low_confidence_fallback": "medium",
    }
    OUT_TIERS.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    print(f"wrote {OUT_TIERS} small={small} medium={medium} large={large}")


def write_store(examples: list[dict], judgments: list[dict], rules: list[dict]) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "examples.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in examples) + ("\n" if examples else ""),
        encoding="utf-8",
    )
    (DATA / "judgments.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "example_id": j["example_id"],
                    "tier": j["tier"],
                    "judged_by": j["judged_by"],
                    "notes": j["notes"],
                },
                ensure_ascii=False,
            )
            for j in judgments
        )
        + ("\n" if judgments else ""),
        encoding="utf-8",
    )
    (DATA / "rules.yaml").write_text(yaml.safe_dump({"rules": rules}, sort_keys=False), encoding="utf-8")
    print(f"wrote {len(examples)} examples, {len(judgments)} judgments, {len(rules)} rules → {DATA}")


def fit_classifier(examples: list[dict], judgments: list[dict]) -> None:
    by_id = {e["example_id"]: e for e in examples}
    labels = [
        TierLabel(j["example_id"], by_id[j["example_id"]]["metadata"], by_id[j["example_id"]]["text"], j["tier"], "human", j["notes"], None)
        for j in judgments
    ]
    fields = {
        "pipeline": sorted({e["metadata"]["pipeline"] for e in examples}),
        "task": sorted({e["metadata"]["task"] for e in examples}),
        "purpose": sorted({e["metadata"]["purpose"] for e in examples}),
        "channel_type": sorted({e["metadata"]["channel_type"] for e in examples}),
        "channel_code": sorted({e["metadata"]["channel_code"] for e in examples}),
    }
    encoder = MetadataEncoder(fields)
    clf = TierClassifier(encoder, HashingEmbedder())
    clf.fit(labels)
    path = DATA / "classifier.joblib"
    clf.save(path)
    report = evaluate_classifier(clf, labels, baseline_tier="medium", metadata_field="task")
    print(f"trained → {path} | n={report.n} accuracy={report.accuracy:.3f} baseline={report.baseline_accuracy:.3f}")
    top = dict(sorted(report.accuracy_by_bucket.items(), key=lambda kv: -kv[1])[:12])
    print("accuracy_by_task:", top)


def smoke_resolve(examples: list[dict]) -> None:
    tiers = load_tiers(OUT_TIERS)
    raw_rules = yaml.safe_load((DATA / "rules.yaml").read_text(encoding="utf-8")).get("rules", [])
    rules = RuleSet(
        rules=tuple(
            Rule.create(r["when"], r["tier"], priority=int(r.get("priority", 0)), optimize_for=r.get("optimize_for"))
            for r in raw_rules
        )
    )
    fields = {
        "pipeline": sorted({e["metadata"]["pipeline"] for e in examples}),
        "task": sorted({e["metadata"]["task"] for e in examples}),
        "purpose": sorted({e["metadata"]["purpose"] for e in examples}),
        "channel_type": sorted({e["metadata"]["channel_type"] for e in examples}),
        "channel_code": sorted({e["metadata"]["channel_code"] for e in examples}),
    }
    clf = TierClassifier.load(DATA / "classifier.joblib", MetadataEncoder(fields), HashingEmbedder())

    fb = resolve({}, "zzzz unfamiliar", RuleSet(), tiers)
    assert fb.fallback_outcome and "medium" in fb.fallback_outcome
    print("fallback OK:", fb.model_name, "|", fb.reason)

    if rules.rules:
        meta = {k: v[0] for k, v in rules.rules[0].when.items()}
        rd = resolve(meta, "x", rules, tiers, classifier=clf)
        assert rd.rule_outcome.startswith("rule")
        print("rule OK:", rd.model_name, "|", rd.reason)

    ex = examples[0]
    cd = resolve(dict(ex["metadata"]), ex["text"], RuleSet(), tiers, classifier=clf)
    assert cd.classifier_outcome is not None
    print("classifier OK:", cd.model_name, "|", cd.reason)


def main() -> None:
    conn = connect()
    cur = conn.cursor()
    fetch_dims(cur)
    rows = fetch_aggregates(cur, window=100000)
    conn.close()

    models_seen = {str(r["model"]) for r in rows if r.get("model")}
    print("models seen:", sorted(models_seen))
    write_tiers_yaml(models_seen)

    examples, judgments, rules, purpose_majority = build_training(rows)
    print("purpose→majority tier:", purpose_majority)

    # Seed one synthetic example per code-default purpose so under-represented tiers
    # (large/medium) still appear when live traffic is flash-lite heavy.
    existing_tasks = {e["metadata"]["task"] for e in examples}
    for purpose, tier in PURPOSE_CODE_DEFAULTS.items():
        if purpose in existing_tasks:
            continue
        example_id = f"ex-code-{purpose.replace('_', '')[:10]}-{uuid.uuid4().hex[:6]}"
        pipe = "news" if purpose != "playout_observation" else "playout"
        meta = {
            "pipeline": pipe,
            "task": purpose,
            "purpose": purpose,
            "channel_code": "generic",
            "channel_type": pipe,
        }
        examples.append(
            {
                "example_id": example_id,
                "metadata": meta,
                "text": f"{pipe} {purpose} channel generic",
                "image_path": None,
            }
        )
        judgments.append(
            {
                "example_id": example_id,
                "tier": tier,
                "judged_by": "tabson-inferencing-defaults",
                "notes": f"code default from tabson-python-inferencing → {tier}",
            }
        )

    if len(examples) < 6:
        raise SystemExit(f"not enough training rows ({len(examples)}); widen call window")

    by_tier: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for ex, ju in zip(examples, judgments, strict=True):
        by_tier[ju["tier"]].append((ex, ju))
    capped_ex: list[dict] = []
    capped_ju: list[dict] = []
    for tier in ("small", "medium", "large"):
        for ex, ju in by_tier.get(tier, [])[:120]:
            capped_ex.append(ex)
            capped_ju.append(ju)
    print("capped counts:", {t: min(120, len(by_tier.get(t, []))) for t in ("small", "medium", "large")})

    write_store(capped_ex, capped_ju, rules)
    fit_classifier(capped_ex, capped_ju)
    smoke_resolve(capped_ex)

    # Catalog-mode task pool for Playground (observed Gemini models).
    pools = {
        "video_processing": [
            "gemini-3.1-flash-lite",
            "gemini-3.5-flash-lite",
            "gemini-3-flash-preview",
        ],
        "headline_similarity": ["gemini-3.1-flash-lite"],
        "qc_corrections": ["gemini-3-flash-preview", "gemini-3.5-flash"],
    }
    (DATA / "task_pools.yaml").write_text(yaml.safe_dump({"pools": pools}, sort_keys=False), encoding="utf-8")
    print("wrote task_pools.yaml", pools)
    print("done.")


if __name__ == "__main__":
    main()
