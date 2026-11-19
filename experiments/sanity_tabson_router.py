# [Fact-Forcing preamble]
#
# Callers: CLI only — `python experiments/sanity_tabson_router.py` (no in-repo import yet).
# Existing files: experiments/train_from_tabson_db.py trains; tests/test_ladder_e2e.py
# covers synthetic ladder fixtures — neither runs a Tabson-trained news/sports/image sanity
# matrix against data/classifier.joblib + data/rules.yaml + config/tiers.tabson.yaml.
# Reads: data/examples.jsonl, data/classifier.joblib, data/rules.yaml,
# config/tiers.tabson.yaml. Writes: nothing (stdout report only).
# User: "restart kar lo .. model running hai mere system pe ? , and uske baad aap full
# sanity test karo .. kya or kaise output aa rhe hai .. i don't want model khud judge na
# kar paye .. so please make sure router , text, image , news content , sports content
# or yeh sab filter kar skta hai.."
"""Offline Tabson ladder sanity: rules → classifier → fallback (no LLM calls)."""

from __future__ import annotations

import io
import json
from pathlib import Path

from PIL import Image

from gradist.classifier import TierClassifier
from gradist.features import HashingEmbedder, MetadataEncoder
from gradist.resolve import resolve
from gradist.rules import rules_from_file
from gradist.tiers import load_tiers

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _load_clf() -> TierClassifier:
    examples = [
        json.loads(line)
        for line in (DATA / "examples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    fields = {
        "pipeline": sorted({e["metadata"]["pipeline"] for e in examples}),
        "task": sorted({e["metadata"]["task"] for e in examples}),
        "purpose": sorted({e["metadata"]["purpose"] for e in examples}),
        "channel_type": sorted({e["metadata"]["channel_type"] for e in examples}),
        "channel_code": sorted({e["metadata"]["channel_code"] for e in examples}),
    }
    return TierClassifier.load(
        DATA / "classifier.joblib", MetadataEncoder(fields), HashingEmbedder()
    )


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (20, 120, 200)).save(buf, format="PNG")
    return buf.getvalue()


def _src(d) -> str:
    if d.classifier_outcome is None and d.fallback_outcome is None:
        return "RULE"
    if d.fallback_outcome:
        return "FALLBACK"
    return "CLF"


def main() -> None:
    tiers = load_tiers(ROOT / "config" / "tiers.tabson.yaml")
    rules = rules_from_file(DATA / "rules.yaml")
    clf = _load_clf()
    img = _png()

    cases: list[tuple[str, dict[str, str], str, bool]] = [
        (
            "RULE headline_similarity",
            {
                "pipeline": "news",
                "task": "headline_similarity",
                "purpose": "headline_similarity",
                "channel_type": "news",
                "channel_code": "1010297",
            },
            "news headline_similarity channel 1010297",
            False,
        ),
        (
            "RULE v3_single",
            {
                "pipeline": "news",
                "task": "v3_single",
                "purpose": "v3_single",
                "channel_type": "news",
                "channel_code": "1010297",
            },
            "news v3_single channel 1010297",
            False,
        ),
        (
            "RULE playout_observation",
            {
                "pipeline": "playout",
                "task": "playout_observation",
                "purpose": "playout_observation",
                "channel_type": "playout",
                "channel_code": "0",
            },
            "playout playout_observation",
            False,
        ),
        (
            "RULE spell_batch news",
            {
                "pipeline": "news",
                "task": "spell_batch",
                "purpose": "spell_batch",
                "channel_type": "news",
                "channel_code": "1",
            },
            "news spell_batch channel 1",
            False,
        ),
        (
            "META sports channel_type + v3 rule",
            {
                "pipeline": "news",
                "task": "v3_single",
                "purpose": "v3_single",
                "channel_type": "sports",
                "channel_code": "99",
            },
            "sports cricket match report",
            False,
        ),
        (
            "TEXT free news NO task",
            {"pipeline": "news", "channel_type": "news"},
            "Breaking: Cabinet approves new education policy amid protests in capital",
            False,
        ),
        (
            "TEXT free sports NO task",
            {"pipeline": "news", "channel_type": "sports"},
            "IPL final: Mumbai Indians chase 190 as Kohli scores century at Wankhede",
            False,
        ),
        (
            "TEXT sports keywords only",
            {},
            "Live cricket score India vs Australia Test match day 3 bowling figures",
            False,
        ),
        (
            "TEXT news politics only",
            {},
            "Parliament debate on GST reform and inflation numbers released today",
            False,
        ),
        ("FALLBACK empty", {}, "", False),
        (
            "IMAGE unknown task",
            {"task": "unknown_vision", "pipeline": "news"},
            "describe the frame",
            True,
        ),
        (
            "IMAGE + known rule task",
            {
                "task": "headline_similarity",
                "pipeline": "news",
                "purpose": "headline_similarity",
                "channel_type": "news",
                "channel_code": "1",
            },
            "headline with image",
            True,
        ),
    ]

    print("=" * 96)
    print("OFFLINE LADDER SANITY (router only — no Gemini / no LLM judge)")
    print("=" * 96)
    print(f"{'CASE':<40} {'SRC':<9} {'TIER':<8} {'MODEL':<28} CONF")
    for name, meta, text, want_img in cases:
        try:
            d = resolve(
                meta,
                text,
                rules,
                tiers,
                classifier=clf,
                image_bytes=img if want_img else None,
                min_confidence=0.55,
            )
            conf = "-" if d.confidence is None else f"{d.confidence:.3f}"
            print(f"{name:<40} {_src(d):<9} {d.tier:<8} {d.model_name:<28} {conf}")
            print(f"  reason: {d.reason}")
            if d.classifier_outcome:
                print(f"  clf:    {d.classifier_outcome}")
            if d.fallback_outcome:
                print(f"  fb:     {d.fallback_outcome}")
        except Exception as exc:  # noqa: BLE001 — report every case
            print(f"{name:<40} ERROR {type(exc).__name__}: {exc}")

    print("\n" + "=" * 96)
    print("CONTENT FILTER PROBE (empty meta; only free text differs)")
    print("=" * 96)
    for label, txt in [
        (
            "news-ish",
            "Election results show ruling party lead in five states after close counting",
        ),
        (
            "sports-ish",
            "Messi scores hat-trick as Argentina win World Cup qualifier against Brazil",
        ),
    ]:
        d = resolve({}, txt, rules, tiers, classifier=clf, min_confidence=0.55)
        print(
            f"{label}: tier={d.tier} model={d.model_name} conf={d.confidence} "
            f"src={_src(d)} reason={d.reason}"
        )

    print("\nclassifier.image_embedder:", getattr(clf, "image_embedder", None))
    print("n_rules:", len(rules.rules))
    print("tiers:", list(tiers.names))


if __name__ == "__main__":
    main()
