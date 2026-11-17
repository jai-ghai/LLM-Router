"""File-backed state for the service: uploaded examples, human judgments, rules, and the
uploaded image files. A single local folder of JSONL/YAML files and an images/ subfolder — no
database, consistent with this project's "nothing external" scope. Meant for one trusted local
user testing this themselves, not a multi-user production store (see service/app.py).
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import yaml

from tierrouter.labels import (
    LabelError,
    RawExample,
    TierLabel,
    load_examples,
    load_human_judgments,
)
from tierrouter.quality import QualityError, QualityRating
from tierrouter.rules import RuleSet, rules_from_file


class Store:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.root.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.examples_path = self.root / "examples.jsonl"
        self.judgments_path = self.root / "judgments.jsonl"
        self.rules_path = self.root / "rules.yaml"
        self.classifier_path = self.root / "classifier.joblib"
        self.history_path = self.root / "history.jsonl"
        self.quality_path = self.root / "quality.jsonl"
        if not self.examples_path.exists():
            self.examples_path.write_text("", encoding="utf-8")
        if not self.judgments_path.exists():
            self.judgments_path.write_text("", encoding="utf-8")
        if not self.rules_path.exists():
            self.rules_path.write_text("rules: []\n", encoding="utf-8")
        if not self.history_path.exists():
            self.history_path.write_text("", encoding="utf-8")
        if not self.quality_path.exists():
            self.quality_path.write_text("", encoding="utf-8")

    # ---- examples -----------------------------------------------------------------------

    def add_example(
        self, metadata: dict[str, str], text: str, image_bytes: bytes | None, image_ext: str
    ) -> RawExample:
        example_id = f"ex-{uuid.uuid4().hex[:12]}"
        image_path: str | None = None
        if image_bytes is not None:
            image_file = self.images_dir / f"{example_id}{image_ext}"
            image_file.write_bytes(image_bytes)
            image_path = str(image_file)
        record = {"example_id": example_id, "metadata": metadata, "text": text, "image_path": image_path}
        with self.examples_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return RawExample(example_id, metadata, text, image_path)

    def list_examples(self) -> list[RawExample]:
        if self.examples_path.stat().st_size == 0:
            return []
        return load_examples(self.examples_path)

    # ---- human judgments ------------------------------------------------------------------

    def add_judgment(self, example_id: str, tier: str, judged_by: str, notes: str) -> None:
        if example_id not in {e.example_id for e in self.list_examples()}:
            raise LabelError(f"example_id '{example_id}' is not a stored example")
        record = {"example_id": example_id, "tier": tier, "judged_by": judged_by, "notes": notes}
        with self.judgments_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def list_judgments(self, known_tiers: set[str]) -> list[TierLabel]:
        by_id = {example.example_id: example for example in self.list_examples()}
        return load_human_judgments(self.judgments_path, by_id, known_tiers)

    # ---- rules --------------------------------------------------------------------------

    def load_rules(self) -> RuleSet:
        return rules_from_file(self.rules_path)

    def save_rules(self, rules: RuleSet) -> None:
        document = {
            "rules": [
                {
                    "when": {k: (list(v) if len(v) > 1 else v[0]) for k, v in rule.when.items()},
                    "tier": rule.tier,
                    **({"priority": rule.priority} if rule.priority else {}),
                }
                for rule in rules.rules
            ]
        }
        self.rules_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    # ---- request history ------------------------------------------------------------------

    def append_history(self, entry: dict[str, object]) -> None:
        """Every /v1/resolve call, so an operator can see what actually happened: which rule or
        classifier path fired, for which request, and what it decided. Append-only JSONL, same
        file-backed-no-database discipline as everything else in this store.
        """
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def history(self, limit: int = 50) -> list[dict[str, object]]:
        """Newest first, capped at `limit`."""
        if self.history_path.stat().st_size == 0:
            return []
        lines = self.history_path.read_text(encoding="utf-8").splitlines()
        entries = [json.loads(line) for line in lines if line.strip()]
        entries.reverse()
        return entries[:limit]

    # ---- quality ratings -------------------------------------------------------------------

    def add_quality_rating(self, model_id: str, metadata: dict[str, str], rating: int, judged_by: str, notes: str) -> None:
        """Raises QualityError if the rating is malformed (1-5, a named reviewer) — validated
        here, not just trusted from the request, same discipline as every other write path."""
        validated = QualityRating(model_id, metadata, rating, judged_by, notes)  # raises QualityError
        record = {
            "model_id": validated.model_id,
            "metadata": dict(validated.metadata),
            "rating": validated.rating,
            "judged_by": validated.judged_by,
            "notes": validated.notes,
        }
        with self.quality_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def list_quality_ratings(self) -> list[QualityRating]:
        if self.quality_path.stat().st_size == 0:
            return []
        lines = self.quality_path.read_text(encoding="utf-8").splitlines()
        ratings = []
        for line in lines:
            if not line.strip():
                continue
            raw = json.loads(line)
            try:
                ratings.append(
                    QualityRating(raw["model_id"], raw["metadata"], raw["rating"], raw["judged_by"], raw.get("notes", ""))
                )
            except QualityError:
                continue  # a malformed stored row should never crash every later read
        return ratings


def guess_image_ext(filename: str | None, content_type: str | None) -> str:
    """A file extension for a stored image upload: from the filename if it has one and known,

    else from the content type, else a safe default. Never trusts an extension blindly enough
    to execute anything — this only picks what suffix a saved image file gets.
    """
    known = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
    if filename:
        suffix = Path(filename).suffix.lower()
        if suffix in known:
            return suffix
    by_type = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "image/bmp": ".bmp",
    }
    return by_type.get((content_type or "").lower(), ".png")
