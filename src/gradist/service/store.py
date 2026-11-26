"""File-backed state for the service: uploaded examples, human judgments, rules, and the
uploaded image files. A single local folder of JSONL/YAML files and an images/ subfolder — no
database, consistent with this project's "nothing external" scope. Meant for one trusted local
user testing this themselves, not a multi-user production store (see service/app.py).
"""

from __future__ import annotations

import dataclasses
import json
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from gradist.catalog import CatalogEntry, entry_to_yaml_dict, load_catalog, replace_entry
from gradist.labels import (
    LabelError,
    RawExample,
    TierLabel,
    load_examples,
    load_human_judgments,
)
from gradist.quality import QualityError, QualityRating
from gradist.reviews import NeedLabel, Preference, ReviewError
from gradist.rules import RuleError, RuleSet, canonical_value, rules_from_file


class Store:
    def __init__(self, root: str | Path, *, catalog_path: str | Path | None = None) -> None:
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
        self.preferences_path = self.root / "preferences.jsonl"
        self.need_labels_path = self.root / "need_labels.jsonl"
        # Task → model id lists for the catalog-first Playground flow (resolve + /v1/task-pools).
        self.task_pools_path = self.root / "task_pools.yaml"
        # Optional path to the live catalog.yaml (GRADIST_CATALOG). Needed so
        # PUT /v1/catalog/{id} can persist tier assignments (P1).
        self.catalog_path = Path(catalog_path) if catalog_path else None
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
        if not self.task_pools_path.exists():
            self.task_pools_path.write_text("pools: {}\n", encoding="utf-8")

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
                    # Only emit optimize_for when the rule actually sets one, so rules created
                    # before P2 keep round-tripping unchanged (no silent "cost" injection).
                    **({"optimize_for": rule.optimize_for} if rule.optimize_for else {}),
                }
                for rule in rules.rules
            ]
        }
        self.rules_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    # ---- catalog tier assignment (P1) -----------------------------------------------------

    def update_catalog_tier(self, model_id: str, tier: str, assigned_by: str) -> CatalogEntry:
        """Set one catalog entry's tier (+ assigned_by) and rewrite catalog.yaml.

        Raises RuleError when no catalog path is configured, the file is missing, or the
        model id is unknown. Does not validate that `tier` is a declared tier — the
        service layer does that (and also allows UNREVIEWED for demotion).
        """
        if self.catalog_path is None:
            raise RuleError("no catalog is configured for this deployment (GRADIST_CATALOG unset)")
        if not self.catalog_path.exists():
            raise RuleError(f"catalog file {self.catalog_path} does not exist")
        catalog = load_catalog(self.catalog_path)
        try:
            current = next(e for e in catalog.entries if e.id == model_id)
        except StopIteration as exc:
            raise RuleError(f"no catalog entry with id '{model_id}'") from exc
        updated = dataclasses.replace(current, tier=tier, assigned_by=assigned_by)
        new_catalog = replace_entry(catalog, updated)
        document = {"catalog": [entry_to_yaml_dict(e) for e in new_catalog.entries]}
        self.catalog_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        return updated

    # ---- task → model pools ---------------------------------------------------------------

    def load_task_pools(self) -> dict[str, list[str]]:
        """Named competition pools: task key → list of catalog model ids.

        Empty dict when unset. Used by Playground / resolve so one request maps to one task's
        selected models (cost/accuracy/balanced among that pool only).
        """
        try:
            data = yaml.safe_load(self.task_pools_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise RuleError(f"cannot read task pools file {self.task_pools_path}: {exc}") from exc
        pools = data.get("pools", {}) if isinstance(data, dict) else {}
        if not isinstance(pools, dict):
            raise RuleError(f"{self.task_pools_path}: 'pools' must be a mapping of task → model id list")
        out: dict[str, list[str]] = {}
        for task, models in pools.items():
            if not isinstance(task, str) or not task.strip():
                raise RuleError(f"{self.task_pools_path}: pool keys must be non-empty strings")
            if not isinstance(models, list) or not all(isinstance(m, str) and m.strip() for m in models):
                raise RuleError(f"{self.task_pools_path}: pools['{task}'] must be a list of model id strings")
            out[canonical_value(task.strip())] = [m.strip() for m in models]
        return out

    def save_task_pools(self, pools: dict[str, list[str]]) -> dict[str, list[str]]:
        cleaned: dict[str, list[str]] = {}
        for task, models in pools.items():
            key = canonical_value(str(task).strip())
            if not key:
                raise RuleError("task pool names must be non-empty")
            if not isinstance(models, list) or not all(isinstance(m, str) and m.strip() for m in models):
                raise RuleError(f"pools['{key}'] must be a list of model id strings")
            ids = [m.strip() for m in models]
            if not ids:
                raise RuleError(f"pools['{key}'] must contain at least one model id")
            cleaned[key] = ids
        self.task_pools_path.write_text(
            yaml.safe_dump({"pools": cleaned}, sort_keys=False),
            encoding="utf-8",
        )
        return cleaned

    def upsert_task_pool(self, task: str, models: list[str]) -> dict[str, list[str]]:
        pools = self.load_task_pools()
        key = canonical_value(task.strip())
        if not key:
            raise RuleError("task name must be non-empty")
        ids = [m.strip() for m in models if isinstance(m, str) and m.strip()]
        if not ids:
            raise RuleError(f"pools['{key}'] must contain at least one model id")
        pools[key] = ids
        return self.save_task_pools(pools)

    def delete_task_pool(self, task: str) -> dict[str, list[str]]:
        pools = self.load_task_pools()
        pools.pop(canonical_value(task.strip()), None)
        return self.save_task_pools(pools)

    # ---- request history ------------------------------------------------------------------

    def append_history(self, entry: dict[str, object]) -> None:
        """Every /v1/resolve call, so an operator can see what actually happened: which rule or
        classifier path fired, for which request, and what it decided. Append-only JSONL, same
        file-backed-no-database discipline as everything else in this store.
        """
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def history(self, limit: int = 50, requested_by: str | None = None) -> list[dict[str, object]]:
        """Newest first, capped at `limit`. `requested_by` keeps only that user's entries and is
        applied before the cap, so a busy colleague cannot push someone's rows out of the page."""
        if self.history_path.stat().st_size == 0:
            return []
        lines = self.history_path.read_text(encoding="utf-8").splitlines()
        entries = [json.loads(line) for line in lines if line.strip()]
        if requested_by is not None:
            entries = [e for e in entries if e.get("requested_by") == requested_by]
        entries.reverse()
        return entries[:limit]

    def find_history(self, request_id: str) -> dict[str, object] | None:
        if not request_id or self.history_path.stat().st_size == 0:
            return None
        for line in self.history_path.read_text(encoding="utf-8").splitlines():
            if line.strip() and f'"{request_id}"' in line:
                entry = json.loads(line)
                if entry.get("request_id") == request_id:
                    return dict(entry)
        return None

    # ---- quality ratings -------------------------------------------------------------------

    def add_quality_rating(
        self,
        model_id: str,
        metadata: dict[str, str],
        rating: int,
        judged_by: str,
        notes: str,
        *,
        request_id: str = "",
        needs: dict[str, float] | None = None,
        wrong_skills: Sequence[str] = (),
    ) -> None:
        """Raises QualityError if the rating is malformed (1-5, a named reviewer) — validated
        here, not just trusted from the request, same discipline as every other write path."""
        validated = QualityRating(
            model_id, metadata, rating, judged_by, notes, request_id, needs, tuple(wrong_skills)
        )  # raises QualityError
        record: dict[str, object] = {
            "model_id": validated.model_id,
            "metadata": dict(validated.metadata),
            "rating": validated.rating,
            "judged_by": validated.judged_by,
            "notes": validated.notes,
        }
        if request_id:
            record["request_id"] = request_id
        if needs:
            record["needs"] = dict(needs)
        if validated.wrong_skills:
            record["wrong_skills"] = list(validated.wrong_skills)
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
                    QualityRating(
                        raw["model_id"],
                        raw["metadata"],
                        raw["rating"],
                        raw["judged_by"],
                        raw.get("notes", ""),
                        raw.get("request_id", ""),
                        raw.get("needs"),
                        tuple(raw.get("wrong_skills", ())),
                    )
                )
            except QualityError:
                continue  # a malformed stored row should never crash every later read
        return ratings

    # ---- preferences and need labels (collected for learners; nothing reads them at resolve) ---

    def add_preference(self, preference: Preference) -> None:
        record: dict[str, object] = {
            "request_id": preference.request_id,
            "picked": preference.picked,
            "other": preference.other,
            "outcome": preference.outcome,
            "judged_by": preference.judged_by,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        if preference.skills:
            record["skills"] = list(preference.skills)
        if preference.needs:
            record["needs"] = dict(preference.needs)
        if preference.notes:
            record["notes"] = preference.notes
        _append_jsonl(self.preferences_path, record)

    def list_preferences(self) -> list[Preference]:
        preferences = []
        for raw in _read_jsonl(self.preferences_path):
            try:
                preferences.append(
                    Preference(
                        raw["request_id"],
                        raw["picked"],
                        raw["other"],
                        raw["outcome"],
                        raw["judged_by"],
                        tuple(raw.get("skills", ())),
                        raw.get("needs"),
                        raw.get("notes", ""),
                    )
                )
            except (KeyError, ReviewError):
                continue
        return preferences

    def add_need_label(self, label: NeedLabel) -> None:
        _append_jsonl(
            self.need_labels_path,
            {
                "request_id": label.request_id,
                "needs": dict(label.needs),
                "rule_needs": dict(label.rule_needs),
                "judged_by": label.judged_by,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

    def list_need_labels(self) -> list[NeedLabel]:
        labels = []
        for raw in _read_jsonl(self.need_labels_path):
            try:
                labels.append(NeedLabel(raw["request_id"], raw["needs"], raw["judged_by"], raw.get("rule_needs", {})))
            except (KeyError, ReviewError):
                continue
        return labels


def _append_jsonl(path: Path, record: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
