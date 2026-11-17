"""FastAPI app: tiers, rules, examples, human judgments, train, evaluate, resolve.

No authentication anywhere in this file. It is built for one trusted local user testing this
project themselves (see __init__.py) — not for exposing to a network or to other people. Put it
behind your own auth layer before that, the same way the earlier R&D project's service did.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from tierrouter.catalog import ModelCatalog
from tierrouter.classifier import ClassifierError, EvalReport, TierClassifier, evaluate_classifier
from tierrouter.features import Dinov2Embedder, HashingEmbedder, MetadataEncoder
from tierrouter.labels import LabelError, labels_from_rules, merge_labels, uncovered_examples
from tierrouter.quality import QualityError, quality_lookup
from tierrouter.resolve import Decision, resolve, validate_setup
from tierrouter.rules import Rule, RuleError, RuleSet
from tierrouter.service.store import Store, guess_image_ext
from tierrouter.tiers import TierTable


class RuleIn(BaseModel):
    when: dict[str, str | list[str]]
    tier: str
    priority: int = 0


class RulesUpdate(BaseModel):
    rules: list[RuleIn]


class JudgmentIn(BaseModel):
    example_id: str
    tier: str
    judged_by: str
    notes: str = ""


class QualityIn(BaseModel):
    model_id: str
    metadata: dict[str, str]
    rating: int
    judged_by: str
    notes: str = ""


class TrainResult(BaseModel):
    n_labels: int
    n_rule_labels: int
    n_human_labels: int
    n_uncovered: int
    uses_images: bool
    report: dict[str, Any]


class Status(BaseModel):
    trained: bool
    n_labels_last_trained_on: int | None
    uses_images: bool


def _infer_fields(examples: list[Any]) -> dict[str, list[str]]:
    fields: dict[str, set[str]] = {}
    for example in examples:
        for key, value in example.metadata.items():
            fields.setdefault(key, set()).add(value)
    return {key: sorted(values) for key, values in fields.items()}


class AppState:
    """Everything that must survive between requests: the tier table, the editable rules, the

    lazily-loaded image embedder, and the trained classifier (None until the first /v1/train).
    """

    def __init__(
        self, tiers: TierTable, store: Store, catalog: ModelCatalog | None = None, quality_field: str | None = None
    ) -> None:
        self.tiers = tiers
        self.store = store
        self.catalog = catalog
        self.quality_field = quality_field
        self.classifier: TierClassifier | None = None
        self.classifier_uses_images: bool = False
        self.last_report: EvalReport | None = None
        self._image_embedder: Dinov2Embedder | None = None
        self._image_lock = threading.Lock()

    def image_embedder(self) -> Dinov2Embedder:
        """Loaded once, on first use, so server startup does not pay for it when no example has

        an image yet. A real download/load the first time it is called (see README's note on
        the `images` extra); cached after that for the life of the process.
        """
        with self._image_lock:
            if self._image_embedder is None:
                self._image_embedder = Dinov2Embedder()
            return self._image_embedder


def create_app(
    tiers: TierTable,
    store: Store,
    cors_origins: list[str] | None = None,
    catalog: ModelCatalog | None = None,
    quality_field: str | None = None,
) -> FastAPI:
    app = FastAPI(title="llm-tier-router", version="0.4.0")
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_methods=["GET", "POST", "PUT"],
            allow_headers=["Content-Type"],
        )
    state = AppState(tiers, store, catalog, quality_field)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/tiers")
    def get_tiers() -> dict[str, Any]:
        return {
            "tiers": [{"name": t.name, "description": t.description, "models": t.models} for t in tiers.tiers],
            "default_provider": tiers.default_provider,
            "low_confidence_fallback": tiers.low_confidence_fallback,
        }

    @app.get("/v1/rules")
    def get_rules() -> dict[str, Any]:
        rules = store.load_rules()
        return {
            "rules": [
                {
                    "when": {k: (list(v) if len(v) > 1 else v[0]) for k, v in r.when.items()},
                    "tier": r.tier,
                    "priority": r.priority,
                }
                for r in rules.rules
            ]
        }

    @app.put("/v1/rules")
    def put_rules(update: RulesUpdate) -> dict[str, Any]:
        try:
            rules = RuleSet(rules=tuple(Rule.create(r.when, r.tier, r.priority) for r in update.rules))
            validate_setup(rules, tiers)
        except (RuleError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        store.save_rules(rules)
        return get_rules()

    @app.get("/v1/catalog")
    def get_catalog() -> dict[str, Any]:
        """The selectable model catalog, or an empty list if this deployment has none configured
        (TIERROUTER_CATALOG unset) — resolve() then falls back to tiers.yaml's single model per
        tier, same as before this feature existed.
        """
        if state.catalog is None:
            return {"catalog": []}
        return {
            "catalog": [
                {
                    "id": e.id,
                    "provider": e.provider,
                    "tier": e.tier,
                    "cost": e.cost,
                    "supports_image": e.supports_image,
                    "label": e.label,
                }
                for e in state.catalog.entries
            ]
        }

    @app.post("/v1/quality")
    def add_quality(rating: QualityIn) -> dict[str, str]:
        """A human's rating of a model's REAL output for a REAL request — the accuracy signal
        no public capability source can give (see quality.py). Independent of TIERROUTER_QUALITY_FIELD:
        ratings can always be recorded; whether they ever INFLUENCE a decision is a separate,
        opt-in choice (that env var).
        """
        try:
            store.add_quality_rating(rating.model_id, rating.metadata, rating.rating, rating.judged_by, rating.notes)
        except QualityError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"status": "ok"}

    @app.get("/v1/quality")
    def get_quality() -> dict[str, Any]:
        """Aggregated scores, bucketed by TIERROUTER_QUALITY_FIELD if one is configured — the
        same field used to apply a quality floor at resolve time (see AppState.quality_field).
        Without one configured, ratings are still stored (via POST) but have nothing to bucket
        by here, so this returns them ungrouped, one row per (model, raw metadata) combination.
        """
        ratings = store.list_quality_ratings()
        if state.quality_field:
            scores = quality_lookup(ratings, state.quality_field)
            return {
                "field": state.quality_field,
                "scores": [
                    {"model_id": model_id, "value": value, "average": round(avg, 2)}
                    for (model_id, value), avg in sorted(scores.items())
                ],
            }
        return {
            "field": None,
            "ratings": [
                {"model_id": r.model_id, "metadata": dict(r.metadata), "rating": r.rating, "judged_by": r.judged_by}
                for r in ratings
            ],
        }

    @app.post("/v1/examples")
    def add_example(
        metadata: Annotated[str, Form()],
        text: Annotated[str, Form()] = "",
        image: Annotated[UploadFile | None, File()] = None,
    ) -> dict[str, Any]:
        import json as _json

        try:
            parsed = _json.loads(metadata)
        except _json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail=f"'metadata' is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise HTTPException(status_code=422, detail="'metadata' must be a JSON object")
        image_bytes = image.file.read() if image is not None else None
        image_ext = guess_image_ext(image.filename if image else None, image.content_type if image else None)
        example = store.add_example({str(k): str(v) for k, v in parsed.items()}, text, image_bytes, image_ext)
        rules = store.load_rules()
        hit = rules.resolve(example.metadata)
        return {
            "example_id": example.example_id,
            "metadata": example.metadata,
            "text": example.text,
            "has_image": example.image_path is not None,
            "covered_by_rule": hit is not None,
            "rule_reason": hit[1] if hit else None,
        }

    @app.get("/v1/examples")
    def list_examples() -> dict[str, Any]:
        rules = store.load_rules()
        examples = store.list_examples()
        rows = []
        for example in examples:
            hit = rules.resolve(example.metadata)
            rows.append(
                {
                    "example_id": example.example_id,
                    "metadata": example.metadata,
                    "text": example.text,
                    "has_image": example.image_path is not None,
                    "covered_by_rule": hit is not None,
                    "rule_reason": hit[1] if hit else None,
                }
            )
        return {"examples": rows}

    @app.get("/v1/examples/uncovered")
    def list_uncovered() -> dict[str, Any]:
        rules = store.load_rules()
        examples = store.list_examples()
        uncovered = uncovered_examples(examples, rules)
        return {
            "examples": [
                {"example_id": e.example_id, "metadata": e.metadata, "text": e.text, "has_image": e.image_path is not None}
                for e in uncovered
            ]
        }

    @app.post("/v1/judgments")
    def add_judgment(judgment: JudgmentIn) -> dict[str, str]:
        if judgment.tier not in tiers.names:
            raise HTTPException(
                status_code=422, detail=f"'{judgment.tier}' is not a declared tier. Known: {sorted(tiers.names)}"
            )
        try:
            store.add_judgment(judgment.example_id, judgment.tier, judgment.judged_by, judgment.notes)
        except LabelError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"status": "ok"}

    @app.post("/v1/train")
    def train() -> TrainResult:
        examples = store.list_examples()
        if not examples:
            raise HTTPException(status_code=422, detail="No examples stored yet. Add some first.")
        rules = store.load_rules()
        rule_labels = labels_from_rules(examples, rules)
        human_labels = store.list_judgments(tiers.names)
        labels = merge_labels(rule_labels, human_labels)
        if not labels:
            raise HTTPException(
                status_code=422,
                detail="No labels at all: no rule matched anything and no judgments are recorded yet.",
            )
        uses_images = any(label.image_path for label in labels)
        fields = _infer_fields(examples)
        encoder = MetadataEncoder(fields) if fields else MetadataEncoder({"_unused": []})
        image_embedder = state.image_embedder() if uses_images else None
        classifier = TierClassifier(encoder, HashingEmbedder(), image_embedder)
        try:
            classifier.fit(labels)
        except ClassifierError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        classifier.save(store.classifier_path)
        state.classifier = classifier
        state.classifier_uses_images = uses_images
        baseline = tiers.low_confidence_fallback
        eval_field = next(iter(fields), None)
        report = (
            evaluate_classifier(classifier, labels, baseline_tier=baseline, metadata_field=eval_field)
            if eval_field
            else None
        )
        state.last_report = report
        return TrainResult(
            n_labels=len(labels),
            n_rule_labels=sum(1 for lbl in labels if lbl.source == "rule"),
            n_human_labels=sum(1 for lbl in labels if lbl.source == "human"),
            n_uncovered=len(uncovered_examples(examples, rules)),
            uses_images=uses_images,
            report=(
                {
                    "accuracy": report.accuracy,
                    "baseline_accuracy": report.baseline_accuracy,
                    "accuracy_by_bucket": report.accuracy_by_bucket,
                    "bucket_sizes": report.bucket_sizes,
                    "field": eval_field,
                }
                if report
                else {}
            ),
        )

    @app.get("/v1/status")
    def status() -> Status:
        report = state.last_report
        return Status(
            trained=state.classifier is not None,
            n_labels_last_trained_on=report.n if report else None,
            uses_images=state.classifier_uses_images,
        )

    @app.post("/v1/resolve")
    def do_resolve(
        metadata: Annotated[str, Form()],
        text: Annotated[str, Form()] = "",
        provider: Annotated[str | None, Form()] = None,
        min_confidence: Annotated[float, Form()] = 0.0,
        selected_models: Annotated[str | None, Form()] = None,
        min_quality: Annotated[float | None, Form()] = None,
        image: Annotated[UploadFile | None, File()] = None,
    ) -> Decision:
        import json as _json

        try:
            parsed = _json.loads(metadata)
        except _json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail=f"'metadata' is not valid JSON: {exc}") from exc
        selected: list[str] | None = None
        if selected_models:
            try:
                selected = _json.loads(selected_models)
            except _json.JSONDecodeError as exc:
                raise HTTPException(status_code=422, detail=f"'selected_models' is not valid JSON: {exc}") from exc
            if not isinstance(selected, list):
                raise HTTPException(status_code=422, detail="'selected_models' must be a JSON array of ids")
        rules = store.load_rules()
        image_bytes = image.file.read() if image is not None else None
        classifier = state.classifier
        if image_bytes is not None and classifier is not None and not classifier.uses_images:
            raise HTTPException(
                status_code=422,
                detail="The trained classifier was not trained with any image examples, so it "
                "cannot use this image. Add an image example and retrain first.",
            )
        clean_metadata = {str(k): str(v) for k, v in parsed.items()}
        quality_scores = quality_lookup(store.list_quality_ratings(), state.quality_field) if state.quality_field else None
        try:
            decision = resolve(
                clean_metadata,
                text,
                rules,
                tiers,
                classifier=classifier,
                provider=provider,
                min_confidence=min_confidence,
                image_bytes=image_bytes,
                catalog=state.catalog,
                selected_models=set(selected) if selected else None,
                quality=quality_scores,
                quality_field=state.quality_field,
                min_quality=min_quality,
            )
        except (RuleError, ClassifierError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        store.append_history(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "metadata": clean_metadata,
                "text": text,
                "has_image": image_bytes is not None,
                "provider": provider,
                "selected_models": selected,
                "min_quality": min_quality,
                "tier": decision.tier,
                "model_name": decision.model_name,
                "confidence": decision.confidence,
                "reason": decision.reason,
                "rule_outcome": decision.rule_outcome,
                "classifier_outcome": decision.classifier_outcome,
                "fallback_outcome": decision.fallback_outcome,
            }
        )
        return decision

    @app.get("/v1/history")
    def get_history(limit: int = 50) -> dict[str, Any]:
        return {"history": store.history(limit)}

    return app
