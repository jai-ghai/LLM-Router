# [Fact-Forcing preamble]
#
# The file(s) and line(s) that will call it: `src/tierrouter/service/main.py` line 49
# (`return create_app(...)`); `tests/test_service.py` fixtures `client`, `catalog_client`,
# `quality_client`; new `tests/test_auth.py` and `tests/test_users.py` also import
# `create_app`.
#
# No existing file serves the same purpose: this IS the FastAPI app module.
#
# The data schemas it reads or writes: same JSONL files via `Store` (examples, judgments,
# rules, history, quality), plus `users.jsonl` via `UsersStore` and a signed session cookie
# set/read via `SessionMiddleware`.
#
# No secrets stored in plaintext: session signing secret comes from `SessionConfig.secret`
# (populated from TIERROUTER_SESSION_SECRET in `main.py`, never persisted here); user
# passwords only land as bcrypt hashes via `auth.hash_password`.
#
# The user's verbatim instruction driving this file: "create auth or required things because
# maybe user have different workflow and they can create difference judgments" — PLAN-v2.md
# §4 (P0 — Auth, Level B).

"""FastAPI app: tiers, rules, examples, human judgments, train, evaluate, resolve, plus
session-based auth with admin/member roles (PLAN-v2.md §4).

Every mutating endpoint requires an authenticated session via `Depends(require_user)`;
`/v1/users*` additionally requires the admin role. Read endpoints require auth by default;
set `allow_read_without_auth=True` at construction to relax them (driven by
`TIERROUTER_ALLOW_READ_WITHOUT_AUTH` in `main.py`). `judged_by` on ratings/judgments is
taken from the session user — client-supplied values are ignored, not trusted.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request as StarletteRequest

from tierrouter.catalog import ModelCatalog
from tierrouter.classifier import ClassifierError, EvalReport, TierClassifier, evaluate_classifier
from tierrouter.features import Dinov2Embedder, HashingEmbedder, MetadataEncoder
from tierrouter.labels import LabelError, labels_from_rules, merge_labels, uncovered_examples
from tierrouter.quality import QualityError, quality_lookup
from tierrouter.resolve import Decision, resolve, validate_setup
from tierrouter.rules import Rule, RuleError, RuleSet
from tierrouter.service.auth import (
    AuthError,
    SessionConfig,
    hash_password,
    login_user,
    optional_user,
    require_admin,
    require_user,
)
from tierrouter.service.store import Store, guess_image_ext
from tierrouter.service.users_store import User, UserError, UsersStore
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
    # `judged_by` is accepted for back-compat but IGNORED — the server writes the session
    # user's username (anti-impersonation; see test
    # `test_post_quality_ignores_client_supplied_judged_by_and_uses_session`).
    judged_by: str = ""
    notes: str = ""


class QualityIn(BaseModel):
    model_id: str
    metadata: dict[str, str]
    rating: int
    judged_by: str = ""  # same treatment as JudgmentIn.judged_by
    notes: str = ""


class LoginIn(BaseModel):
    username: str
    password: str


class UserCreateIn(BaseModel):
    username: str
    password: str
    role: str = "member"
    notes: str = ""


class UserUpdateIn(BaseModel):
    disabled: bool


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
    lazily-loaded image embedder, the trained classifier, and the users store.
    """

    def __init__(
        self,
        tiers: TierTable,
        store: Store,
        users: UsersStore,
        allow_read_without_auth: bool,
        catalog: ModelCatalog | None = None,
        quality_field: str | None = None,
    ) -> None:
        self.tiers = tiers
        self.store = store
        self.users = users
        self.allow_read_without_auth = allow_read_without_auth
        self.catalog = catalog
        self.quality_field = quality_field
        self.classifier: TierClassifier | None = None
        self.classifier_uses_images: bool = False
        self.last_report: EvalReport | None = None
        self._image_embedder: Dinov2Embedder | None = None
        self._image_lock = threading.Lock()

    def image_embedder(self) -> Dinov2Embedder:
        """Loaded once, on first use, so server startup does not pay for it when no example has
        an image yet.
        """
        with self._image_lock:
            if self._image_embedder is None:
                self._image_embedder = Dinov2Embedder()
            return self._image_embedder


def create_app(
    tiers: TierTable,
    store: Store,
    users: UsersStore,
    session_config: SessionConfig,
    cors_origins: list[str] | None = None,
    catalog: ModelCatalog | None = None,
    quality_field: str | None = None,
    allow_read_without_auth: bool = False,
) -> FastAPI:
    app = FastAPI(title="llm-tier-router", version="0.5.0")

    # CORS first so preflights never reach route handlers; then SessionMiddleware so routes
    # can read `request.session`. `https_only` is off because the service is intended to run
    # behind a trusted reverse proxy that terminates TLS; the operator's deployment docs
    # call this out.
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,  # so the signed session cookie flows cross-origin
            allow_methods=["GET", "POST", "PUT"],
            allow_headers=["Content-Type"],
        )
    app.add_middleware(
        SessionMiddleware,
        secret_key=session_config.secret,
        session_cookie=session_config.cookie_name,
        max_age=session_config.max_age_seconds,
        same_site="lax",
        https_only=False,
    )

    state = AppState(tiers, store, users, allow_read_without_auth, catalog, quality_field)
    # The auth dependency reads `request.app.state.users` to look up sessions.
    app.state.users = users

    def _require_read(user: User | None = Depends(optional_user)) -> User | None:
        """Policy for read endpoints: an authenticated user if present; None allowed only
        when `allow_read_without_auth` is set."""
        if user is None and not state.allow_read_without_auth:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="authentication required"
            )
        return user

    # ---- health + auth -----------------------------------------------------------------

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        """Unauthenticated: load balancers and monitoring hit this before anyone logs in."""
        return {"status": "ok"}

    @app.post("/v1/auth/login")
    def do_login(payload: LoginIn, request: StarletteRequest) -> dict[str, Any]:
        if users.is_empty():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="no users exist yet; run `python -m tierrouter.service.bootstrap "
                "--username <name>` before anyone can log in",
            )
        try:
            user = login_user(users, payload.username, payload.password)
        except AuthError as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
        request.session["user_id"] = user.id
        return {"user": user.public()}

    @app.post("/v1/auth/logout")
    def do_logout(request: StarletteRequest) -> dict[str, str]:
        request.session.clear()
        return {"status": "ok"}

    @app.get("/v1/auth/me")
    def whoami(user: User = Depends(require_user)) -> dict[str, Any]:
        return {"user": user.public()}

    # ---- users (admin only) ------------------------------------------------------------

    @app.get("/v1/users")
    def list_users_endpoint(_: User = Depends(require_admin)) -> dict[str, Any]:
        return {"users": [u.public() for u in users.list_users()]}

    @app.post("/v1/users")
    def create_user_endpoint(payload: UserCreateIn, _: User = Depends(require_admin)) -> dict[str, Any]:
        if payload.role not in ("admin", "member"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="role must be 'admin' or 'member'",
            )
        # The string literal check above narrows `payload.role` to the Role Literal, but
        # mypy doesn't flow-type it through a str param — explicit assertion keeps it strict.
        role: Any = payload.role
        try:
            created = users.add(
                payload.username, hash_password(payload.password), role, notes=payload.notes
            )
        except (AuthError, UserError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc
        return {"user": created.public()}

    @app.put("/v1/users/{user_id}")
    def update_user_endpoint(
        user_id: str, payload: UserUpdateIn, _: User = Depends(require_admin)
    ) -> dict[str, Any]:
        try:
            updated = users.set_disabled(user_id, payload.disabled)
        except UserError as exc:
            if "no user with id" in str(exc):
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc
        return {"user": updated.public()}

    # ---- tiers / rules -----------------------------------------------------------------

    @app.get("/v1/tiers")
    def get_tiers(_: User | None = Depends(_require_read)) -> dict[str, Any]:
        return {
            "tiers": [{"name": t.name, "description": t.description, "models": t.models} for t in tiers.tiers],
            "default_provider": tiers.default_provider,
            "low_confidence_fallback": tiers.low_confidence_fallback,
        }

    def _rules_payload() -> dict[str, Any]:
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

    @app.get("/v1/rules")
    def get_rules(_: User | None = Depends(_require_read)) -> dict[str, Any]:
        return _rules_payload()

    @app.put("/v1/rules")
    def put_rules(update: RulesUpdate, _: User = Depends(require_user)) -> dict[str, Any]:
        try:
            rules = RuleSet(rules=tuple(Rule.create(r.when, r.tier, r.priority) for r in update.rules))
            validate_setup(rules, tiers)
        except (RuleError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        store.save_rules(rules)
        return _rules_payload()

    # ---- catalog -----------------------------------------------------------------------

    @app.get("/v1/catalog")
    def get_catalog(_: User | None = Depends(_require_read)) -> dict[str, Any]:
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

    # ---- quality ratings ---------------------------------------------------------------

    @app.post("/v1/quality")
    def add_quality(rating: QualityIn, user: User = Depends(require_user)) -> dict[str, str]:
        """`judged_by` is always the session user's username — the client-supplied field is
        ignored (anti-impersonation; see
        `test_post_quality_ignores_client_supplied_judged_by_and_uses_session`)."""
        try:
            store.add_quality_rating(
                rating.model_id, rating.metadata, rating.rating, user.username, rating.notes
            )
        except QualityError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"status": "ok"}

    @app.get("/v1/quality")
    def get_quality(_: User | None = Depends(_require_read)) -> dict[str, Any]:
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

    # ---- examples + judgments ----------------------------------------------------------

    @app.post("/v1/examples")
    def add_example(
        metadata: Annotated[str, Form()],
        text: Annotated[str, Form()] = "",
        image: Annotated[UploadFile | None, File()] = None,
        _: User = Depends(require_user),
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
    def list_examples_endpoint(_: User | None = Depends(_require_read)) -> dict[str, Any]:
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
    def list_uncovered_endpoint(_: User | None = Depends(_require_read)) -> dict[str, Any]:
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
    def add_judgment(judgment: JudgmentIn, user: User = Depends(require_user)) -> dict[str, str]:
        """`judged_by` is always the session user's username — same anti-impersonation
        discipline as /v1/quality."""
        if judgment.tier not in tiers.names:
            raise HTTPException(
                status_code=422, detail=f"'{judgment.tier}' is not a declared tier. Known: {sorted(tiers.names)}"
            )
        try:
            store.add_judgment(judgment.example_id, judgment.tier, user.username, judgment.notes)
        except LabelError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"status": "ok"}

    # ---- train + status ----------------------------------------------------------------

    @app.post("/v1/train")
    def train(_: User = Depends(require_user)) -> TrainResult:
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
    def status_endpoint(_: User | None = Depends(_require_read)) -> Status:
        report = state.last_report
        return Status(
            trained=state.classifier is not None,
            n_labels_last_trained_on=report.n if report else None,
            uses_images=state.classifier_uses_images,
        )

    # ---- resolve + history -------------------------------------------------------------

    @app.post("/v1/resolve")
    def do_resolve(
        metadata: Annotated[str, Form()],
        text: Annotated[str, Form()] = "",
        provider: Annotated[str | None, Form()] = None,
        min_confidence: Annotated[float, Form()] = 0.0,
        selected_models: Annotated[str | None, Form()] = None,
        min_quality: Annotated[float | None, Form()] = None,
        image: Annotated[UploadFile | None, File()] = None,
        user: User = Depends(require_user),
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
                "requested_by": user.username,
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
    def get_history(limit: int = 50, _: User | None = Depends(_require_read)) -> dict[str, Any]:
        return {"history": store.history(limit)}

    return app
