# [Fact-Forcing preamble]
#
# The file(s) and line(s) that will call it: `src/gradist/service/main.py` line 49
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
# (populated from GRADIST_SESSION_SECRET in `main.py`, never persisted here); user
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
`GRADIST_ALLOW_READ_WITHOUT_AUTH` in `main.py`). `judged_by` on ratings/judgments is
taken from the session user — client-supplied values are ignored, not trusted.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import yaml
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request as StarletteRequest

from gradist.catalog import (
    VALID_OPTIMIZE_FOR,
    ModelCatalog,
    OptimizeFor,
    entry_to_yaml_dict,
    label_catalog_tiers,
    replace_entry,
)
from gradist.classifier import ClassifierError, EvalReport, TierClassifier, evaluate_classifier

# create_app / train endpoint — SentenceTransformer for content-aware fit when installed.
from gradist.features import (
    Dinov2Embedder,
    Embedder,
    HashingEmbedder,
    MetadataEncoder,
    SentenceTransformerEmbedder,
)
from gradist.labels import LabelError, labels_from_rules, merge_labels, uncovered_examples
from gradist.qc_risk import QcRiskHead
from gradist.quality import QualityError, quality_lookup
from gradist.quality_head import TierQualityHead
from gradist.resolve import Decision, resolve, validate_setup
from gradist.rules import Rule, RuleError, RuleSet, canonical_value
from gradist.service.auth import (
    AuthError,
    SessionConfig,
    hash_password,
    login_user,
    optional_user,
    require_admin,
    require_user,
)
from gradist.service.store import Store, guess_image_ext
from gradist.service.users_store import User, UserError, UsersStore
from gradist.tiers import UNREVIEWED_TIER, TierTable


class RuleIn(BaseModel):
    when: dict[str, str | list[str]]
    tier: str
    priority: int = 0
    optimize_for: str | None = None
    """None means "no per-rule preference". When set, it must be one of 'cost', 'accuracy', or
    'balanced' (validated in `put_rules`); a bad value yields 422."""


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


class CatalogTierIn(BaseModel):
    tier: str
    """A declared tier name from tiers.yaml, or the UNREVIEWED sentinel (demotion)."""


class TaskPoolIn(BaseModel):
    models: list[str]


class TaskPoolsIn(BaseModel):
    pools: dict[str, list[str]]


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
    # Deployment facts the Settings screen shows (no secrets). Additive; older UIs ignore.
    quality_field: str | None = None
    catalog_configured: bool = False
    allow_read_without_auth: bool = False
    unreviewed_count: int = 0
    quality_head: bool = False
    qc_risk: bool = False



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
        self.quality_head: TierQualityHead | None = None
        self.qc_risk: QcRiskHead | None = None
        self.loaded_n_labels: int | None = None
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


MINILM = "sentence-transformers/all-MiniLM-L6-v2"


def _classifier_meta_path(store: Store) -> Path:
    return store.root / "classifier.meta.json"


def _load_saved_models(state: AppState, store: Store) -> None:
    """Reload what training saved, so a restart doesn't silently come back untrained.

    The classifier is only reloaded with the exact embedders recorded in its sidecar — a
    different text/image embedder would change the feature width it was fitted on.
    """
    root = store.root
    head_path, risk_path = root / "quality_head.joblib", root / "qc_risk.joblib"
    meta_path = _classifier_meta_path(store)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    needs_minilm = head_path.exists() or risk_path.exists() or (meta or {}).get("text_embedder") == "minilm"
    minilm: SentenceTransformerEmbedder | None = None
    if needs_minilm:
        try:
            minilm = SentenceTransformerEmbedder(MINILM)
        except ImportError:
            minilm = None
    if minilm is not None and head_path.exists():
        state.quality_head = TierQualityHead.load(head_path, minilm)
    if minilm is not None and risk_path.exists():
        state.qc_risk = QcRiskHead.load(risk_path, minilm)
    if meta is None or not store.classifier_path.exists():
        return
    text_embedder: Embedder | None = minilm if meta["text_embedder"] == "minilm" else HashingEmbedder()
    if text_embedder is None:
        return
    state.classifier = TierClassifier.load(
        store.classifier_path,
        MetadataEncoder(meta["fields"]),
        text_embedder,
        state.image_embedder() if meta["uses_images"] else None,
    )
    state.classifier_uses_images = bool(meta["uses_images"])
    state.loaded_n_labels = meta.get("n_labels")


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
    app = FastAPI(title="Gradist", version="0.5.0")

    # CORS first so preflights never reach route handlers; then SessionMiddleware so routes
    # can read `request.session`. `https_only` is off because the service is intended to run
    # behind a trusted reverse proxy that terminates TLS; the operator's deployment docs
    # call this out.
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,  # so the signed session cookie flows cross-origin
            allow_methods=["GET", "POST", "PUT", "DELETE"],
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
    _load_saved_models(state, store)
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
                detail="no users exist yet; run `python -m gradist.service.bootstrap "
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
                    "optimize_for": r.optimize_for,
                }
                for r in rules.rules
            ]
        }

    @app.get("/v1/rules")
    def get_rules(_: User | None = Depends(_require_read)) -> dict[str, Any]:
        return _rules_payload()

    @app.put("/v1/rules")
    def put_rules(update: RulesUpdate, _: User = Depends(require_user)) -> dict[str, Any]:
        for index, r in enumerate(update.rules):
            if r.optimize_for is not None and r.optimize_for not in VALID_OPTIMIZE_FOR:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"rules[{index}].optimize_for must be one of "
                        f"{list(VALID_OPTIMIZE_FOR)!r} or null; got '{r.optimize_for}'"
                    ),
                )
        try:
            rules = RuleSet(
                rules=tuple(
                    Rule.create(
                        r.when,
                        r.tier,
                        r.priority,
                        # Narrowed by the membership check above; mypy can't flow-type this
                        # through a str param so it needs the Any hop.
                        optimize_for=(r.optimize_for or None),  # type: ignore[arg-type]
                    )
                    for r in update.rules
                )
            )
            validate_setup(rules, tiers)
        except (RuleError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        store.save_rules(rules)
        return _rules_payload()

    # ---- catalog -----------------------------------------------------------------------

    def _catalog_payload() -> dict[str, Any]:
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
                    # P3 — PLAN-v2.md §6: modality booleans for the Catalog UI chips.
                    "supports_video": e.supports_video,
                    "supports_audio": e.supports_audio,
                    "label": e.label,
                    # P1 — who last assigned the tier (empty until an admin reviews).
                    "assigned_by": e.assigned_by,
                }
                for e in state.catalog.entries
            ]
        }

    @app.get("/v1/catalog")
    def get_catalog(_: User | None = Depends(_require_read)) -> dict[str, Any]:
        return _catalog_payload()

    # Static path BEFORE /v1/catalog/{model_id} so "label-tiers" is never treated as an id.
    @app.post("/v1/catalog/label-tiers")
    def label_catalog_by_cost(
        force: bool = False,
        user: User = Depends(require_admin),
    ) -> dict[str, Any]:
        """Admin-only: assign browse-only tier labels from cost thirds (assigned_by=heuristic).

        Cheapest ~⅓ of models → first tier name, middle → second, expensive → last.
        Does not change resolve() — pick_from_pool ignores tier. By default preserves rows
        an admin manually labeled; pass `?force=true` to rewrite every row.
        """
        if state.catalog is None:
            raise HTTPException(
                status_code=422,
                detail="no catalog is configured for this deployment (GRADIST_CATALOG unset)",
            )
        # Cheapest→expensive by declared rank (0=small). Never list(tiers.names) — it's a set.
        ordered_tiers = [name for name, _rank in sorted(tiers.rank().items(), key=lambda kv: kv[1])]
        labeled = label_catalog_tiers(
            state.catalog,
            ordered_tiers,
            only_unreviewed_or_heuristic=not force,
        )
        if store.catalog_path is not None:
            document = {"catalog": [entry_to_yaml_dict(e) for e in labeled.entries]}
            store.catalog_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        state.catalog = labeled
        return _catalog_payload()

    @app.put("/v1/catalog/{model_id}")
    def put_catalog_tier(
        model_id: str, payload: CatalogTierIn, user: User = Depends(require_admin)
    ) -> dict[str, Any]:
        """Admin-only: assign (or demote) a catalog entry's tier. Persists to catalog.yaml
        and updates the in-memory catalog so the next resolve() sees it without a restart.
        `tier` must be a declared tier or UNREVIEWED (Q6.5.b demotion)."""
        if state.catalog is None:
            raise HTTPException(
                status_code=422,
                detail="no catalog is configured for this deployment (GRADIST_CATALOG unset)",
            )
        tier = payload.tier.strip()
        allowed = set(tiers.names) | {UNREVIEWED_TIER}
        if tier not in allowed:
            raise HTTPException(
                status_code=422,
                detail=f"tier must be one of {sorted(allowed)}; got '{tier}'",
            )
        try:
            updated = store.update_catalog_tier(model_id, tier, assigned_by=user.username)
        except RuleError as exc:
            msg = str(exc)
            if "no catalog entry with id" in msg:
                raise HTTPException(status_code=404, detail=msg) from exc
            raise HTTPException(status_code=422, detail=msg) from exc
        state.catalog = replace_entry(state.catalog, updated)
        return _catalog_payload()

    # ---- task → model pools ------------------------------------------------------------

    @app.get("/v1/task-pools")
    def get_task_pools(_: User | None = Depends(_require_read)) -> dict[str, Any]:
        try:
            return {"pools": store.load_task_pools()}
        except RuleError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.put("/v1/task-pools")
    def put_task_pools(payload: TaskPoolsIn, _: User = Depends(require_user)) -> dict[str, Any]:
        try:
            return {"pools": store.save_task_pools(payload.pools)}
        except RuleError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.put("/v1/task-pools/{task}")
    def put_one_task_pool(
        task: str, payload: TaskPoolIn, _: User = Depends(require_user)
    ) -> dict[str, Any]:
        try:
            return {"pools": store.upsert_task_pool(task, payload.models)}
        except RuleError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.delete("/v1/task-pools/{task}")
    def delete_one_task_pool(task: str, _: User = Depends(require_user)) -> dict[str, Any]:
        try:
            return {"pools": store.delete_task_pool(task)}
        except RuleError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    # ---- quality ratings ---------------------------------------------------------------

    @app.get("/v1/quality/field")
    def get_quality_field(_: User | None = Depends(_require_read)) -> dict[str, str | None]:
        """The configured quality-bucket field name (e.g. "task"), or null when quality
        flooring is not enabled for this deployment. The History rating modal uses this to
        decide whether to show the field chip; judged_by is never a client concern (P0)."""
        return {"field": state.quality_field}

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
        encoder_fields = fields or {"_unused": []}
        encoder = MetadataEncoder(encoder_fields)
        image_embedder = state.image_embedder() if uses_images else None
        # Prefer a small sentence embedder when installed — content dynamics for HF/text labels.
        text_embedder: Embedder
        try:
            text_embedder = SentenceTransformerEmbedder(MINILM)
            text_kind = "minilm"
        except ImportError:
            text_embedder = HashingEmbedder()
            text_kind = "hashing"
        classifier = TierClassifier(encoder, text_embedder, image_embedder)
        try:
            classifier.fit(labels)
        except ClassifierError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        classifier.save(store.classifier_path)
        _classifier_meta_path(store).write_text(
            json.dumps(
                {"text_embedder": text_kind, "uses_images": uses_images, "fields": encoder_fields, "n_labels": len(labels)}
            ),
            encoding="utf-8",
        )
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
        unreviewed = 0
        if state.catalog is not None:
            unreviewed = sum(1 for e in state.catalog.entries if e.tier == UNREVIEWED_TIER)
        return Status(
            trained=state.classifier is not None,
            n_labels_last_trained_on=report.n if report else state.loaded_n_labels,
            uses_images=state.classifier_uses_images,
            quality_field=state.quality_field,
            catalog_configured=state.catalog is not None,
            allow_read_without_auth=state.allow_read_without_auth,
            unreviewed_count=unreviewed,
            quality_head=state.quality_head is not None,
            qc_risk=state.qc_risk is not None,
        )

    # ---- resolve + history -------------------------------------------------------------

    @app.post("/v1/resolve")
    def do_resolve(
        metadata: Annotated[str, Form()],
        text: Annotated[str, Form()] = "",
        provider: Annotated[str | None, Form()] = None,
        min_confidence: Annotated[float, Form()] = 0.0,
        selected_models: Annotated[str | None, Form()] = None,
        task: Annotated[str | None, Form()] = None,
        min_quality: Annotated[float | None, Form()] = None,
        optimize_for: Annotated[str, Form()] = "cost",
        image: Annotated[UploadFile | None, File()] = None,
        user: User = Depends(require_user),
    ) -> Decision:
        import json as _json

        try:
            parsed = _json.loads(metadata)
        except _json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail=f"'metadata' is not valid JSON: {exc}") from exc
        if optimize_for not in VALID_OPTIMIZE_FOR:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"'optimize_for' must be one of {list(VALID_OPTIMIZE_FOR)!r}; "
                    f"got '{optimize_for}'"
                ),
            )
        # Membership check above is enough for mypy to narrow the form field to OptimizeFor.
        optimize_for_literal: OptimizeFor = optimize_for
        # None = "no filter / all models". A parsed empty list must stay empty (not collapse
        # to None), or an empty task pool would silently compete over the full catalog.
        selected: list[str] | None = None
        selected_explicit = False
        if selected_models is not None and selected_models != "":
            try:
                parsed_selected = _json.loads(selected_models)
            except _json.JSONDecodeError as exc:
                raise HTTPException(status_code=422, detail=f"'selected_models' is not valid JSON: {exc}") from exc
            if not isinstance(parsed_selected, list):
                raise HTTPException(status_code=422, detail="'selected_models' must be a JSON array of ids")
            selected = [str(m) for m in parsed_selected]
            selected_explicit = True
        task_name = canonical_value((task or "").strip()) or None
        # Catalog-first: a named task pool supplies the competition set when the caller did
        # not pass selected_models explicitly.
        if not selected_explicit and task_name and state.catalog is not None:
            try:
                pools = store.load_task_pools()
            except RuleError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            if task_name not in pools:
                raise HTTPException(
                    status_code=422,
                    detail=f"unknown task pool '{task_name}'; save models for it on Catalog first",
                )
            selected = list(pools[task_name])
            selected_explicit = True
        if selected_explicit and selected is not None and len(selected) == 0:
            raise HTTPException(
                status_code=422,
                detail="selection pool is empty — add models on Catalog or pick a non-empty task pool",
            )
        rules = store.load_rules()
        image_bytes = image.file.read() if image is not None else None
        classifier = state.classifier
        # Catalog-first resolve never runs the classifier; only gate images on the ladder path.
        if (
            state.catalog is None
            and image_bytes is not None
            and classifier is not None
            and not classifier.uses_images
        ):
            raise HTTPException(
                status_code=422,
                detail="The trained classifier was not trained with any image examples, so it "
                "cannot use this image. Add an image example and retrain first.",
            )
        clean_metadata = {str(k): str(v) for k, v in parsed.items()}
        if task_name:
            clean_metadata.setdefault("task", task_name)
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
                selected_models=set(selected) if selected_explicit and selected is not None else None,
                quality=quality_scores,
                quality_field=state.quality_field,
                min_quality=min_quality,
                optimize_for=optimize_for_literal,
                quality_head=state.quality_head,
                qc_risk=state.qc_risk,
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
                "task": task_name,
                "selected_models": selected,
                "min_quality": min_quality,
                "optimize_for_requested": optimize_for_literal,
                "optimize_for": decision.optimize_for,
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
