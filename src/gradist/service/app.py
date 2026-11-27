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
# maybe user have different workflow and they can create difference judgments".

"""FastAPI app: tiers, rules, examples, human judgments, train, evaluate, resolve, plus
session-based auth with admin/member roles.

Every mutating endpoint requires an authenticated session via `Depends(require_user)`;
`/v1/users*` additionally requires the admin role. Read endpoints require auth by default;
set `allow_read_without_auth=True` at construction to relax them (driven by
`GRADIST_ALLOW_READ_WITHOUT_AUTH` in `main.py`). `judged_by` on ratings/judgments is
taken from the session user — client-supplied values are ignored, not trusted.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import tempfile
import threading
import uuid
from collections import Counter
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
    VALID_DEADLINES,
    VALID_OPTIMIZE_FOR,
    Deadline,
    ModelCatalog,
    OptimizeFor,
    entry_to_yaml_dict,
    label_catalog_tiers,
    replace_entry,
)
from gradist.classifier import (
    ClassifierError,
    EvalReport,
    TierClassifier,
    evaluate_classifier,
    with_screen_text,
)

# create_app / train endpoint — SentenceTransformer for content-aware fit when installed.
from gradist.features import (
    Dinov2Embedder,
    Embedder,
    HashingEmbedder,
    MetadataEncoder,
    SentenceTransformerEmbedder,
    load_image_bytes,
)
from gradist.feedback import learn_skills
from gradist.labels import (
    LabelError,
    TierLabel,
    labels_from_rules,
    merge_labels,
    uncovered_examples,
)
from gradist.neighbours import NeighbourIndex, ReviewedRow, rows_from_reviews
from gradist.ocr import ScreenReader, join_lines, merge_frame_lines
from gradist.qc_risk import QcRiskHead
from gradist.quality import QualityError, QualityRating, quality_lookup
from gradist.quality_head import TierQualityHead
from gradist.resolve import Decision, resolve, validate_setup
from gradist.reviews import NeedLabel, Preference, ReviewError
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
from gradist.skills import load_task_skills, request_needs
from gradist.tiers import UNREVIEWED_TIER, TierTable
from gradist.video_profile import VideoProfile, VideoProfiler


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
    model_id: str = ""
    metadata: dict[str, str] = {}
    rating: int
    judged_by: str = ""  # same treatment as JudgmentIn.judged_by
    notes: str = ""
    request_id: str = ""
    """When set, model_id and metadata come from that history entry, not the client."""
    wrong_skills: list[str] = []
    """"What went wrong?" — skills the output got wrong. Only kept with request_id."""


class PreferenceIn(BaseModel):
    request_id: str
    other_model: str
    outcome: str
    """"picked", "other" or "tie"."""
    skills: list[str] = []
    notes: str = ""


class NeedLabelIn(BaseModel):
    request_id: str
    needs: dict[str, float]


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



CHANNEL_FIELDS = ("channel_name", "network", "channel_type", "genre")


def _channel_directory(examples: list[Any]) -> dict[str, dict[str, str]]:
    """channel_code → the most common channel_name / network / type / genre seen with it."""
    seen: dict[str, dict[str, Counter[str]]] = {}
    for example in examples:
        code = example.metadata.get("channel_code")
        if not code:
            continue
        per_field = seen.setdefault(code, {f: Counter() for f in CHANNEL_FIELDS})
        for f in CHANNEL_FIELDS:
            if example.metadata.get(f):
                per_field[f][example.metadata[f]] += 1
    return {
        code: {f: counts.most_common(1)[0][0] for f, counts in fields.items() if counts}
        for code, fields in sorted(seen.items())
    }


def _with_ocr(label: TierLabel, reader: ScreenReader) -> TierLabel:
    if not label.image_path:
        return label
    lines = reader.read(load_image_bytes(label.image_path))
    return dataclasses.replace(label, text=with_screen_text(label.text, join_lines(lines)))


MAX_OCR_FRAMES = 8


def _read_frames(reader: ScreenReader, frames: list[bytes]) -> dict[str, Any]:
    """OCR several frames of one clip and merge their lines. Each frame is cached by content, so
    frames already read by POST /v1/ocr cost nothing here."""
    try:
        per_frame = [reader.read(frame) for frame in frames]
    except Exception as exc:  # noqa: BLE001 — OCR is advisory; a bad frame must not fail the caller
        return {"engine": reader.engine, "lines": [], "frames_read": len(frames), "error": str(exc)}
    return {
        "engine": reader.engine,
        "lines": [{"text": ln.text, "confidence": ln.confidence} for ln in merge_frame_lines(per_frame)],
        "frames_read": len(frames),
        "lines_per_frame": [len(lines) for lines in per_frame],
    }


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
        self.ocr_enabled = os.environ.get("GRADIST_OCR") == "1"
        self._screen_reader: ScreenReader | None = None
        skills_path = Path(os.environ.get("GRADIST_TASK_SKILLS") or "config/task_skills.yaml")
        self.task_skills = load_task_skills(skills_path) if skills_path.exists() else {}
        self.neighbours = NeighbourIndex(HashingEmbedder(), os.environ.get("GRADIST_NEIGHBOURS") or "auto")
        self.free_models_for_video = os.environ.get("GRADIST_FREE_MODELS_FOR_VIDEO") == "1"
        self.ease_easy_video = os.environ.get("GRADIST_EASE_EASY_VIDEO", "1") != "0"
        self._video_profiler: VideoProfiler | None = None

    def video_profiler(self) -> VideoProfiler:
        """Reads uploaded videos (video_profile.py); its models load on the first upload."""
        with self._image_lock:
            if self._video_profiler is None:
                reader = None if os.environ.get("GRADIST_OCR") != "1" else self._screen_reader_unlocked()
                self._video_profiler = VideoProfiler(
                    self.store.root / "video_profiles",
                    text_counter=reader.count_text if reader is not None else None,
                )
            return self._video_profiler

    def image_embedder(self) -> Dinov2Embedder:
        """Loaded once, on first use, so server startup does not pay for it when no example has
        an image yet.
        """
        with self._image_lock:
            if self._image_embedder is None:
                self._image_embedder = Dinov2Embedder()
            return self._image_embedder

    def screen_reader(self) -> ScreenReader | None:
        """OCR for frames, when GRADIST_OCR=1. Built on first use (the models take ~30 s to load)."""
        if not self.ocr_enabled:
            return None
        with self._image_lock:
            return self._screen_reader_unlocked()

    def _screen_reader_unlocked(self) -> ScreenReader:
        if self._screen_reader is None:
            langs = tuple(s.strip() for s in os.environ.get("GRADIST_OCR_LANGS", "hi,en").split(",") if s.strip())
            self._screen_reader = ScreenReader(langs, cache_path=self.store.root / "ocr_cache.json")
        return self._screen_reader


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
    if minilm is not None:
        state.neighbours = NeighbourIndex(minilm, state.neighbours.mode)
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
        reads_screen_text=bool(meta.get("reads_screen_text", False)),
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
                detail="no users exist yet; set GRADIST_ADMIN_USERNAME and GRADIST_ADMIN_PASSWORD "
                "and restart, or run `python -m gradist.service.bootstrap "
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
                    # Modality booleans for the Catalog UI chips.
                    "supports_video": e.supports_video,
                    "supports_audio": e.supports_audio,
                    "label": e.label,
                    # P1 — who last assigned the tier (empty until an admin reviews).
                    "assigned_by": e.assigned_by,
                    "capability": e.capability,
                    "capability_source": e.capability_source,
                    "multilingual": e.multilingual,
                    "multilingual_source": e.multilingual_source,
                    "supports_structured_output": e.supports_structured_output,
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

    @app.get("/v1/channels")
    def get_channels(_: User | None = Depends(_require_read)) -> dict[str, Any]:
        return {"channels": _channel_directory(store.list_examples())}

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
        `test_post_quality_ignores_client_supplied_judged_by_and_uses_session`).

        With `request_id`, the rating is for that request's pick: the model, metadata and skill
        needs are copied from history (so the rating also teaches feedback.learn_skills), and a
        member may only rate their own requests."""
        model_id, metadata = rating.model_id, rating.metadata
        needs: dict[str, float] | None = None
        if rating.request_id:
            entry = owned_history(rating.request_id, user)
            model_id = str(entry.get("model_name") or "")
            metadata = {str(k): str(v) for k, v in dict(entry.get("metadata") or {}).items()}  # type: ignore[call-overload]
            needs = history_needs(entry)
        if not model_id:
            raise HTTPException(status_code=422, detail="model_id (or request_id) is required")
        try:
            store.add_quality_rating(
                model_id,
                metadata,
                rating.rating,
                user.username,
                rating.notes,
                request_id=rating.request_id,
                needs=needs,
                wrong_skills=rating.wrong_skills if rating.request_id else (),
            )
        except QualityError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"status": "ok"}

    def reviewed_rows(ratings: list[QualityRating]) -> list[ReviewedRow]:
        """Ratings and comparisons given by request id, joined with that request's text."""
        preferences = store.list_preferences()
        ids = {r.request_id for r in ratings if r.request_id} | {p.request_id for p in preferences}
        return rows_from_reviews(
            [(r.request_id, r.model_id, r.rating) for r in ratings if r.request_id],
            [(p.request_id, p.winner, p.loser, p.picked, p.other) for p in preferences],
            store.history_texts(ids),
        )

    def owned_history(request_id: str, user: User) -> dict[str, object]:
        entry = store.find_history(request_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"no request with id '{request_id}'")
        if user.role != "admin" and entry.get("requested_by") != user.username:
            raise HTTPException(status_code=403, detail="you can only review your own requests")
        return entry

    def history_needs(entry: dict[str, object]) -> dict[str, float] | None:
        raw = entry.get("skill_needs")
        return {str(k): float(v) for k, v in raw.items()} if isinstance(raw, dict) else None

    @app.post("/v1/preferences")
    def add_preference(body: PreferenceIn, user: User = Depends(require_user)) -> dict[str, str]:
        """Which was better for this request: the router's pick or another model? Outputs are
        produced outside the router; the pick and the request's needs come from history."""
        entry = owned_history(body.request_id, user)
        other = body.other_model.strip()
        if state.catalog is not None and other and all(e.id != other for e in state.catalog.entries):
            raise HTTPException(status_code=422, detail=f"'{other}' is not in the catalog")
        try:
            store.add_preference(
                Preference(
                    body.request_id,
                    str(entry.get("model_name") or ""),
                    other,
                    body.outcome,
                    user.username,
                    tuple(body.skills),
                    history_needs(entry),
                    body.notes,
                )
            )
        except ReviewError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"status": "ok"}

    @app.get("/v1/preferences")
    def get_preferences(_: User | None = Depends(_require_read)) -> dict[str, Any]:
        return {
            "preferences": [
                {"request_id": p.request_id, "picked": p.picked, "other": p.other, "outcome": p.outcome,
                 "skills": list(p.skills), "judged_by": p.judged_by}
                for p in store.list_preferences()
            ]
        }

    @app.post("/v1/need-labels")
    def add_need_label(body: NeedLabelIn, user: User = Depends(require_user)) -> dict[str, str]:
        """What this request really needed, per skill, as corrected by a reviewer."""
        entry = owned_history(body.request_id, user)
        try:
            store.add_need_label(NeedLabel(body.request_id, body.needs, user.username, history_needs(entry) or {}))
        except ReviewError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"status": "ok"}

    @app.get("/v1/need-labels")
    def get_need_labels(_: User | None = Depends(_require_read)) -> dict[str, Any]:
        return {
            "need_labels": [
                {"request_id": n.request_id, "needs": dict(n.needs), "rule_needs": dict(n.rule_needs), "judged_by": n.judged_by}
                for n in store.list_need_labels()
            ]
        }

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
        reader = state.screen_reader() if uses_images else None
        if reader is not None:
            labels = [_with_ocr(label, reader) for label in labels]
            reader.save_cache()
        classifier = TierClassifier(encoder, text_embedder, image_embedder, reads_screen_text=reader is not None)
        try:
            classifier.fit(labels)
        except ClassifierError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        classifier.save(store.classifier_path)
        _classifier_meta_path(store).write_text(
            json.dumps(
                {
                    "text_embedder": text_kind,
                    "uses_images": uses_images,
                    "reads_screen_text": reader is not None,
                    "fields": encoder_fields,
                    "n_labels": len(labels),
                }
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

    @app.post("/v1/ocr")
    def read_screen_text(
        frames: Annotated[list[UploadFile], File()], _: User = Depends(require_user)
    ) -> dict[str, Any]:
        """Read frames ahead of a resolve (the Playground calls this as soon as a clip is picked),
        so sending the same frames to /v1/resolve later hits the cache instead of waiting."""
        if len(frames) > MAX_OCR_FRAMES:
            raise HTTPException(status_code=422, detail=f"at most {MAX_OCR_FRAMES} frames per request")
        reader = state.screen_reader()
        if reader is None:
            return {"engine": None, "lines": [], "frames_read": 0}
        return _read_frames(reader, [f.file.read() for f in frames])

    @app.post("/v1/video/profile")
    def profile_video(video: Annotated[UploadFile, File()], _: User = Depends(require_user)) -> dict[str, Any]:
        """Read what is in a clip (speech, language, sounds, picture, on-screen text) once; pass the
        returned profile_id to /v1/resolve for every request about that clip."""
        try:
            profiler = state.video_profiler()
            suffix = Path(video.filename or "clip.mp4").suffix or ".mp4"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                shutil.copyfileobj(video.file, tmp)
            try:
                profile = profiler.profile(Path(tmp.name))
            finally:
                Path(tmp.name).unlink(missing_ok=True)
        except ImportError as exc:
            raise HTTPException(
                status_code=501, detail=f"video profiling needs the 'video' extra (pip install .[video]): {exc}"
            ) from exc
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"could not read that video: {exc}") from exc
        return {"profile_id": profile.id, "profile": profile.as_dict()}

    @app.get("/v1/video/profile/{profile_id}")
    def get_video_profile(profile_id: str, _: User = Depends(require_user)) -> dict[str, Any]:
        profile = _saved_profile(profile_id)
        return {"profile_id": profile.id, "profile": profile.as_dict()}

    def _saved_profile(profile_id: str) -> VideoProfile:
        if not profile_id.isalnum():
            raise HTTPException(status_code=422, detail="'video_profile_id' must be the id /v1/video/profile returned")
        profile = VideoProfiler(store.root / "video_profiles").load(profile_id)
        if profile is None:
            raise HTTPException(status_code=404, detail=f"no video profile '{profile_id}' — upload the clip to /v1/video/profile")
        return profile

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
        deadline: Annotated[str | None, Form()] = None,
        video_profile_id: Annotated[str | None, Form()] = None,
        image: Annotated[UploadFile | None, File()] = None,
        ocr_frames: Annotated[list[UploadFile] | None, File()] = None,
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
        deadline_literal: Deadline | None = None
        if deadline:
            if deadline not in VALID_DEADLINES:
                raise HTTPException(
                    status_code=422, detail=f"'deadline' must be one of {list(VALID_DEADLINES)!r}; got '{deadline}'"
                )
            deadline_literal = deadline
        video_profile = _saved_profile(video_profile_id) if video_profile_id else None
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
        frame_bytes = [f.file.read() for f in ocr_frames or []]
        if len(frame_bytes) > MAX_OCR_FRAMES:
            raise HTTPException(status_code=422, detail=f"at most {MAX_OCR_FRAMES} ocr_frames per request")
        screen: dict[str, Any] = {"engine": None, "lines": []}
        reader = state.screen_reader() if (image_bytes or frame_bytes) else None
        if reader is not None and frame_bytes:
            screen = _read_frames(reader, frame_bytes)
        elif reader is not None and image_bytes:
            try:
                screen = {
                    "engine": reader.engine,
                    "lines": [{"text": ln.text, "confidence": ln.confidence} for ln in reader.read(image_bytes)],
                }
            except Exception as exc:  # noqa: BLE001 — OCR is advisory; a bad frame must not fail the resolve
                screen = {"engine": reader.engine, "lines": [], "error": str(exc)}
        screen_text = " | ".join(line["text"] for line in screen["lines"])
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
        ratings = store.list_quality_ratings()
        quality_scores = quality_lookup(ratings, state.quality_field) if state.quality_field else None
        learned_skills = learn_skills(ratings)
        similar_requests = state.neighbours.assess(text, reviewed_rows(ratings)) if state.catalog is not None else None
        risk = (
            state.qc_risk.risk(clean_metadata, text)
            if state.qc_risk is not None and state.qc_risk.applies_to(clean_metadata, text)
            else None
        )
        skill_needs = request_needs(
            clean_metadata,
            text,
            screen_text=screen_text,
            has_image=image_bytes is not None,
            qc_risk=risk,
            task_skills=state.task_skills,
            video=video_profile,
        )
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
                screen_text=screen_text,
                skill_needs=skill_needs,
                learned_skills=learned_skills,
                similar_requests=similar_requests,
                deadline=deadline_literal,
                free_models_for_video=state.free_models_for_video,
                ease_easy_video=state.ease_easy_video,
            )
        except (RuleError, ClassifierError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        decision = dataclasses.replace(decision, request_id=uuid.uuid4().hex)
        if image_bytes or frame_bytes:
            decision.signals["screen_text"] = screen
        if video_profile is not None:
            decision.signals["video_profile"] = video_profile.as_dict()
        store.append_history(
            {
                "request_id": decision.request_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "requested_by": user.username,
                "skill_needs": {s: round(v, 2) for s, v in skill_needs.needs.items() if v > 0},
                "difficulty": None if skill_needs.difficulty is None else round(skill_needs.difficulty, 2),
                "metadata": clean_metadata,
                "text": text,
                "has_image": image_bytes is not None,
                "video": None
                if video_profile is None
                else {
                    "profile_id": video_profile.id,
                    "content_type": video_profile.content_type,
                    "language": video_profile.language,
                    "speech_share": video_profile.speech_share,
                },
                "deadline": deadline_literal,
                "provider": provider,
                "task": task_name,
                "selected_models": selected,
                "min_quality": min_quality,
                "optimize_for_requested": optimize_for_literal,
                "optimize_for": decision.optimize_for,
                "tier": decision.tier,
                "model_name": decision.model_name,
                "shortlist": [str(c["id"]) for c in decision.candidates if c.get("id")],
                "confidence": decision.confidence,
                "reason": decision.reason,
                "rule_outcome": decision.rule_outcome,
                "classifier_outcome": decision.classifier_outcome,
                "fallback_outcome": decision.fallback_outcome,
            }
        )
        return decision

    @app.get("/v1/history")
    def get_history(
        limit: int = 50, requested_by: str | None = None, viewer: User = Depends(require_user)
    ) -> dict[str, Any]:
        # Requests are private to whoever made them; only admins may see (or filter) everyone's.
        # This deliberately ignores GRADIST_ALLOW_READ_WITHOUT_AUTH: request text is user data.
        owner = (requested_by or None) if viewer.role == "admin" else viewer.username
        entries = store.history(limit, requested_by=owner)
        ratings_by_request: dict[str, list[dict[str, object]]] = {}
        for rating in store.list_quality_ratings():
            if rating.request_id:
                ratings_by_request.setdefault(rating.request_id, []).append(
                    {"rating": rating.rating, "judged_by": rating.judged_by}
                )
        preferences_by_request: dict[str, list[dict[str, object]]] = {}
        for p in store.list_preferences():
            preferences_by_request.setdefault(p.request_id, []).append(
                {"other": p.other, "outcome": p.outcome, "skills": list(p.skills), "judged_by": p.judged_by}
            )
        need_label_by_request = {n.request_id: {"needs": dict(n.needs), "judged_by": n.judged_by} for n in store.list_need_labels()}
        for entry in entries:
            request_id = str(entry.get("request_id") or "")
            for key, found in (
                ("ratings", ratings_by_request.get(request_id)),
                ("preferences", preferences_by_request.get(request_id)),
                ("need_label", need_label_by_request.get(request_id)),
            ):
                if found:
                    entry[key] = found
        return {"history": entries}

    return app
