[Fact-Forcing gate preamble — this file is a planning document, updated in-place as decisions
land.

Supersedes: nothing. PLAN.md (the original v1 plan) stays authoritative for the already-shipped
Tasks 1-8 scope; this file adds a v2 scope on top and cross-references PLAN.md and
ARCHITECTURE.md rather than replacing them.

Linked from: no other file yet — the operator (Jai) will link it from README.md once this plan
is approved and coding starts. Until then, discoverable by filename next to PLAN.md.

Operator's verbatim concerns that drove writing this file (Hinglish, do not translate away):
"first just work on my project .. keep analysis and samjho ek baat or 1 issue - catalog yeh
dyanamic nhi hai model ki capability pta honi chaiye dekhiyo open source llm router mai chalta
hai mujhe or industry level walo ki bhi capability pta honi chaiye tabhi ho payega na .. and
another thing is yaml file mai fix likh gyi hai new models or uski details aa nhi skti , koi
open source api ho ya kuch bhi , another part is mujhe cost ek sath accuracy jaiad matter karti
hai because isme video processing bhi hai toh video ka output correct hona important hai uske
liye mai human declared label or human quality check de dunga .. agar kisko bhout jaida
accuracy chaiye toh warna flowsahi hai hamara i guess"

Follow-up operator choices:
- "redesign_plan" + "tests_too" (write the complete plan first, review, then code; include tests
  for new behaviour in each PR).
- Auth Q5.5.a: Level B (session auth, single tenant, per-user audit). Confirmed 2026-11-17.
- P3 naming: rename to "Modality expansion + human QC loop" per the call transcript's
  "text-primary, video-secondary, modality-agnostic" framing. Confirmed 2026-11-17.
- Q4.5 (Rule carries optimize_for): Yes.
- Q5.5.b (re-tag task at rating): No.
- Q6.5.a (merge refresh + import scripts): Yes.
- Q6.5.b (allow demote to UNREVIEWED): Yes.
- Q6.5.c (default provider filter on import): Error with helpful usage.

Call-transcript context reconciled (2026-11-17, pasted by operator):
- "General Platform, Not Just Tapson" — the project is NOT a BARC/Tapson-specific tool.
- "Most content in the company is text-based; video is secondary" — modality-agnostic design
  stays; P3 is framed accordingly, not as a video-first piece.
- "LangChain chosen as the abstraction layer ... Single API across all providers." The current
  `src/tierrouter/llm_stub.py` is the planned seam. P2's output contract (`model_name` string)
  stays compatible with a future factory layer; no scope creep into LangChain itself here.
- "Factory pattern over integrations to be built later. Jai to start on the ML classifier part."
  So the classifier/routing side is Jai's workstream — this plan stays inside it.

Duplicates check: no other PLAN-v2.md / PLAN_v2.md / plan-v2.md exists in the repo. Confirmed
via Glob earlier.

Schema: plain markdown, same shape as PLAN.md — top-level ##-sections, tables, fenced code
blocks for commands and file lists, no emoji.
]

# Plan v2: Auth, dynamic catalog, accuracy-first routing, modality QC loop

Builds on `PLAN.md` and `ARCHITECTURE.md`. Status: **P0 + P2 + P3 shipped; next is P1**.
All six operator decisions in §11 are confirmed. Each piece (P0, P2, P3, P1) ships as its
own focused commit.

## 1. Why this plan exists

The operator's three concerns, grounded by code inspection, were:

1. "catalog yeh dynamic nhi hai — model ki capability pta honi chaiye … open source llm router
   mai chalta hai mujhe, or industry level walo ki bhi capability pta honi chaiye."
2. "yaml file mai fix likh gyi hai — new models or uski details aa nhi skti … koi open source
   api ho ya kuch bhi."
3. "cost ek sath accuracy jaiad matter karti hai … uske liye mai human declared label or human
   quality check de dunga. agar kisko bhout jaida accuracy chaiye toh warna flow sahi hai."

Plus a fourth concern surfaced during Q5.5.a review:

4. "create auth or required things because maybe user have different workflow and they can
   create difference judgments." → P0 below.

## 2. What's already in place today (grounded in the code)

| Concern | Code that already exists | What it does NOT do |
|---|---|---|
| OSS + industry capability source | `src/tierrouter/capability.py` fetches OpenRouter `/api/v1/models`. OpenRouter covers both industry models (GPT-5, Claude, Gemini, Grok) and open-source (Llama, Qwen, DeepSeek, Mistral, Phi, Yi). Public endpoint, no key needed for the list. | Only used offline from `experiments/refresh_catalog.py`. `resolve()` never calls it — by design (ARCHITECTURE.md §1). |
| YAML as source of truth | `experiments/refresh_catalog.py` refreshes `cost`, `context_length`, `supports_image` for models already in `catalog.yaml`, matching by id suffix. | Does NOT add new models. Manual. No "discover" path. |
| Cost + accuracy | `src/tierrouter/quality.py` models 1-5 human ratings keyed by `(model_id, metadata-field value)`. `catalog.pick()` takes `min_quality` as a floor. `resolve()` wires it through. `/v1/quality` POST+GET already exist in `service/app.py`. | No "pick the HIGHEST-quality model at the tier" (accuracy-first) mode — only "cheapest above the floor." |
| Modality coverage | `supports_image: bool` on every `CatalogEntry`. `capability.py` reads OpenRouter's `input_modalities` list but only extracts `"image"`. | No `supports_video`, no `supports_audio`. Nothing in the UI prompts a human to rate a real output. |
| Identity and audit | None. `service/app.py` top comment: "built for one trusted local user ... no authentication anywhere." | No way to attribute a rating/rule/catalog edit to a specific user. No per-user judgment separation. |

## 3. Guardrails that still hold (do not relax)

Carried from v1 — any proposed change that violates one of these is rejected, not merged.

- **G1 — No model is ever called from `resolve()` or anywhere on a request's hot path.**
  (ARCHITECTURE.md §1.) Live capability data is fetched only from offline scripts.
- **G2 — Capability is never silently downgraded below the decided tier.** Already enforced in
  `catalog.pick()`; every new selection mode must preserve this.
- **G3 — Human quality judgments are the source of truth for "did this actually work".** No
  public benchmark or automatic metric is substituted for a reviewer's rating.
- **G4 — A model nobody has reviewed yet is not punished for a missing rating.** Already
  enforced in `_passes_quality`; preserved in every new mode here.
- **G5 — Tier assignment (`small`/`medium`/`large`) stays a human decision.** No API, including
  OpenRouter, is trusted to decide what tier a new model belongs to for *this* deployment's
  tasks. The dynamic-import path (P1) only produces *candidates* for human review.
- **G6 — Modality-agnostic design.** Text is primary, image/video/audio secondary (call
  transcript). The feature vector, catalog, and routing logic all stay general; no modality
  gets special-cased in the hot path.
- **G7 — Classifier never sees a model name.** It predicts an abstract tier; the tier →
  concrete model mapping is config + runtime selection. (ARCHITECTURE.md §5.) P2's
  `optimize_for` lives at the catalog level, never inside the classifier.

## 4. P0 — Auth, Level B: session auth + per-user audit

**Why this is first:** P3's rating modal records `judged_by`, and that needs to be a
trustworthy user id, not free-text. Building P3 against an impersonable identity means
rebuilding the modal later. P0 lands first so every subsequent piece can rely on it.

### 4.1 Design

- **Users** stored in the existing `store/` directory as `users.jsonl` (append-only; matches
  how `examples/`, `judgments/`, `quality/`, `history` are already stored — see
  `service/store.py`). One line per user, each line a `{id, username, password_hash, role,
  created_at, disabled}` record.
- **Password hashing:** `passlib[bcrypt]`. One new runtime dep, declared in `pyproject.toml`.
- **Session:** `starlette.middleware.sessions.SessionMiddleware`. Cookie-based,
  `httponly=True`, `samesite="lax"`. Secret from `TIERROUTER_SESSION_SECRET` env var; the
  service refuses to start if that env var is unset (no insecure fallback).
- **Bootstrap:** `python -m tierrouter.service.bootstrap --username <name>` prompts for a
  password, writes the first user (role `admin`), and exits. The service refuses to accept
  logins until at least one user exists.
- **Roles:** `admin` and `member`. `admin` can create/disable users, assign catalog tiers,
  edit rules. `member` can rate decisions, add examples, add judgments, trigger train, and
  resolve. Same split as "operator configures the system / reviewer uses it."
- **Hot path impact:** one `request.session` lookup + one jsonl read for the current user.
  Negligible; the jsonl is small by design.

### 4.2 API surface

New endpoints:

```
POST   /v1/auth/login       form: username, password        → {user: {id, username, role}}
POST   /v1/auth/logout                                      → {status: "ok"}
GET    /v1/auth/me                                          → {user: ...} | 401
GET    /v1/users            admin only                      → {users: [...]}
POST   /v1/users            admin only; form fields         → {user: ...}
PUT    /v1/users/{id}       admin only; {disabled: bool}    → {user: ...}
```

Every mutating endpoint gains a FastAPI `Depends(require_user)` dependency:

- `POST /v1/quality` — ratings now carry `judged_by = session.user.id`; client no longer sends it
- `POST /v1/judgments` — same treatment
- `POST /v1/examples` — carries `added_by`
- `POST /v1/train`, `POST /v1/resolve` — require auth, record `requested_by` in history
- `PUT  /v1/rules` — carries `updated_by`
- `PUT  /v1/catalog/{id}` (new in P1) — carries `assigned_by`

Read endpoints (`GET /v1/catalog`, `/v1/tiers`, `/v1/history`, `/v1/examples`, `/v1/rules`,
`/v1/status`, `/v1/quality`) also require auth by default, but a config toggle
`TIERROUTER_ALLOW_READ_WITHOUT_AUTH=1` relaxes this for ops/monitoring use.

### 4.3 UI surface

- `/login` route — centered form, submits to `/v1/auth/login`, stores nothing client-side
  (session is cookie-based).
- Shell gets a session loader: on app mount, `GET /v1/auth/me` → if 401, redirect to `/login`;
  if 200, cache the user in a React context.
- Sidebar footer status block (currently static "Routing live / Latency 42ms") gets a new
  row: "Signed in as @username  ·  role" and a Logout action.
- New admin-only screen `Settings → Users`: list, invite, disable. Nav entry visible only
  when `role === "admin"`.
- All rating/edit modals read `judged_by`/`updated_by` from context, not from a form field.

### 4.4 Files touched

```
pyproject.toml                        (+ passlib[bcrypt], + python-multipart already present)
src/tierrouter/service/auth.py        (new: User model, hash_password, verify_password,
                                       require_user dependency, require_admin dependency)
src/tierrouter/service/users_store.py (new: append-only users.jsonl reader/writer)
src/tierrouter/service/app.py         (SessionMiddleware wire-up, new /v1/auth/* and
                                       /v1/users routes, Depends on every mutating route)
src/tierrouter/service/store.py       (plumb session user id through write paths so each
                                       rating/judgment record carries its author)
src/tierrouter/service/bootstrap.py   (new: CLI to create the first admin user)
src/tierrouter/service/__init__.py    (export bootstrap entry point)
src/tierrouter/quality.py             (judged_by description updated: now a user id, not a
                                       free string; existing check "non-empty" still holds)
ui/src/api.ts                         (login, logout, me helpers; response types)
ui/src/auth.tsx                       (new: AuthProvider context, useUser hook)
ui/src/App.tsx                        (wrap in AuthProvider, redirect on 401)
ui/src/screens/Login.tsx              (new)
ui/src/screens/Users.tsx              (new, admin-only)
ui/src/Nav.tsx                        (conditional Users entry, sign-out action)
ui/src/screens/Resolve.tsx            (remove manual judged_by field if any)
ui/src/screens/History.tsx            (rating modal reads user from context, not form)
ui/src/screens/Rules.tsx              (saved-by indicator per rule)
tests/test_auth.py                    (new, see 4.5)
tests/test_users.py                   (new)
tests/test_service.py                 (update: every mutating test logs in first)
```

### 4.5 Tests to add

- `test_auth.py::test_login_with_correct_password_sets_session_cookie`
- `test_auth.py::test_login_with_wrong_password_returns_401`
- `test_auth.py::test_login_with_disabled_user_returns_401`
- `test_auth.py::test_logout_clears_session`
- `test_auth.py::test_me_returns_401_without_session`
- `test_auth.py::test_service_refuses_to_start_without_session_secret` — defense against
  running the server with an empty secret by accident.
- `test_auth.py::test_service_refuses_login_before_bootstrap` — no users → no login possible.
- `test_users.py::test_only_admin_can_create_users`
- `test_users.py::test_cannot_create_user_with_duplicate_username`
- `test_users.py::test_disabled_user_cannot_login`
- `test_service.py::test_post_quality_without_session_returns_401`
- `test_service.py::test_post_quality_ignores_client_supplied_judged_by_and_uses_session` —
  important: the client cannot impersonate even if it tries.
- `test_service.py::test_put_rules_records_updated_by_from_session`
- `test_service.py::test_get_catalog_allows_read_when_relaxed_env_set`

### 4.6 Non-goals for P0 (explicitly)

- No OAuth, no SSO, no email verification, no password reset flow. Password reset is a CLI
  reissue by an admin (`python -m tierrouter.service.bootstrap --reset <username>`).
- No workspaces (that was Level C, deferred).
- No per-permission ACLs beyond `admin`/`member`. Two roles is enough for this scope.
- No audit log of logins. Add later if there is a concrete need.

## 5. P2 — Accuracy-first selection mode

**Why this is second:** small, purely additive, directly answers "agar kisko bahut jaida
accuracy chahiye." With P0 in place, Decision records can attribute the mode to a user.

### 5.1 Design

Add one parameter to `catalog.pick()` and `resolve()`:

```
optimize_for: Literal["cost", "accuracy", "balanced"] = "cost"
```

Behaviour:

| Mode | Scoring function (within the decided tier, after filters) | Tiebreaker |
|---|---|---|
| `cost` | min `cost` | lexical `id` for determinism |
| `accuracy` | max `quality[(id, value)]` | min `cost` |
| `balanced` | max `quality / log1p(cost + 1)` | min `cost` |

Edge cases, each with a visible `note` in `Decision.reason`:

- **Accuracy mode, no quality ratings at all** → fall back to `cost` mode with the note
  `"optimize_for=accuracy requested but no ratings exist yet; used cost instead"`. (G4.)
- **Accuracy mode, quality ratings exist but none for this `(task, value)`** → same fallback,
  different note: `"no ratings for task=<value>; used cost"`.
- **`min_quality` + `optimize_for=accuracy`** → compose. First filter out ratings below floor
  (existing behaviour), then pick highest rated remaining. G4 still holds: unreviewed models
  stay in the candidate pool.
- **Balanced mode** is documented as a heuristic in the docstring, not a theoretical optimum.

### 5.2 API surface

- `src/tierrouter/catalog.py`: `pick(..., optimize_for)` — the only non-cost branches live here.
- `src/tierrouter/resolve.py`: `resolve(..., optimize_for)` — thin passthrough.
- `src/tierrouter/service/app.py`: `/v1/resolve` gains `optimize_for: str = "cost"` form field.
  Validated against the three allowed literals; 422 on anything else.
- `src/tierrouter/rules.py`: `Rule` gains optional `optimize_for` field. When a rule matches,
  its `optimize_for` wins over the request's — operators can pin accuracy mode for high-
  stakes tasks (e.g. QC on video frames, legal summarisation) at the rule layer, once.
- `ui/src/types.ts`: `Decision` unchanged (reason already carries notes). `ResolveRequest`
  and `Rule` gain `optimize_for?: 'cost' | 'accuracy' | 'balanced'`.
- `ui/src/screens/Resolve.tsx`: 3-way toggle above the "Decide" button, default `cost`.
- `ui/src/screens/Rules.tsx`: per-rule optional `optimize_for` field.

### 5.3 Files touched

```
src/tierrouter/catalog.py        (pick + Decision reason threading)
src/tierrouter/resolve.py        (new param, pass through, rule-override precedence)
src/tierrouter/rules.py          (Rule.optimize_for + YAML load/save)
src/tierrouter/service/app.py    (new form field on /v1/resolve)
ui/src/types.ts                  (ResolveRequest, Rule types)
ui/src/api.ts                    (resolve() signature)
ui/src/screens/Resolve.tsx       (toggle)
ui/src/screens/Rules.tsx         (per-rule control)
tests/test_catalog.py            (new cases)
tests/test_resolve.py            (new cases)
tests/test_rules.py              (optimize_for serialisation)
tests/test_service.py            (new /v1/resolve form-field case)
```

### 5.4 Tests to add

- `test_catalog.py::test_pick_accuracy_picks_highest_rated_when_ratings_exist`
- `test_catalog.py::test_pick_accuracy_falls_back_to_cost_with_note_when_no_ratings`
- `test_catalog.py::test_pick_accuracy_respects_min_quality_floor`
- `test_catalog.py::test_pick_accuracy_unreviewed_model_still_eligible` — G4 regression.
- `test_catalog.py::test_pick_balanced_prefers_cheap_when_quality_ties`
- `test_catalog.py::test_pick_never_downgrades_below_decided_tier_in_any_mode` — G2 regression
  across all three modes.
- `test_resolve.py::test_resolve_threads_optimize_for_through_to_catalog`
- `test_resolve.py::test_rule_optimize_for_overrides_request_optimize_for`
- `test_rules.py::test_rule_optimize_for_roundtrips_through_yaml`
- `test_service.py::test_resolve_endpoint_accepts_optimize_for_cost_accuracy_balanced`
- `test_service.py::test_resolve_endpoint_rejects_invalid_optimize_for_with_422`

## 6. P3 — Modality expansion + human QC loop in UI

**Why this is third:** without a way to *collect* ratings from a UI, P2's accuracy mode has
nothing to optimise on. Modality coverage (text/image/video/audio) matters for all content
types, not just video — framed per the call transcript.

### 6.1 Design — modality

Extend `CatalogEntry` with:

```python
supports_video: bool = False
supports_audio: bool = False
```

`capability.py` already parses `architecture.input_modalities`. Extend the extractor to also
look for `"video"` and `"audio"` tokens — preserving the current `supports_image` logic
unchanged.

`catalog.pick()` gains `needs_video: bool = False`, `needs_audio: bool = False`. Same filter
shape as `needs_image`.

`resolve()` infers `needs_video`/`needs_audio` from `metadata` keys:

- If `metadata["modality"] == "video"` → `needs_video=True`.
- If `metadata["modality"] == "audio"` → `needs_audio=True`.

This is a **convention**, not magic — documented in the `resolve()` docstring. Operators who
use a different key pass it explicitly to `pick()` from their own glue code.

### 6.2 Design — UI rating loop

- `ui/src/screens/History.tsx`: every history row gets a "Rate" action. Opens a modal with:
  1-5 stars, optional `notes`, and a dropdown for `task` (populated from
  `TIERROUTER_QUALITY_FIELD` on the server via a new `/v1/quality/field` endpoint, or hidden
  if no field is configured). `judged_by` is NOT a form field — the server records
  `session.user.id` (P0).
- On submit → `POST /v1/quality` with `model_id` = the row's `model_name`, `metadata` = the
  row's recorded `metadata` (same `canonical_value` normalisation applies on both sides).
- `ui/src/screens/Catalog.tsx`: existing average-rating pill stays; add a "Modalities" row
  showing `text / image / video / audio` chips based on the four booleans (text is always
  implicitly on).

### 6.3 Files touched

```
src/tierrouter/catalog.py            (CatalogEntry fields, pick filters)
src/tierrouter/capability.py         (parse video + audio from input_modalities)
src/tierrouter/resolve.py            (infer needs_video/audio from metadata)
src/tierrouter/service/app.py        (expose supports_video/audio in /v1/catalog;
                                      new /v1/quality/field endpoint)
experiments/refresh_catalog.py       (sync the two new booleans like supports_image)
config/catalog.example.yaml          (add supports_video/audio example rows)
ui/src/types.ts                      (CatalogEntry fields)
ui/src/api.ts                        (quality POST helper, quality/field GET)
ui/src/screens/History.tsx           (Rate action + modal)
ui/src/screens/Catalog.tsx           (modality chips)
tests/test_capability.py             (video + audio parsing from a mock payload)
tests/test_catalog.py                (needs_video / needs_audio filters)
tests/test_resolve.py                (metadata-driven needs_video inference)
tests/test_service.py                (POST /v1/quality shape unchanged; GET /v1/catalog
                                      includes new fields)
```

### 6.4 Tests to add

- `test_capability.py::test_parses_video_and_audio_from_input_modalities`
- `test_capability.py::test_absent_modalities_default_to_false` — not a crash.
- `test_catalog.py::test_pick_filters_out_entries_without_video_when_needs_video`
- `test_catalog.py::test_pick_raises_rule_error_when_no_video_candidate_in_filter`
- `test_resolve.py::test_metadata_modality_video_sets_needs_video_on_pick`
- `test_resolve.py::test_metadata_modality_audio_sets_needs_audio_on_pick`
- `test_service.py::test_catalog_endpoint_exposes_modality_booleans`
- `test_service.py::test_quality_field_endpoint_returns_configured_field_or_null`
- `test_service.py::test_rating_modal_judged_by_is_session_user` (regression against
  client-supplied impersonation)

## 7. P1 — Dynamic OpenRouter import + UNREVIEWED tier

**Why this is last:** biggest surface change. With a 5-entry catalog today, discovery matters
less until P2 and P3 are working.

### 7.1 Design — import flow

New script: `experiments/sync_catalog.py` (**merged with `refresh_catalog.py`** per Q6.5.a).
The old `refresh_catalog.py` stays as a 3-line shim that calls into the new script with
`--only-existing`, for anyone with old habits.

```
python experiments/sync_catalog.py \
    --out config/catalog.yaml \
    --providers openai,anthropic,google,meta-llama,qwen,deepseek,mistralai \
    [--modalities image,video] \
    [--only-existing | --add-new] \
    [--dry-run]
```

- Pulls full OpenRouter list via the existing `capability.py`.
- With `--only-existing`: behaves like today's `refresh_catalog.py` — facts only, no new rows.
- With `--add-new`: for each model NOT already in the target YAML AND matching the filters,
  emits a new entry with:
  - `id`: OpenRouter's bare id (after the `provider/` prefix), same convention as
    `match_by_suffix`.
  - `provider`: OpenRouter's provider namespace.
  - `tier: UNREVIEWED` — the new sentinel.
  - `cost`, `context_length`, `supports_image`, `supports_video`, `supports_audio`: live.
  - `label`: OpenRouter's human name if present, else the id.
- Default mode (neither flag): errors with a usage message (Q6.5.c).
- Default `--providers` (unset): errors with a usage message listing common providers.

### 7.2 Design — UNREVIEWED sentinel

- `TierTable.names` keeps meaning "declared, routable tiers."
- Add a constant `UNREVIEWED_TIER = "UNREVIEWED"` in `tiers.py`.
- `catalog.load_catalog`: accepts `UNREVIEWED` as a tier string without complaint, same way
  it accepts any other string today. The validator check happens in `pick()`.
- `catalog.pick()`: entries with `tier == UNREVIEWED` are **always excluded** from the
  candidate pool, regardless of `selected_models`. A reviewer must assign a real tier before
  `resolve()` will pick a model. This is G5 enforced in code, not comment.
- `/v1/catalog` GET: continues to return all entries (so the UI can show the pending list);
  decision-time filtering is unchanged (unreviewed entries just never win).

### 7.3 Design — UI review flow

- `ui/src/screens/Catalog.tsx`: top-of-page banner "N models pending review" when any entry
  has `tier == UNREVIEWED`. Clicking it filters the master list to those rows.
- On an unreviewed row's detail panel: a tier picker (`small`/`medium`/`large` — read from
  `/v1/tiers`) + "Assign" button → `PUT /v1/catalog/{id}` with the new tier.
- On an already-reviewed row: a "Demote to UNREVIEWED" button in the detail panel's menu
  (Q6.5.b). Does not delete the row or its rating history; just removes it from routing
  until re-approved.
- `PUT /v1/catalog/{id}`: new endpoint. Admin-only (P0). Updates **only** that entry's
  `tier`. Persists to `catalog.yaml` on disk (mirroring how `/v1/rules` already persists via
  `store`). Records `assigned_by = session.user.id` in the row.

### 7.4 Files touched

```
experiments/sync_catalog.py          (new; merges refresh_catalog.py + import semantics)
experiments/refresh_catalog.py       (shim, 3 lines, deprecates to sync_catalog.py)
src/tierrouter/catalog.py            (UNREVIEWED filter in pick; CatalogEntry gains
                                      assigned_by: str = "")
src/tierrouter/tiers.py              (UNREVIEWED_TIER constant; export)
src/tierrouter/service/app.py        (PUT /v1/catalog/{id} admin-only;
                                      store.update_catalog_tier)
src/tierrouter/service/store.py      (update_catalog_tier + file write)
ui/src/types.ts                      (tier type now includes "UNREVIEWED")
ui/src/api.ts                        (updateCatalogTier helper)
ui/src/screens/Catalog.tsx           (pending-review banner, tier assignment, demote)
ui/src/theme.ts                      (tone for UNREVIEWED — neutral grey, distinct from the
                                      3 tier tones so it never looks like a routable choice)
tests/test_catalog.py                (UNREVIEWED is filtered out by pick in every mode)
tests/test_resolve.py                (resolve raises/reports when every candidate at the
                                      decided tier is UNREVIEWED)
tests/test_service.py                (PUT /v1/catalog/{id}: 200 on known id as admin,
                                      403 as member, 404 on unknown, 422 on invalid tier)
tests/test_sync_catalog.py           (new: --add-new emits UNREVIEWED rows;
                                      --only-existing matches old refresh_catalog behaviour)
```

### 7.5 Tests to add

- `test_catalog.py::test_pick_excludes_unreviewed_in_cost_mode`
- `test_catalog.py::test_pick_excludes_unreviewed_in_accuracy_mode`
- `test_catalog.py::test_pick_excludes_unreviewed_in_balanced_mode`
- `test_catalog.py::test_pick_excludes_unreviewed_even_if_in_selected_models`
- `test_resolve.py::test_resolve_reports_when_every_candidate_is_unreviewed`
- `test_service.py::test_put_catalog_tier_as_admin_updates_and_persists`
- `test_service.py::test_put_catalog_tier_as_member_returns_403`
- `test_service.py::test_put_catalog_tier_unknown_id_returns_404`
- `test_service.py::test_put_catalog_tier_invalid_tier_returns_422`
- `test_sync_catalog.py::test_add_new_emits_unreviewed_entries_for_unknown_models`
- `test_sync_catalog.py::test_only_existing_leaves_tier_untouched` — regression vs. today's
  `refresh_catalog.py`.

## 8. Ordering, dependencies, and what unblocks what

```
P0 (auth, Level B) ──┐
                     ├──► Every mutating endpoint now carries user id
                     ▼
              P2 (accuracy-first mode) — purely additive, cost mode stays default
                     │
                     ▼
              P3 (modality + QC loop) — rating modal uses session user (P0)
                     │
                     ▼
              P1 (dynamic import)     — PUT catalog gated on admin role (P0)
```

- **P0 ships first and alone.** Nothing in P2/P3/P1 is interesting without it.
- **P2 ships second.** `optimize_for=cost` stays the default, so no behavioural change for
  any existing caller; new UI toggle + Decision reason notes are the only visible lift.
- **P3 ships third.** Adds data entry; no routing change.
- **P1 ships last.** Lowest payoff per unit of UX work until there's a bigger catalog.

Each piece is a separate commit + separate PR. If something regresses, we revert that one
piece without unwinding the others.

## 9. Validation (added to the project's existing commands)

```bash
ruff check . && mypy src
pytest -q                                      # everything, including the new tests in each P
pytest -q tests/test_auth.py                   # P0 subset
pytest -q tests/test_catalog.py -k accuracy    # P2 subset
pytest -q tests/test_capability.py -k video    # P3 subset
pytest -q tests/test_sync_catalog.py           # P1 subset
```

End-to-end smoke on each piece (manual, documented in the PR body, not a CI job):

- **P0**: run `python -m tierrouter.service.bootstrap --username admin` → set a password →
  start service → UI redirects to `/login` → log in → sidebar shows "@admin, admin" → create
  a `member` user → log in as that member → verify Users screen is not visible from nav.
- **P2**: logged in → `/v1/resolve` with `optimize_for=accuracy` → verify `Decision.reason`
  names the mode and the fallback note if no ratings exist.
- **P3**: rate a decision from History → refresh Catalog → see the average update on that
  model's detail panel; `judged_by` on the server-side record equals the logged-in user's id.
- **P1**: run `python experiments/sync_catalog.py --add-new --providers openai,qwen --dry-run`
  → see diff → run without `--dry-run` → UI Catalog shows "N pending review" → assign tiers
  as admin → the newly-assigned models become selectable in Playground.

## 10. What this plan does NOT do

- **No live-call capability lookup from `resolve()`.** G1 holds.
- **No change to the classifier's contract.** G7 holds. Tiers stay a 3-way label;
  `UNREVIEWED` is a catalog-level concept, never a classifier output; `optimize_for` is a
  catalog-level selection, never a classifier parameter.
- **No OAuth / SSO / password reset.** Level B means session auth with admin-driven user
  management, nothing more (§4.6).
- **No workspaces.** That was Level C, deferred until there's evidence of multiple teams.
- **No model-calling adapter.** LangChain integration stays in `llm_stub.py` as a factory
  stub (PLAN.md Task 7, call transcript "Factory pattern over integrations to be built
  later").
- **No ratings-driven *automatic* tier assignment.** Even in P1, a human (admin) assigns
  the tier. G5 holds.
- **No modality-specific feature-vector change.** The feature builder stays modality-
  agnostic; the thumbnail slot and image embedder are the only non-text features, matching
  ARCHITECTURE.md §4. Video/audio support is a catalog-side capability flag, not a
  classifier input.

## 11. Operator decisions — all six confirmed

| # | Decision | Answer |
|---|---|---|
| Q4.5 | Does a Rule carry `optimize_for`? | **Yes** |
| Q5.5.a | Rating modal `judged_by` source | **Level B — session auth, per-user audit** |
| Q5.5.b | Let reviewer re-tag `task` at rating time? | **No** |
| Q6.5.a | Merge refresh + import into one script? | **Yes** — `experiments/sync_catalog.py` with `--only-existing` and `--add-new`; `refresh_catalog.py` becomes a shim |
| Q6.5.b | Allow demote back to UNREVIEWED via UI? | **Yes** |
| Q6.5.c | Default provider filter on import when none passed? | **Error** with a helpful usage listing common providers |

Status: **P0 shipped** (`deb2fc5`). **P2 shipped** (`8043cd9`). **P3 shipped**
(modality flags + History Rate modal). Next: **P1 — Dynamic OpenRouter import + UNREVIEWED**.
