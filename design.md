# mRouter design reference

The "what": modules, data shapes, storage and API. The "how and why" is in
[ARCHITECTURE.md](ARCHITECTURE.md). Code names keep the original package name: Python package
`gradist`, environment variables `GRADIST_*`.

## Modules (`src/gradist/`)

| Module | Responsibility |
|---|---|
| `features.py` | `MetadataEncoder` (one-hot + unknown bucket per field), `SentenceTransformerEmbedder` (MiniLM / bge-m3), `Dinov2Embedder` (384-d image), `HashingEmbedder` (tests), `build_feature_vector` |
| `classifier.py` | `TierClassifier`: fit / `predict_proba` / `explain()` (inputs read + ablated odds); `with_screen_text`; save/load with `classifier.meta.json` |
| `quality_head.py` | `TierQualityHead`: per-tier score regression; cheapest tier within the `optimize_for` margin |
| `qc_risk.py` | `QcRiskHead`: P(needs QC fix) for sub-story-shaped requests; escalation threshold at the riskiest ~20% |
| `ocr.py` | `ScreenReader` (EasyOCR, lazy, sha1-keyed cache, one read per frame under concurrency), `ScreenLine`, `join_lines`, `merge_frame_lines` |
| `skills.py` | `request_needs()` → `SkillNeeds`: 0–1 need per skill (video, audio, hindi, reasoning, long_input, writing, fast) with reasons, from the task profile in `config/task_skills.yaml`, modality, Devanagari share, QC risk and length; `load_task_skills` |
| `rules.py` | `Rule` / `RuleSet`: metadata match → tier, most specific wins, optional `optimize_for` |
| `tiers.py` | `TierTable` from `tiers*.yaml`: tier order, per-provider fallback model, `low_confidence_fallback` |
| `catalog.py` | `ModelCatalog`: filter by tier / scope / provider / modality / quality / context window, `pick_ranked` by fit (with skill needs) or by mode (without); tier labelling helpers |
| `feedback.py` | `learn_skills(ratings)` → model → skill → `SkillEvidence` (need-weighted mean score, weight, count); `blend` with the catalog value as a prior worth `PRIOR_WEIGHT` = 3 ratings; `fast` is never learned |
| `fit.py` | `fit()` = Σ need × model skill + q × capability − λ × price; `model_skills` (catalog → 0–1 per skill, estimated × 0.85), `MODE_WEIGHTS` per `optimize_for`, `rank_by_fit`, `describe_fit` |
| `capability.py` | Parse OpenRouter's public model list (price, context, image/video/audio support) — offline scripts only |
| `quality.py` | Human quality ratings keyed by `GRADIST_QUALITY_FIELD` |
| `labels.py` | Rule-derived seed labels for training |
| `resolve.py` | The ladder: `resolve()` → `Decision`; `tier_by_need`, `has_domain_signal` |
| `llm_stub.py` | Placeholder for a future model-calling layer (raises; not wired) |
| `service/app.py` | FastAPI app factory, `AppState`, all endpoints, `/v1/channels` directory, OCR wiring |
| `service/main.py` | Builds the app from `GRADIST_*` environment variables |
| `service/auth.py` | bcrypt passwords, signed session cookie, `require_user` / `require_admin` |
| `service/store.py`, `users_store.py` | Flat-file persistence under `GRADIST_DATA` |
| `service/bootstrap.py` | CLI: create the first user / reset a password |

## Request and decision

`POST /v1/resolve` is `multipart/form-data`:

| Field | Type | Notes |
|---|---|---|
| `metadata` | JSON object (required) | Any fields, e.g. `channel_code`, `channel_name`, `genre`, `pipeline`, `purpose`, `modality` |
| `text` | string | Request content |
| `image` | file | An image or one video frame; with `modality=video` only video-capable models qualify |
| `ocr_frames` | file, repeatable (≤ 8) | Frames to read for on-screen text instead of `image`; lines are merged across frames |
| `optimize_for` | `cost` \| `balanced` \| `accuracy` | Default `cost` |
| `provider` | string | One or comma-separated providers |
| `selected_models` | JSON array of model ids | Explicit candidate scope |
| `task` | string | Named task pool (used when `selected_models` is absent) |
| `min_confidence` | float | Below this the classifier defers to the fallback |
| `min_quality` | float | Quality floor; needs `GRADIST_QUALITY_FIELD` |

`Decision` fields:

| Field | Meaning |
|---|---|
| `model_name`, `tier`, `reason`, `confidence`, `optimize_for` | The answer |
| `request_id` | Id of the history entry; rate the pick with `POST /v1/quality {request_id, rating}` |
| `rule_outcome`, `classifier_outcome`, `fallback_outcome` | What each ladder step did |
| `tier_scores`, `tier_scores_kind` | Evidence for the tier (classifier probabilities or quality-head scores) |
| `candidates`, `pool_size` | Top 8 of the tier after ranking (each with `fit`, `fit_parts` per term, `skill_sources` and `learned` {skill: prior, value, ratings}), and the pool size |
| `signals` | What was received: text size, metadata used/ignored, image bytes, modality, providers, `screen_text` {engine, lines, frames_read}, `classifier_inputs` {metadata buckets, embedders, `image_used`, `screen_text_used`, odds `without_image` / `without_text` / `without_screen_text` / `without_metadata`}, QC-risk score, `skill_needs` {needs, reasons, task_profile, input_tokens} — what the candidates are ranked against |

## Configuration

| Variable | Required | Purpose |
|---|---|---|
| `GRADIST_SESSION_SECRET` | yes | Signs the session cookie; the API refuses to start without it |
| `GRADIST_TIERS` | yes | Tier table, e.g. `config/tiers.general.yaml` |
| `GRADIST_CATALOG` | no | Multi-model catalog, e.g. `config/catalog.yaml`; without it each tier uses its single model |
| `GRADIST_DATA` | no | Store directory (default `data`) |
| `GRADIST_RULES_SEED` | no | Rules copied in only while the store's rules are empty |
| `GRADIST_QUALITY_FIELD` | no | Metadata field that buckets quality ratings (e.g. `task`) |
| `GRADIST_CORS_ORIGINS` | no | Allowed browser origins (default `http://localhost:5174`) |
| `GRADIST_ALLOW_READ_WITHOUT_AUTH` | no | `1` opens GET endpoints for monitoring; writes stay gated |
| `GRADIST_OCR` | no | `1` reads on-screen text from frames (needs the `ocr` extra) |
| `GRADIST_OCR_LANGS` | no | EasyOCR languages, default `hi,en` |
| `GRADIST_TASK_SKILLS` | no | Skill needs per task, default `config/task_skills.yaml` (skipped if missing) |

## Storage (`GRADIST_DATA`, gitignored)

| Path | Contents |
|---|---|
| `examples.jsonl`, `judgments.jsonl` | Training examples `{example_id, metadata, text, image_path}` and tier labels |
| `images/`, `thumbnails/`, `video_clips/` | Example frames and source media |
| `classifier.joblib`, `classifier.meta.json` | Tier classifier and how to rebuild its embedders (`text_embedder`, `uses_images`, `reads_screen_text`, fields) |
| `quality_head.joblib`, `qc_risk.joblib` | The two other learned heads (loaded at start if present) |
| `ocr_cache.json` | OCR lines by image sha1 |
| `rules.yaml`, `task_pools.yaml` | Operator rules and named model pools |
| `quality.jsonl` | Human quality ratings; rows rated by `request_id` also carry that request's `needs` |
| `history.jsonl` | Every resolve: `request_id`, request summary, `skill_needs`, decision |
| `users.jsonl` | Users with bcrypt hashes, role, enabled flag (UTF-8 without BOM) |

## API

Auth levels: **public**; **read** = signed in, or open when `GRADIST_ALLOW_READ_WITHOUT_AUTH=1`;
**user** = signed in; **admin** = signed-in admin.

| Method and path | Auth | Purpose |
|---|---|---|
| `GET /healthz` | public | Liveness |
| `POST /v1/auth/login`, `POST /v1/auth/logout` | public | Session cookie |
| `GET /v1/auth/me` | user | Current user |
| `GET/POST /v1/users`, `PUT /v1/users/{id}` | admin | Manage operators |
| `GET /v1/tiers` | read | Tier table |
| `GET /v1/rules` · `PUT /v1/rules` | read · user | Routing rules |
| `GET /v1/catalog` | read | Catalog with capability and prices |
| `POST /v1/catalog/label-tiers`, `PUT /v1/catalog/{model_id}` | admin | Assign tiers / review new models |
| `GET /v1/channels` | read | channel_code → name, network, type, genre (from examples) |
| `GET /v1/task-pools` · `PUT /v1/task-pools`, `PUT/DELETE /v1/task-pools/{task}` | read · user | Named model pools |
| `GET /v1/quality/field`, `GET /v1/quality` · `POST /v1/quality` | read · user | Quality ratings. `POST {request_id, rating}` rates that request's pick: model, metadata and needs come from history (404 unknown id, 403 someone else's request for members) and feed the learned skills |
| `GET /v1/examples`, `GET /v1/examples/uncovered` · `POST /v1/examples` | read · user | Training examples |
| `POST /v1/judgments` | user | Label an example |
| `POST /v1/train` | user | Refit the classifier (OCR applied when enabled) |
| `GET /v1/status` | read | Trained state, label count, which heads are loaded, deployment facts |
| `POST /v1/ocr` | user | Read on-screen text from up to 8 `frames` and cache it (the Playground calls this as soon as a clip is picked) |
| `POST /v1/resolve` | user | Make a decision (see above) |
| `GET /v1/history` | user | Recent decisions — members get only their own; admins get everyone's, or one user's with `?requested_by=` |

Interactive schema: <http://localhost:8000/docs>.

## Console (`ui/`)

Sidebar (**Platform**): Overview, Routing rules, Request logs, Playground. User menu: Documentation, Settings, and Operators for admins. The training screens
(Examples, Human judging, Train, Evaluate) remain in the code but are hidden from navigation;
their API endpoints stay available. The catalog is managed through the API and the Playground's
model picker.
