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
| `skills.py` | `request_needs()` → `SkillNeeds`: 0–1 need per skill (video, audio, hindi, reasoning, long_input, writing, fast) with reasons, from the task profile in `config/task_skills.yaml`, modality, channel type, `language`, any Indian script in the text or screen text, QC risk, length, `duration_s` and an optional `VideoProfile` (speech / singing → audio, spoken language → hindi, content type → difficulty); difficulty reads only the request's own text, not figures on screen; `load_task_skills` |
| `video_profile.py` | `VideoProfiler.profile(path)` → `VideoProfile` cached by sha256 in `video_profiles/`: duration, speech share and turns (Silero VAD), language (Whisper small over 3 speech windows, kept at ≥ 0.5), sound groups (AudioSet AST over 6 × 10 s windows), scene changes / black / frozen (histogram correlation of grey frames every 2 s), text boxes per frame (EasyOCR detect, optional), `classify_content` → news / music / reality / talk / drama / ads / unknown with reasons. Models load lazily; tests pass stand-ins |
| `rules.py` | `Rule` / `RuleSet`: metadata match → tier, most specific wins, optional `optimize_for` |
| `tiers.py` | `TierTable` from `tiers*.yaml`: tier order, per-provider fallback model, `low_confidence_fallback` |
| `catalog.py` | `ModelCatalog`: filter by tier / scope / provider / modality / quality / context window, audio (need ≥ 0.6 → must hear), `deadline="now"` (no `:batch`), `exclude_free` (no `:free`; each filter keeps all when it would empty the pool), `pick_ranked` by fit (with skill needs) or by mode (without); tier labelling helpers |
| `feedback.py` | `learn_skills(ratings)` → model → skill → `SkillEvidence` (need-weighted mean score, weight, count); `blend` with the catalog value as a prior worth `PRIOR_WEIGHT` = 3 ratings; `fast` is never learned |
| `neighbours.py` | `NeighbourIndex.assess(text, rows)` → `NeighbourVotes`: the K = 5 most similar reviewed requests (cosine ≥ 0.3; MiniLM when loaded, else character n-gram hashing) vote Σ sim × (score − 3)/2 ÷ Σ sim × n/(n + 3) per model; gated by 50 reviewed requests and a leave-one-out agreement ≥ 0.6 (`auto`) |
| `reviews.py` | `Preference` (pick vs another model: picked / other / tie, `winner`, `loser`) and `NeedLabel` (corrected needs), validated at construction (`ReviewError`) |
| `fit.py` | `fit()` = Σ need × model skill + q × (0.5 + difficulty) × capability + 0.5 × similar-request vote (when active) − λ × price (q unscaled when difficulty is None); reasoning and writing count up to `enough_skill` = 0.4 + 0.6 × difficulty; `:batch` endpoints have `fast` 0; `model_skills` (catalog → 0–1 per skill, estimated × 0.85), `MODE_WEIGHTS` per `optimize_for`, `rank_by_fit`, `describe_fit` |
| `capability.py` | Parse OpenRouter's public model list (price, context, image/video/audio support) — offline scripts only |
| `quality.py` | Human quality ratings keyed by `GRADIST_QUALITY_FIELD` |
| `labels.py` | Rule-derived seed labels for training |
| `resolve.py` | The ladder: `resolve()` → `Decision`; `tier_by_need`, `has_domain_signal`; eases an `easy_clip` one tier down (`signals.eased_from`) unless QC risk raised it |
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
| `video_profile_id` | string | Id from `POST /v1/video/profile`; its speech, language and content type feed the skill needs (422 malformed, 404 unknown) |
| `deadline` | `now` \| `offline` | `now` drops `:batch` endpoints; omitted = no filter |

`Decision` fields:

| Field | Meaning |
|---|---|
| `model_name`, `tier`, `reason`, `confidence`, `optimize_for` | The answer |
| `request_id` | Id of the history entry; rate the pick with `POST /v1/quality {request_id, rating}` |
| `rule_outcome`, `classifier_outcome`, `fallback_outcome` | What each ladder step did |
| `tier_scores`, `tier_scores_kind` | Evidence for the tier (classifier probabilities or quality-head scores) |
| `candidates`, `pool_size` | Top 8 of the tier after ranking (each with `fit`, `fit_parts` per term, `skill_sources` and `learned` {skill: prior, value, ratings}), and the pool size |
| `signals` | What was received: text size, metadata used/ignored, image bytes, modality, providers, `screen_text` {engine, lines, frames_read}, `classifier_inputs` {metadata buckets, embedders, `image_used`, `screen_text_used`, odds `without_image` / `without_text` / `without_screen_text` / `without_metadata`}, QC-risk score, `skill_needs` {needs, reasons, task_profile, input_tokens, difficulty, difficulty_reasons} — what the candidates are ranked against, `similar_requests` {active, status, agreement, votes {model: {vote, neighbours}}, neighbours [{similarity, model_id, score}]}, `moved_by_ratings` [{id, rank, fit, learned}] — up to 5 models in the tier whose skills ratings changed, with their rank even outside the top 8, `video_profile` (the profile used), `deadline` |

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
| `GRADIST_NEIGHBOURS` | no | Similar-request votes: `auto` (default: on after 50 reviewed requests and ≥ 60% held-out agreement), `on`, `off` |
| `GRADIST_FREE_MODELS_FOR_VIDEO` | no | `1` allows `:free` tiers for video requests (excluded by default) |
| `GRADIST_EASE_EASY_VIDEO` | no | `0` turns off easing: by default a video whose needs carry `easy_clip` goes one tier below the decided tier |

## Storage (`GRADIST_DATA`, gitignored)

| Path | Contents |
|---|---|
| `examples.jsonl`, `judgments.jsonl` | Training examples `{example_id, metadata, text, image_path}` and tier labels |
| `images/`, `thumbnails/`, `video_clips/` | Example frames and source media |
| `classifier.joblib`, `classifier.meta.json` | Tier classifier and how to rebuild its embedders (`text_embedder`, `uses_images`, `reads_screen_text`, fields) |
| `quality_head.joblib`, `qc_risk.joblib` | The two other learned heads (loaded at start if present) |
| `ocr_cache.json` | OCR lines by image sha1 |
| `video_profiles/<sha256>.json` | Video profiles by file hash (`version` must match `PROFILE_VERSION`) |
| `rules.yaml`, `task_pools.yaml` | Operator rules and named model pools |
| `quality.jsonl` | Human quality ratings; rows rated by `request_id` also carry that request's `needs` and optional `wrong_skills` |
| `preferences.jsonl` | Comparisons `{request_id, picked, other, outcome: picked/other/tie, skills, needs, judged_by}`; not read at resolve yet |
| `need_labels.jsonl` | Corrected needs `{request_id, needs, rule_needs, judged_by}`; not read at resolve yet |
| `history.jsonl` | Every resolve: `request_id`, request summary, `skill_needs`, `difficulty`, `shortlist` (top candidate ids), `video` {profile_id, content_type, language, speech_share}, `deadline`, decision. `GET /v1/history` adds `ratings`, `preferences` and the latest `need_label` |
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
| `GET /v1/quality/field`, `GET /v1/quality` · `POST /v1/quality` | read · user | Quality ratings. `POST {request_id, rating}` rates that request's pick: model, metadata and needs come from history (404 unknown id, 403 someone else's request for members) and feed the learned skills; optional `wrong_skills` |
| `POST /v1/preferences` · `GET /v1/preferences` | user · read | `{request_id, other_model, outcome, skills?, notes?}`: was the pick better than another catalog model? Same 404/403 checks |
| `POST /v1/need-labels` · `GET /v1/need-labels` | user · read | `{request_id, needs}`: what the request really needed (0–1 per skill). Same 404/403 checks |
| `GET /v1/examples`, `GET /v1/examples/uncovered` · `POST /v1/examples` | read · user | Training examples |
| `POST /v1/judgments` | user | Label an example |
| `POST /v1/train` | user | Refit the classifier (OCR applied when enabled) |
| `GET /v1/status` | read | Trained state, label count, which heads are loaded, deployment facts |
| `POST /v1/ocr` | user | Read on-screen text from up to 8 `frames` and cache it (the Playground calls this as soon as a clip is picked) |
| `POST /v1/video/profile` · `GET /v1/video/profile/{id}` | user | Profile an uploaded `video` (cached by hash; 501 without the `video` extra, 422 if it cannot be decoded) · read a saved profile |
| `POST /v1/resolve` | user | Make a decision (see above) |
| `GET /v1/history` | user | Recent decisions — members get only their own; admins get everyone's, or one user's with `?requested_by=` |

Interactive schema: <http://localhost:8000/docs>.

## Console (`ui/`)

Sidebar (**Platform**): Overview, Routing rules, Request logs, Playground. User menu: Documentation, Settings, and Operators for admins. The training screens
(Examples, Human judging, Train, Evaluate) remain in the code but are hidden from navigation;
their API endpoints stay available. The catalog is managed through the API and the Playground's
model picker.
