# Design: llm-tier-router

Status: reflects the code as it stands (library + FastAPI service + UI). Complements
ARCHITECTURE.md (scope, decisions, rationale) and PLAN.md (task-by-task build plan) — this
document describes *how the system is actually built*: components, contracts, data flow, data
models, storage layout, and the API surface. Where this disagrees with ARCHITECTURE.md/PLAN.md on
a decision, those files are the source of truth for *why*; this file is the source of truth for
*what is there*.

## 1. System overview

```
                      ┌─────────────────────────────────────────────────────┐
                      │                     tierrouter (library)              │
                      │                                                       │
   metadata, text,    │   rules.py ──► RuleSet.resolve()  ──┐                 │
   optional image ───►│                                     ├─► resolve.py ──┼──► Decision
                      │   classifier.py ──► TierClassifier ─┘   (precedence:  │    (tier,
                      │        ▲                                rule >        │     model_name,
                      │        │ features.py                    classifier >  │     confidence,
                      │        │ (metadata one-hot + text/image   fallback)    │     reason)
                      │        │  embeddings)                                 │
                      │   tiers.py (tiers.yaml) ── catalog.py (catalog.yaml, optional) │
                      └─────────────────────────────────────────────────────┘
                                          ▲
                                          │ imports, no network call
                      ┌───────────────────┴───────────────────┐
                      │        service/ (FastAPI, optional)     │
                      │  app.py (routes) + store.py (JSONL/YAML  │
                      │  files on disk) + main.py (env wiring)   │
                      └───────────────────┬───────────────────┘
                                          │ HTTP (JSON + multipart)
                      ┌───────────────────┴───────────────────┐
                      │           ui/ (React + MUI, optional)   │
                      │  Overview · Tiers · Rules · Examples ·  │
                      │  HumanJudging · Train · Evaluate ·      │
                      │  Resolve · Catalog · History            │
                      └─────────────────────────────────────────┘
```

Three layers, each independently optional, matching README/ARCHITECTURE.md:

1. **Library** (`src/tierrouter/`) — the only part another project needs to import. No HTTP, no
   database, no model-calling. Pure functions/classes over in-memory data and two or three YAML
   config files.
2. **Service** (`src/tierrouter/service/`) — a FastAPI app wrapping the library for one local,
   trusted operator to manage rules/examples/training without touching YAML or code. Adds a
   flat-file store; still never calls a model.
3. **UI** (`ui/`) — a React/MUI console calling the service's HTTP API. Pure client; no business
   logic duplicated from the library.

## 2. Request/decision data flow

```
Caller supplies:
  metadata: dict[str, str]      e.g. {"anchor": "jane_doe", "channel_type": "news"}
  text: str                     the request's text content
  image_bytes: bytes | None     raw file bytes (optional, only if the classifier uses images)

1. rules.RuleSet.resolve(metadata)
   -> (tier, reason) from the highest (priority, specificity) matching rule, or None
   -> IF MATCH: confidence = 1.0, skip the classifier entirely (operator override always wins)

2. ELSE IF a classifier is available:
   features.build_feature_vector(metadata, text, metadata_encoder, text_embedder,
                                  image_bytes, image_embedder)
     -> metadata one-hot (+ unknown bucket per field)
        ++ text embedding
        ++ [image embedding or zero-vector, has-image flag]   (only if image_embedder is set)
   classifier.TierClassifier.predict(...) -> (tier, confidence)
   IF confidence < min_confidence: tier = tiers.low_confidence_fallback, reason says so
   ELSE: tier = predicted tier, reason = "classifier, confidence X.XX"

3. ELSE (no rule, no classifier): tier = tiers.low_confidence_fallback, confidence = None

4. Tier -> model name:
   IF a ModelCatalog is supplied: catalog.pick(tier, tier_rank, selected, provider, needs_image)
     -> cheapest matching model AT that tier, or the cheapest model at a HIGHER tier if none
        exists at the exact tier within the current filters (never downgrades below the decided
        tier) -> model_name, optional note appended to reason
   ELSE: tiers.TierTable.model_for(tier, provider) -> the single configured model name

Result: Decision(tier, model_name, confidence, reason)   -- returned, never executed.
```

Key invariants enforced in code, not just convention:
- A rule match is never second-guessed by the classifier (`resolve.py`).
- An unknown metadata *value* always lands in a field's `__unknown__` bucket; an unknown
  metadata *field name* (not declared when the encoder was built) raises (`features.py`,
  `MetadataEncoder`) — a typo in a field name cannot silently shrink the feature vector.
- A request with no image is always valid even when the classifier was trained with images: it
  gets a real zero-vector plus an explicit `0.0` flag, never confused with a real embedding that
  happens to be near zero (`features.build_feature_vector`).
- The model catalog only ever substitutes a model at the decided tier or a strictly higher one,
  ranked by `tiers.yaml`'s declared tier order (`catalog.py`, `tiers.TierTable.rank()`).

## 3. Component contracts

| Module | Responsibility | Key types / functions |
|---|---|---|
| `features.py` | Build the numeric feature vector from metadata + text + optional image | `MetadataEncoder`, `Embedder` protocol (`HashingEmbedder`, `SentenceTransformerEmbedder`), `ImageEmbedder` protocol (`Dinov2Embedder`), `build_feature_vector()` |
| `rules.py` | Operator-authored overrides, most-specific-wins | `Rule`, `RuleSet` (`.resolve()`, `.validate()`, `.coverage()`), `rules_from_file()` |
| `tiers.py` | Tier names, descriptions, one model per tier per provider, fallback tier | `TierSpec`, `TierTable` (`.model_for()`, `.rank()`), `load_tiers()` |
| `catalog.py` | Optional multi-model-per-tier catalog, cost-based pick within a tier | `CatalogEntry`, `ModelCatalog` (`.pick()`), `load_catalog()` |
| `labels.py` | Training-label loading/merging per the rule+human labeling strategy | `RawExample`, `TierLabel`, `load_examples()`, `labels_from_rules()`, `uncovered_examples()`, `stratified_sample()`, `export_for_human_judging()`, `load_human_judgments()`, `merge_labels()` |
| `classifier.py` | Fit/predict a 3-tier classifier, report accuracy vs. a baseline | `TierClassifier` (`.fit()`, `.predict()`, `.save()`/`.load()`), `evaluate_classifier()`, `EvalReport` |
| `resolve.py` | Compose the above into one decision | `Decision`, `resolve()`, `validate_setup()` |
| `llm_stub.py` | Explicit "not built yet" marker for the future model-calling layer | `get_client()` (raises `NotImplementedError`) |
| `service/app.py` | HTTP surface over the library, for one local operator | `create_app()`, `AppState` |
| `service/store.py` | Flat-file persistence for the service | `Store`, `guess_image_ext()` |
| `service/main.py` | Environment-variable wiring, process entrypoint | `create()` |

Design rules that hold across the library (not just one module):
- **No module calls a model or makes a network request.** `resolve()` returns a model *name*
  string and stops (ARCHITECTURE.md §1).
- **Encoders/embedders are supplied by the caller, not persisted inside a saved classifier.**
  `TierClassifier.save()`/`.load()` only round-trips the fitted scikit-learn model and the
  "constant tier" fallback; the same `MetadataEncoder`/`Embedder`/`ImageEmbedder` instances must
  be passed again at `load()` time, or predictions silently use a feature vector of a different
  width/meaning than the one the classifier was trained on.
- **`joblib.load` is `pickle`.** Only load a classifier file this process (or a trusted pipeline)
  saved — documented in `classifier.py` and repeated here because it is a real code-execution
  risk, not a style note.
- **Config errors raise immediately and specifically**, not as a generic exception: `RuleError`
  for rules/tiers/catalog, `LabelError` for labeling files, `ClassifierError` for fit/predict
  misuse. Every raise names the bad value and, where useful, the set of valid values.

## 4. Data models

### 4.1 `Decision` (the library's output)
```python
Decision(tier: str, model_name: str, confidence: float | None, reason: str)
```
`confidence` is `1.0` for a rule hit, the classifier's own class probability when it decided, or
`None` when neither a rule nor a classifier produced anything (fallback with nothing trained).

### 4.2 `TierLabel` (one training example, labeled)
```python
TierLabel(example_id: str, metadata: dict[str, str], text: str, tier: str,
          source: Literal["rule", "human"], reason: str, image_path: str | None)
```
`source` is kept through training *and* evaluation — rule-derived and human-judged accuracy are
reported separately (`evaluate_classifier` takes the merged set, but PLAN.md §8's discipline is
to compare them, not pool them blindly, since a rule-derived label only ever repeats what the
rule already encoded).

### 4.3 Config files (all YAML, all company-agnostic — no value is hardcoded in the library)

**`tiers.yaml`** (required)
```yaml
tiers:
  - name: small
    description: Cheapest tier. Simple, low-ambiguity requests.
    models: {gemini: ..., anthropic: ..., bedrock: ...}
  - name: medium
    ...
  - name: large
    ...
default_provider: <one of the provider keys above>
low_confidence_fallback: medium   # must be one of the tier names
```

**`rules.yaml`** (operator-editable, via UI or file)
```yaml
rules:
  - when: {channel_type: [news, live], anchor: jane_doe}   # OR within a field, AND across fields
    tier: large
    priority: 0   # optional, default 0; breaks ties when two rules match with equal specificity
```

**`catalog.yaml`** (optional; additive on top of `tiers.yaml`)
```yaml
catalog:
  - id: <model id>
    provider: <provider name>
    tier: small            # must be a tiers.yaml tier name
    cost: 1                 # a RELATIVE number, any unit the deployer's cost model uses
    supports_image: false
    label: <display name>
```

### 4.4 Service storage layout (`TIERROUTER_DATA`, default `./data`) — flat files, no database
```
data/
├── examples.jsonl     # RawExample rows: example_id, metadata, text, image_path
├── judgments.jsonl    # human TierLabel rows: example_id, tier, judged_by, notes
├── rules.yaml         # the live, UI-editable RuleSet (seeded once from TIERROUTER_RULES_SEED
│                      #   only if this file is still empty — never overwritten after that)
├── classifier.joblib  # last-trained TierClassifier (joblib/pickle — see §3's warning)
├── history.jsonl      # append-only log of every /v1/resolve call and its Decision
└── images/            # uploaded example images, named ex-<id>.<ext>
```

## 5. Service API (FastAPI, `src/tierrouter/service/app.py`)

No auth. Built for one trusted local operator (README, docstring in `app.py`) — put it behind
your own auth layer before exposing it to a network or other people.

| Method | Path | Purpose |
|---|---|---|
| GET | `/healthz` | Liveness check |
| GET | `/v1/tiers` | Tier names, descriptions, per-provider models, fallback tier |
| GET | `/v1/rules` | Current rule set |
| PUT | `/v1/rules` | Replace the rule set (validated against known tier names; `422` on error) |
| GET | `/v1/catalog` | Catalog entries, or `{"catalog": []}` if none configured |
| POST | `/v1/examples` | Add a training example (multipart: `metadata` JSON, `text`, optional `image`) |
| GET | `/v1/examples` | List stored examples, each flagged with whether a rule covers it |
| GET | `/v1/examples/uncovered` | Examples no rule reaches — the human-judging sample pool |
| POST | `/v1/judgments` | Record a human tier judgment for one example |
| POST | `/v1/train` | Merge rule-derived + human labels, fit a fresh classifier, save it, return an eval report |
| GET | `/v1/status` | Whether a classifier is trained, how many labels it last trained on, whether it uses images |
| POST | `/v1/resolve` | Run `resolve()` for one request (multipart: `metadata`, `text`, `provider`, `min_confidence`, `selected_models`, optional `image`); appends to history |
| GET | `/v1/history` | Most recent `/v1/resolve` calls, newest first, `limit` query param |

Notable behavior baked into the routes, not left implicit:
- `/v1/train` infers metadata fields (and their known values) from whatever examples are
  currently stored — it does not need a pre-declared field list the way the raw `MetadataEncoder`
  constructor does.
- The image embedder (`Dinov2Embedder`) is loaded lazily, on first use, behind a lock
  (`AppState.image_embedder`) — server startup never pays its load cost when no example has an
  image yet.
- `/v1/resolve` rejects an image upload with `422` if the currently-trained classifier was not
  trained with any image examples, rather than silently ignoring the image or crashing on a
  width mismatch.
- `TIERROUTER_RULES_SEED` only seeds `rules.yaml` once, when the store's own file is still empty
  — it can never clobber rules an operator has already edited through the UI.

## 6. UI (`ui/`, React + MUI, talks only to the service's HTTP API)

One screen per API concern, no business logic duplicated client-side:

`Overview` · `Tiers` · `Rules` · `Examples` · `HumanJudging` · `Train` · `Evaluate` · `Resolve` ·
`Catalog` · `History`

`ui/src/api.ts` is the single HTTP client boundary; `types.ts` mirrors the service's response
shapes. The UI is a thin operator console for the service — it is not a separate source of truth
for any rule, label, or decision.

## 7. Classifier design

- **Model:** multinomial logistic regression (`sklearn.linear_model.LogisticRegression`), chosen
  over a more complex model per MMR-Bench's finding (ARCHITECTURE.md §3) that simple classifiers
  over embeddings match complex ones for this kind of routing decision.
- **Degenerate case:** a label set with only one distinct tier cannot be fit by
  `LogisticRegression` (it refuses one-class input) — handled explicitly as a "constant"
  classifier that always predicts that tier at confidence `1.0`, rather than erroring.
- **Confidence is the predicted class's own probability**, in `[0, 1]` — not calibrated against
  anything external. `resolve()`'s `min_confidence` threshold is the trust decision, kept
  deliberately outside the classifier itself (ARCHITECTURE.md §1): the classifier only ever
  reports what it believes, never decides whether that belief should be acted on.
- **Evaluation (`evaluate_classifier`)** reports accuracy against a fixed baseline tier
  (`tiers.low_confidence_fallback`, i.e. "always medium") plus per-metadata-field-bucket
  accuracy, including the `__unknown__` bucket specifically — so a classifier that quietly does
  worse on unfamiliar metadata values is visible, not hidden inside an aggregate number.
- **Training/eval scripts** live in `experiments/` (`train_classifier.py`; see `PLAN.md` Task 8
  for `evaluate.py`'s intended shape: classifier vs. always-medium vs. oracle).

## 8. Explicit non-goals (restated from ARCHITECTURE.md §1, kept here for completeness)

- Calling any model. `resolve()`/`Decision` stop at a model *name*.
- A LangChain (or other) calling-library integration — `llm_stub.get_client()` is a one-function
  stub that raises `NotImplementedError` until that phase starts.
- Output verification, escalation, cost tracking, or tracing of an actual model call.
- Multi-user auth/authorization on the service — it is a single-operator tool today.
- A production-grade database — the service store is flat JSONL/YAML files by design.

## 9. Known simplifications and open items

| Item | Current state | Why it is a simplification, not a gap |
|---|---|---|
| Model catalog | `catalog.py` exists, is optional, and resolve.py already supports it alongside `tiers.yaml`'s single-model-per-tier path | README/ARCHITECTURE.md §8 flag a full model-catalog redesign as "under discussion next" — today's catalog is additive, not yet the primary path |
| Thumbnail/image input | Fully implemented (`Dinov2Embedder`, feature vector slot, service upload path, UI `ImagePicker`) | Was a stubbed "later input" in ARCHITECTURE.md §1 at the time it was written; it is real now, not a stub — this document reflects that it shipped |
| Labeling strategy | Strategies 1 (rule-derived) + 2 (human-judged, stratified) implemented in `labels.py` and the service's `/v1/examples`, `/v1/judgments`, `/v1/train` | Strategies 3/4 (oracle-by-running, LLM-as-judge cross-check) and 6 (production feedback loop) remain future phases per PLAN.md |
| `experiments/evaluate.py` | Not yet written (only `train_classifier.py` exists under `experiments/`) | PLAN.md Task 8 describes its intended shape (classifier vs. always-medium vs. oracle) |
| Confidence threshold tuning | `min_confidence` defaults to `0.0` (trusts every prediction); no evidence-based default is set | PLAN.md §8: a real cutoff needs held-out evaluation data first, which depends on the labeling strategy maturing |
