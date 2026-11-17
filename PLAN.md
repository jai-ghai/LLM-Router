# Plan: llm-tier-router

See ARCHITECTURE.md for the design this plan builds. Status: final for Tasks 1–3; Task 4 onward
is blocked on the labeling decision below.

## Summary

Build a general-purpose, 3-tier model-selection classifier: metadata + text (+ a reserved
thumbnail slot) in, a tier out, then a config lookup to a real model name. No model is called
here. Usable standalone or imported by a consumer such as a BARC service.

## Complexity and files

Complexity: Medium. About 14 files for Tasks 1–6 (feature builder, rules, classifier, resolver,
training/eval scripts, tests). Thumbnail support and the LangChain stub are interface-only in
this phase — no implementation files yet.

## NOT building (restated from ARCHITECTURE.md §1)

- Anything that calls a model, verifies output, escalates, or traces a call.
- A LangChain integration (a one-file stub interface only, Task 7).
- A service or UI.
- Real thumbnail embedding.

## Mandatory reading before Task 4

- This file's "Labeling strategy" section — the decision gates Task 4.
- `R&D/llmrouter/src/llmrouter/rules.py` — the rule design Task 3 ports (read-only reference;
  that project is not modified).
- `R&D/docs/adr/0012-multilingual-input.md` — why `bge-m3` was chosen for text; Task 2 reuses it.

## Step-by-step tasks

### Task 1: Project skeleton
- **ACTION**: `pyproject.toml` (deps: `pydantic`, `pyyaml`, `numpy`, `scikit-learn`, `joblib`;
  optional extra for `sentence-transformers`). `src/tierrouter/__init__.py`. `config/tiers.example.yaml`
  with the tier → model-per-provider shape from ARCHITECTURE.md §5, every value a `REPLACE_ME`.
- **VALIDATE**: package installs editable; example config parses as YAML.

### Task 2: Feature builder (`src/tierrouter/features.py`)
- **ACTION**: `MetadataEncoder`: built from a caller-supplied `{field_name: [known_values]}`,
  one-hot per field plus one "unknown" column per field for any value not in the list, including
  a missing field. `TextEmbedder`: same interface shape as the earlier project's
  `Embedder` protocol (`HashingEmbedder` for tests, `SentenceTransformerEmbedder("BAAI/bge-m3")`
  for real use). A `ThumbnailSlot` placeholder that returns an explicit "not implemented" marker
  rather than silently zeros, so a caller cannot mistake absence for a real feature.
- **GOTCHA**: never raise on an unseen metadata value — route it to "unknown"; do raise on an
  unseen *field name* someone forgot to declare.
- **VALIDATE**: unseen value → unknown bucket, not an error; a request missing a field still
  produces a correctly-shaped vector.

### Task 3: Rules (`src/tierrouter/rules.py`)
- **ACTION**: Port the earlier project's `Rule`/`RuleSet` (attribute matching, most-specific-wins,
  YAML load/save) trimmed to only what this phase needs: no `RuleStore` history/versioning yet
  (that was for a live-editable service, which does not exist here), keep `validate()` against
  the tier names (not a model registry, since there is no registry here — only `tiers.yaml`).
- **VALIDATE**: same behavior as the earlier project's `test_rules.py` cases, adapted.

### Task 4: Labeled training data loader (`src/tierrouter/labels.py`) — BLOCKED
- **ACTION**: Depends entirely on which labeling strategy is chosen (see below). Do not start
  until that choice is made; the record schema differs by strategy.

### Task 5: Tier classifier (`src/tierrouter/classifier.py`)
- **ACTION**: 3-class classifier over the Task 2 feature vector (start with multinomial logistic
  regression — same reasoning as the earlier project: MMR-Bench found simple routers matched
  complex ones). `fit`, `predict` (tier + confidence), `save`/`load` via `joblib`, same "only load
  files from your own artifact store" caveat as the earlier project (`joblib` is `pickle`).
- **GOTCHA**: a confidence threshold below which the classifier should not be trusted — define it,
  make it configurable, and have a documented fallback tier for low confidence.
- **VALIDATE**: beats "always medium" on a held-out split; reports per-metadata-field accuracy,
  including the "unknown" bucket specifically.

### Task 6: Resolver (`src/tierrouter/resolve.py`)
- **ACTION**: `resolve(request, rules, classifier, tier_map) -> Decision(tier, model_name,
  confidence, reason)`. Rule match wins outright (reason: "rule <description>"); otherwise the
  classifier decides (reason: "classifier, confidence X.XX"); low confidence falls back to the
  configured default tier (reason says so explicitly).
- **VALIDATE**: a rule always overrides the classifier; a low-confidence classification is visibly
  marked, not silently accepted.

### Task 7: LangChain stub (`src/tierrouter/llm_stub.py`)
- **ACTION**: One factory function, `get_client(model_name: str) -> None`, docstring only: "Not
  implemented. When the team is ready to call models, this returns a LangChain chat model for
  `model_name`." No LangChain import, no dependency added.
- **VALIDATE**: importable; calling it raises `NotImplementedError` with that message.

### Task 8: Training and evaluation scripts (`experiments/`)
- **ACTION**: `train_classifier.py` (loads labels via Task 4's loader, fits, saves). `evaluate.py`
  (classifier vs. always-medium vs. oracle, per-field accuracy, tier distribution — same shape as
  the earlier project's Phase 0 report).
- **VALIDATE**: reproducible on a fixed seed; stops with a clear message if the labeled set is too
  small to split (same floor as the earlier project: at least 4 examples).

## Labeling strategy: options

The classifier needs, per training example, which tier was actually sufficient. Nothing in this
repository produces that today. Options, roughly cheapest-to-start to most-work:

| # | Strategy | How it works | Needs | Main risk |
|---|---|---|---|---|
| 1 | **Rule-derived seed labels** | Turn today's informal knowledge ("channel X is hard", "anchor Y is straightforward") into the Task 3 rules, and use the rule's tier as a first label for examples it covers | No new data collection | Encodes existing bias; only labels what someone already suspects, so the classifier cannot learn anything the rules didn't already know |
| 2 | **Small human-judged golden set** | A person looks at ~100–300 examples spread across channels/anchors/types and tags the tier each one genuinely needs | A few hours of someone's time, and a clear definition of "sufficient" | Small N is noisy; needs stratified sampling, not a random grab, or rare cases are invisible |
| 3 | **Oracle-by-running** (what the earlier project's Phase 0 planned for video) | Run each example through all 3 tiers, score each output (by a human or a rule), label = cheapest tier that passed | Access to all 3 real models + a quality check | Costs real API calls for every labeled example; needs the "did it pass" check to be trustworthy, which is its own small project |
| 4 | **LLM-as-judge distillation** (RouteLLM's approach) | A strong model judges whether the cheap tier's output is as good as the expensive tier's; label = smallest tier the judge accepted | A judge model + the same two outputs as option 3 | The judge can be biased or miscalibrated; still needs outputs from multiple tiers, so it does not avoid the cost of option 3, it only avoids a human reviewer |
| 5 | **Weak supervision from existing downstream signal** | Use a signal that already exists for a reason unrelated to this classifier — for BARC, QC correction rate per channel/content-type from `sub_story_log_detail` (read-only reference, not modified) — as a proxy: heavily-corrected content probably needed a bigger tier | No new labeling work; reuses data that already exists | A proxy, not a direct label — correlation between "got corrected" and "needed a bigger model" is unverified; worth checking before trusting it |
| 6 | **Production feedback loop (active learning)** | Ship with a rule-based or cautious baseline, log real outcomes (success/failure, or a caller-reported result) over time, and use that as labels to train and improve the classifier later | A way to deploy something first and a way to receive outcome feedback — neither exists yet | Slowest to produce a trained classifier; needs the earlier project's trace-style logging, which is explicitly not in this phase's scope |

**My recommendation**, since no labels exist today and nothing here should be blocked
indefinitely: start with **1 + 2 together** — rules for what is already known (fast, free),
topped up with a few hundred human-judged examples sampled across the metadata fields so the
classifier also learns from clearly non-rule-covered cases. Treat **3/4 (oracle-by-running)** as
the next step once there is a labeled set worth comparing against, to check whether the human
judgments and the rules actually agree with what the models themselves produce. Hold **6** as the
long-term source once something is deployed — it is the only strategy that keeps improving without
repeated manual labeling, but it needs a deployed thing to produce feedback from.

This choice decides Task 4's record schema (a `TierLabel` needs different fields depending on
whether it is a person's judgment, a rule-covered example, or an oracle run) — so it should be
settled, or at least option 1+2 approved to start, before Task 4 begins.

## Validation commands (once there is code)

```bash
ruff check . && mypy src
pytest -q
python experiments/evaluate.py --data labeled.jsonl
```

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| No labels exist, so no task past Task 3 can be verified | Certain today | High | This plan's whole "Labeling strategy" section; Tasks 1–3 proceed without it |
| Rules encode today's bias into the classifier's seed labels | Medium | Medium | Treat rule-derived labels as a seed, not the final set; compare against the golden set |
| Thumbnail is a stated goal but has no plan yet | Medium | Low for now | Explicitly marked "not implemented" everywhere rather than faked |
| Confusable with the earlier, bigger `R&D/llmrouter` project | Medium | Medium | Separate folder, separate README stating the relationship, no shared code path |

## Confidence

6/10 for Tasks 1–3 (straightforward, no open decisions). Task 4 onward cannot be scored until the
labeling strategy is chosen.
