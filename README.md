# Gradist

A small, general-purpose library: given a request's metadata and text (and later, a thumbnail),
predict which **tier** of model should handle it — small, medium, or large. The resolver then
turns that tier into a concrete model name from a config file. It does not call the model itself.

Operator guide (run, integrate, screenshots): [docs/USER_GUIDE.md](docs/USER_GUIDE.md) — also in the
console under **Documentation**.

This replaces the earlier, larger `R&D/llmrouter` project's scope (which also called models,
verified output, escalated, and ran a full service and UI). That project is left as-is and
untouched. This one is deliberately smaller, matching the 2026-11-16 team decision:

- A **learned classifier**, not a rules-only tool like LiteLLM. LiteLLM is still fine as the
  thin API layer underneath, later, once a model is actually called.
- **General-purpose first.** Tapson/BARC is one consumer, not the target. Most company content
  is text; video is secondary.
- **Metadata and text first.** Thumbnails are a planned input, stubbed for now.
- **Resolve only.** `resolve(request) -> (tier, model_name, confidence, reason)`. No calling,
  no verification, no escalation here — that is a separate, later layer (a LangChain-based
  integration layer is stubbed only, per the team's decision not to build it yet).

Status: **built and verified for this scope.** The labeling strategy (PLAN.md, "Labeling
strategy: options") was decided (rule-derived labels + a human-judged golden set). `rules.py`,
`features.py` (text embedding via `bge-m3`, plus a real image embedder, `Dinov2Embedder`),
`classifier.py`, `resolve.py`, a FastAPI service, and a Material UI console are all written and
have been exercised end to end, including real image uploads through the live UI. See
ARCHITECTURE.md §4 and §8 for what is still a deliberate simplification (fixed tiers, a single
candidate model per tier) rather than a gap — a model-catalog redesign is under discussion next.

## Layout

```
llm-tier-router/
├── ARCHITECTURE.md        the design: scope, components, data flow, what is deliberately NOT built
├── PLAN.md                the task-by-task build plan, including the labeling decision
├── config/
│   └── tiers.example.yaml tier -> model name per provider, company-agnostic (copy to tiers.yaml)
├── src/gradist/        the library (skeleton; see PLAN.md Task 1 onward)
├── experiments/           training and evaluation scripts (not yet written)
└── tests/                 (not yet written)
```

## Using this from another project (including BARC)

Nothing in `src/gradist/` may import or assume anything about a specific consumer (no BARC
channel codes, no video-specific types in the core package). A consumer like a BARC service:

1. Installs this as a local/editable package (`pip install -e path/to/llm-tier-router`) once it
   has code, or vendors just `src/gradist/` — both work, it has no service dependencies.
2. Supplies its own `tiers.yaml` (its model names) and, if it wants metadata features, its own
   list of known metadata values (e.g. its anchor names, its channel codes) — the library never
   hardcodes anyone's values; unknown values fall into a shared "unknown" bucket (see
   ARCHITECTURE.md, "Feature vector").
3. Calls `resolve(request)` and gets back a model name to call however it already calls models.
