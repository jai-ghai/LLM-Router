# Architecture: llm-tier-router

Date: 2026-11-16. Status: final for this scope (see README.md for how this scope was decided).

## 1. Problem and scope

**Problem.** Different requests need different amounts of model capability. Sending every
request to the same model either overpays for easy requests or underpays (and loses quality) on
hard ones. A classifier that looks at a request's metadata and text should be able to predict
which tier of model it needs, before any model is called.

**In scope.**
- A feature vector built from request metadata (one-hot, company-agnostic), request text (an
  embedding), and an optional thumbnail (a real image embedding — `Dinov2Embedder` — not a stub).
- A classifier that maps that feature vector to one of **3 tiers**.
- A `resolve()` function that turns a tier into a concrete model name from a config file, with
  optional rule overrides (for example "channel X always gets the large tier", set by whoever
  deploys this, in their own config).
- A way to evaluate the classifier against a fixed-model baseline and an oracle, the same idea
  validated in the earlier R&D project's research (papers: FrugalGPT, RouteLLM, RouterDC,
  MMR-Bench — a learned choice beats a fixed model, the oracle sets the ceiling).

**Explicitly out of scope for this phase** (the team's 2026-11-16 decision):
- **Calling any model.** `resolve()` returns a model name and stops. The caller calls it however
  it already does.
- **LangChain integration.** Stubbed as an empty factory interface only (`tierrouter.llm_stub`),
  so a later phase has a place to put it without changing the classifier's contract. Not wired
  to anything, not tested against a real provider, no dependency added yet.
- **Output verification, escalation, cost tracking, tracing.** All of that lived in the earlier
  `R&D/llmrouter` project, which is left untouched. If this classifier proves out, that project's
  adapters and pipeline are the natural next layer — but that choice is deferred.

A FastAPI service and a Material UI console *were* built (see README.md) so one operator can
manage rules/examples/training without touching YAML or code directly — this was added once the
classifier and feature pipeline were proven, not part of the original "resolve-only" scope above,
which still holds for the library itself (`src/tierrouter/`).

## 2. Why not LiteLLM alone, why not LangChain yet

LiteLLM (and LangChain) give one API across providers — useful once something needs to *call* a
model. Neither decides *which* model to call; that decision is exactly what a team would still
have to build, by hand, per use case. This project is that decision, built once, generally. Once
it is validated, a thin LangChain (or LiteLLM) layer underneath is a small, separate addition —
hence it is stubbed, not built, so the classifier's contract does not end up shaped by a calling
library before the classifier itself is proven.

## 3. Components

```
Request (metadata dict, text, optional thumbnail)
   │
   ├─ 1. Feature builder  : metadata one-hot (+ unknown bucket) · text embedding · thumbnail slot
   ├─ 2. Rule overrides    : optional, same "most specific wins" design as the earlier project's
   │                         rules.py — ported in trimmed form, config-driven, no BARC-specific
   │                         values in the library itself
   ├─ 3. Tier classifier   : feature vector -> one of 3 tiers, with a confidence score
   └─ 4. Tier -> model map : tiers.yaml, per provider, supplied by whoever deploys this
        │
        result: (tier, model_name, confidence, reason)
```

A rule, when it matches, skips the classifier entirely for that request (same precedence as the
earlier project: rules are an operator's explicit override, and always win). Without a matching
rule, the classifier decides.

## 4. Feature vector

Three kinds of input, each independently optional so the same classifier works for a text-only
company use case and for BARC's metadata-and-video use case:

| Input | How it is encoded | Notes |
|---|---|---|
| Metadata (anchor, channel/video type, or anything else the caller names) | One-hot per field, plus one **"unknown" column per field** | The library ships no company's value list. A consumer passes its own list of known values per field (for example BARC's anchor names) when it builds a feature builder. A value not in that list goes to "unknown" — it must not crash or silently get ignored. |
| Text | An embedding | Reuses the earlier project's embedder choice and reasoning (`bge-m3`, multilingual, MIT; `all-MiniLM-L6-v2` as a lighter English-only comparison) — that decision (ADR-0012 in the old project) still holds; it is about the text itself, not about anything video-specific. |
| Thumbnail | An image embedding (`Dinov2Embedder`, `facebook/dinov2-small`, Apache-2.0 — license checked the same way as the text embedder) | Optional: a request with no thumbnail is encoded as a zero-vector plus a distinct "has-image" flag, so absence is never confused with a near-zero real embedding. `sentence-transformers/clip-ViT-B-32` was considered and rejected — no explicit license tag on its model card, same reasoning as rejecting Arch-Router in the old project's ADR-0006. |

## 5. Tiers are generic, not Gemini- or BARC-specific

The classifier predicts an abstract tier (`small` / `medium` / `large` are the working names,
changeable). `tiers.yaml` maps each tier to a real model name **per provider family**, for
example:

```yaml
tiers:
  - name: small
    models: {gemini: REPLACE_ME, anthropic: claude-haiku-5-5, bedrock: REPLACE_ME}
  - name: medium
    models: {gemini: REPLACE_ME, anthropic: claude-sonnet-5-5, bedrock: REPLACE_ME}
  - name: large
    models: {gemini: REPLACE_ME, anthropic: claude-opus-5-5, bedrock: REPLACE_ME}
```

This is what makes the same trained classifier usable by a team on Gemini and a team on Claude:
the classifier never sees a model name, only a tier. Retraining is needed only if the *task*
changes, not if the model lineup changes.

## 6. What carries over from the earlier project, and what does not

| From `R&D/llmrouter` | Carried over? | Why |
|---|---|---|
| `rules.py` (attribute rules, most-specific-wins, YAML store) | Yes, trimmed | Validated design, no video assumptions in it already |
| Embedding model choice and its license check (ADR-0012) | Yes | About text, not video |
| `types.py` request shape, `policy.py`, `verify.py`, `pipeline.py`, adapters, `service/`, `ui/` | No | All of it assumes "call and verify a model", which is explicitly out of scope now |
| `docs/adr/0002` ("router decision separate from provider adapters") | Superseded for this phase | There are no adapters in this phase to be separate from; the principle (classifier never calls a model) still holds and is restated in §1 |

## 7. Labeling decision (resolved)

The classifier needs labeled training data: for a given request, which tier was actually
necessary. Decided per PLAN.md, "Labeling strategy: options": combine strategy 1 (rule-derived
seed labels) with strategy 2 (a small human-judged golden set covering what no rule reaches).
Strategies 3/4 (oracle-by-running, LLM-as-judge) remain a later cross-check; strategy 6
(production feedback loop) remains a long-term goal once real traffic exists.

## 8. Evaluation, before trusting the classifier

Same discipline as the earlier project's Phase 0: compare the classifier against (a) always the
medium tier, (b) an oracle (the cheapest tier that was actually sufficient, from the labels), and
report per-field accuracy (does it do worse on an "unknown" bucket value?) and the tier
distribution it produces. If it does not clearly beat "always medium" on held-out data, it is not
ready to decide anything.
