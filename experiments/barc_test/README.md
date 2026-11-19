# barc_test: one worked test, not part of the general library

This folder is DATA for one test run against `src/gradist`, built from reading Minfy's BARC
production code (`BARC Workspace/tabson-python-inferencing`, read-only, on 2026-11-16). Nothing
in `src/gradist/*.py` knows this folder exists or contains anything BARC-specific — see
ARCHITECTURE.md §6 and README.md's "Using this from another project" section. A different
consumer supplies its own `tiers.yaml`, `rules.yaml` and examples; this is what BARC's would
plausibly look like, for testing only.

## What is real here, and what is synthetic

- **`tiers.yaml`'s three model names are real**: read from `gemini/client.py` and
  `agentic_qc/core.py`'s `os.getenv(..., "default")` calls — these are the literal defaults the
  inferencing service falls back to. Not verified against a live Gemini API; not verified that
  these specific ids are still valid today.
- **`rules.yaml` encodes a real, verified finding, not an assumption**: production does **not**
  choose a bigger or smaller model by content type (news vs. sports). It chooses by **what the
  call is for** — a cheap text check, the main video analysis, or the QC review pass — set by
  separate env vars (`MODEL_ID`, `QC_MODEL_ID`, `HEADLINE_SIMILARITY_MODEL_ID`, ...) with no
  per-channel or per-genre override anywhere in this codebase. So the rules below are keyed on
  a `task` attribute, not `channel_type`.
- **`examples.jsonl` is synthetic.** This code repository holds no actual customer broadcast
  content, no real headlines, no real substory text — it is code, not a data export, so none
  was available to take. The metadata vocabulary (`genre`) is real (`core/taxonomy.py`'s
  `STORY_GENRES`); the `channel_flow` values (`news`/`sports`/`playout`) are the three real flow
  names in `pipelines/` and `news/`; the text in every row is written for this test, not quoted
  from anything. Do not read this file as real BARC data.
- **No thumbnail.** `grep -rn -i "thumbnail\|extract_frame\|keyframe"` across the whole
  inferencing repo found nothing — production does not extract or use a thumbnail anywhere
  today, so there was nothing to base `ThumbnailSlot` on, and it stays unimplemented.

## Run it

```
cd llm-tier-router
./.venv/Scripts/python experiments/train_classifier.py \
    --tiers experiments/barc_test/tiers.yaml \
    --rules experiments/barc_test/rules.yaml \
    --examples experiments/barc_test/examples.jsonl
```
