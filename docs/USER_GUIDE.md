# mRouter user guide

mRouter picks a **model tier** and a concrete **model name** for a request. It does **not** call
the model — your application does that with the `model_name` returned by `/v1/resolve`. The code
package and settings keep their original name (`gradist`, `GRADIST_*`).

The in-app **Documentation** page (user menu) carries the same guide. Screenshots live in
`docs/images/` and `ui/public/docs/images/`.

![Overview](images/overview.png)

## Where it is useful

- Workloads with stable metadata (`task`, channel, content type, …) that should map to tiers.
- Cost-first routing by default, accuracy-first where it matters.
- Video and image work: the frame, its on-screen text and the channel all inform the tier.
- Audit: every decision is stored with its evidence and can be rated.

If you only use one model, or need a streaming proxy with provider failover, mRouter is the
decision step in front of that — not the proxy itself.

## How a request is resolved

1. **Rules** — metadata match → tier (+ optional per-rule `optimize_for`). Rules always win.
2. **Classifier** — requests with an image or domain metadata: gives P(needs at least each tier);
   `optimize_for` takes the highest tier whose chance clears its threshold (cost 70%, balanced
   50%, accuracy 30%). Metadata fields it was not trained on are listed in
   `signals.metadata_ignored`.
3. **Quality head** — plain-text requests: predicts each tier's score and takes the cheapest tier
   within the `optimize_for` margin (accuracy 0.03, balanced 0.06, cost 0.10).
4. **Fallback** — otherwise, or below `min_confidence`, the configured fallback tier.
5. **QC-risk escalation** — requests with text and `channel_name`: the riskiest ~20% of
   sub-stories are raised to `large`. Never lowers a tier, never overrides a rule.
6. **Catalog pick** — within the tier, by `optimize_for`: `cost` → cheapest; `accuracy` → highest
   benchmark capability; `balanced` → cheapest at or above the tier's median capability. Human
   quality ratings replace capability where present. Capability is the Artificial Analysis
   intelligence index from OpenRouter where measured, otherwise estimated from price and release
   date (shown with `~`). Models marked `UNREVIEWED` are never selected.

`provider` accepts one or several comma-separated providers (`anthropic,google`). The scope can
be the whole catalog, a hand-picked list (`selected_models`) or a named task pool (`task`).

### What the router reads

- **Metadata** — each trained field; unseen values fall into an "unknown" bucket.
- **Text** — a sentence embedding (MiniLM).
- **Image / video frame** — a 384-number DINOv2 fingerprint of the frame's look. A video is never
  uploaded: the browser sends one still plus `modality=video`, which limits the pick to models
  that accept video.
- **On-screen text** — with `GRADIST_OCR=1` (and the `ocr` extra installed) the frame is read with
  EasyOCR in Hindi + English (`GRADIST_OCR_LANGS` for others); lines are returned in
  `signals.screen_text`, cached in `data/ocr_cache.json`, and used in training
  (`reads_screen_text` in `classifier.meta.json`). The first read of a frame takes ~8 s on CPU.
- **Recorder file names** — `1010013_2026.07.28_07.40.00-….mp4` fills `channel_code` and the
  channel's name, network, type and genre (from `GET /v1/channels`) in the Playground, without
  overwriting anything you typed.

The decision returns `tier_scores`, the top-8 `candidates`, `pool_size` and `signals`. When the
classifier decides, `signals.classifier_inputs` lists exactly what it read and the tier odds
**without** the image, text, screen text or metadata — the Playground shows this as "What moved
the decision".

![Playground with a video decision](images/playground.png)

## Run locally

1. Install: `pip install -e ".[service,embeddings,images,ocr]"` in a virtualenv, and `npm install`
   in `ui/`.
2. Create `.env` (gitignored) — see the table below. `ui/.env.local` must point `VITE_API_BASE`
   at `http://localhost:8000`.
3. Create the first user: `python -m gradist.service.bootstrap --username <name>` (prompts for a
   password; `--role member` for non-admins, `--reset <name>` to reset a password).
4. Run `scripts\start.ps1`: API on port 8000, console on 5174, each in its own window.

Trained models (`classifier`, `quality_head`, `qc_risk`) are reloaded from `data/` on every start;
`POST /v1/train` is only needed after examples or judgments change.

| Variable | Required | Purpose |
|---|---|---|
| `GRADIST_SESSION_SECRET` | yes | Signs session cookies — `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
| `GRADIST_TIERS` | yes | Tier table, e.g. `config/tiers.general.yaml` |
| `GRADIST_CATALOG` | no | `config/catalog.yaml` (353 models, 17 providers) |
| `GRADIST_DATA` | no | Store directory, default `data` |
| `GRADIST_QUALITY_FIELD` | no | Field that buckets quality ratings, e.g. `task` |
| `GRADIST_CORS_ORIGINS` | no | Console origin, default `http://localhost:5174` |
| `GRADIST_RULES_SEED` | no | Rules file copied in while the store has none |
| `GRADIST_ALLOW_READ_WITHOUT_AUTH` | no | `1` opens GET endpoints for monitoring |
| `GRADIST_OCR`, `GRADIST_OCR_LANGS` | no | `1` turns on screen-text reading; languages default `hi,en` |

## Integrate from another app

```bash
curl -c cookies.txt -X POST http://localhost:8000/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"svc","password":"..."}'

curl -b cookies.txt -X POST http://localhost:8000/v1/resolve \
  -F 'metadata={"channel_code":"1010013","modality":"video"}' \
  -F 'text=summarise the bulletin' \
  -F 'image=@frame.jpg' \
  -F 'optimize_for=balanced'
```

```python
import httpx

with httpx.Client(base_url="http://localhost:8000", timeout=60.0) as client:
    client.post("/v1/auth/login", json={"username": "svc", "password": "..."})
    decision = client.post(
        "/v1/resolve",
        data={"metadata": '{"task":"news"}', "text": "...", "optimize_for": "balanced"},
    ).json()
    model = decision["model_name"]   # call your provider with this
```

mRouter holds no provider API keys. The full API is in [design.md](../design.md#api) and at
<http://localhost:8000/docs>.

## Console map

| Where | Screen | Purpose |
|---|---|---|
| Sidebar | Overview | Volume, tier mix and how decisions were made (your own requests; admins see all) |
| Sidebar | Routing rules | Metadata → tier overrides; always win |
| Sidebar | Request logs | Your decisions for audit; rate model output. Admins see every operator's, with a "Requested by" filter |
| Sidebar | Playground | Try a resolve with metadata, text, image or video |
| User menu | Documentation | This guide |
| User menu | Settings | Live deployment facts (no secrets) |
| User menu | Operators | Admins: who may sign in |

The training screens (Examples, Human judging, Train, Evaluate) are hidden from navigation; their
endpoints remain for scripted retraining.

![Request logs](images/request-logs.png)

## Retraining and catalog upkeep

| Script (`experiments/`) | Does |
|---|---|
| `train_from_tabson_db.py`, `train_from_video_clips.py`, `train_from_hf_content.py` | Build labelled examples and fit the classifier |
| `eval_ocr.py` | Held-out accuracy with vs. without OCR |
| `train_quality_head.py` (after `probe_xroutebench.py`) | Fit the plain-text quality head |
| `qc_outcome_dataset.py` → `train_qc_risk.py` | Fit the QC-risk escalation head |
| `sync_catalog.py --add-new` | Import new OpenRouter models as `UNREVIEWED` |
| `refresh_catalog.py`, `enrich_capability.py` | Refresh prices/modalities and capability scores |

## Known limitations

- All labelled video frames are `large`, so video currently always routes to `large`; more varied
  frame labels are needed before the image can choose a smaller tier.
- One process = one data directory, catalog and rule set; run separate deployments for separate
  products.
