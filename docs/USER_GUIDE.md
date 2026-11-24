# Gradist user guide

Gradist picks a **model tier** and a concrete **model name** from request metadata. It does
**not** call the model — your application does that after `/v1/resolve`.

Screenshots used by the in-app Documentation page: `docs/images/` (also under `ui/public/docs/images/`).

## Where it is useful

- Workloads with stable metadata (`task`, `channel`, content type, …) that should map to tiers.
- Cost-first routing by default, with accuracy-first when human ratings exist.
- Reviewing newly imported catalog models before they can be selected.
- Audit: every resolve is stored with rule / classifier / fallback outcomes and can be rated.

If you only use one model, or you need a full streaming proxy with provider failover, Gradist is
the decision step — not that proxy.

## How resolve works

1. **Rules** — metadata match → tier (+ optional per-rule `optimize_for`).
2. **Classifier** — requests with an image or domain metadata: if trained and confidence ≥
   `min_confidence`, the classifier gives P(needs at least each tier) and `optimize_for` takes the
   highest tier whose chance clears its threshold (cost 70%, balanced 50%, accuracy 30%).
   Metadata fields the classifier was not trained on are ignored (listed in `signals.metadata_ignored`).
3. **Quality head** — plain-text requests (no domain metadata) when `data/quality_head.joblib`
   exists: predicts each tier's score and takes the cheapest tier within the `optimize_for` margin
   (accuracy 0.03, balanced 0.06, cost 0.10).
4. **Fallback** — otherwise the configured low-confidence fallback.
5. **QC-risk escalation** — when `data/qc_risk.joblib` exists and the request carries text and
   `channel_name`: the riskiest ~20% of sub-stories (by predicted QC fix) are raised to `large`.
   Never lowers a tier, never overrides a rule.
6. **Catalog pick** — within the tier, pick by `optimize_for`: `cost` takes the cheapest;
   `accuracy` the highest benchmark capability; `balanced` the cheapest model at or above the
   tier's median capability. Human quality ratings, when present, replace capability.
   Capability is the Artificial Analysis intelligence index from OpenRouter where measured, else
   estimated from price and release date (shown with `~`). Refresh it with
   `python experiments/enrich_capability.py`. Models marked `UNREVIEWED` are never selected.

`provider` accepts one or several comma-separated providers (`provider=anthropic,google`). The
decision returns `tier_scores` (the evidence for the tier), `candidates` (the top 8 of the tier),
`pool_size`, and `signals` (what the router received: text size, metadata used and ignored, image
bytes, modality, providers). When the classifier decides, `signals.classifier_inputs` lists what it
read (each metadata value and whether it was seen in training, the text and image embedders and
their sizes) and the tier odds with the image, text or metadata removed — so you can see which
input the decision rests on. The image part is a DINOv2 visual embedding of the frame; it does not
read on-screen text. A video is never uploaded: the browser sends one still frame plus
`modality=video`, which limits the pick to models that accept video.

## Run locally

Put the variables below in `.env` (gitignored) and run `scripts\start.ps1`: it starts the API on
8000 and the UI on 5174 in two windows. `ui/.env.local` must point `VITE_API_BASE` at port 8000.
A trained classifier, quality head and QC-risk head are reloaded from `data/` on every start, so
`POST /v1/train` is only needed after examples or judgments change.

Or by hand:

```powershell
# API
$env:GRADIST_SESSION_SECRET = "<random>"
$env:GRADIST_TIERS = "config/tiers.example.yaml"
$env:GRADIST_CORS_ORIGINS = "http://localhost:5174"
# optional: GRADIST_CATALOG, GRADIST_QUALITY_FIELD=task
python -m uvicorn gradist.service.main:create --factory --port 8000

python -m gradist.service.bootstrap --username <name>

# UI
cd ui
npm run dev
```

## Integrate from another app

```bash
curl -c cookies.txt -X POST http://localhost:8000/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"svc","password":"..."}'

curl -b cookies.txt -X POST http://localhost:8000/v1/resolve \
  -F 'metadata={"task":"news"}' \
  -F 'text=summarise' \
  -F 'optimize_for=cost'
```

Use `model_name` from the JSON decision with your provider. Gradist does not hold provider API keys.

```python
import httpx

with httpx.Client(base_url="http://localhost:8000", timeout=30.0) as client:
    client.post("/v1/auth/login", json={"username": "svc", "password": "..."})
    decision = client.post(
        "/v1/resolve",
        data={"metadata": '{"task":"news"}', "text": "...", "optimize_for": "balanced"},
    ).json()
    model = decision["model_name"]
```

Catalog sync:

```bash
python experiments/sync_catalog.py \
  --out config/catalog.yaml \
  --providers openai,anthropic,google \
  --add-new
```

## Console map

| Screen | Purpose |
|--------|---------|
| Overview | Volume / tier mix from history |
| Routing rules | Editable rules |
| Model tiers | Tier table |
| Model catalog | Models; admin assign/demote UNREVIEWED |
| Request logs | Resolve audit + quality ratings |
| Playground | Manual `/v1/resolve` |
| Examples / Judging / Train / Evaluate | Classifier loop |
| Operators | User management |
| Settings | Live deployment facts |
| Documentation | This guide in-app |

## Multi-workspace

Not built. One Gradist process uses one data directory, one catalog, and one rule set. For hard
isolation between products, run separate deployments until a real tenant model exists.
