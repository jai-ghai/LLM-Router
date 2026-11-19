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
2. **Classifier** — if trained and confidence ≥ `min_confidence` → predicted tier.
3. **Fallback** — otherwise the configured low-confidence fallback.
4. **Catalog pick** — within the tier, pick by `optimize_for`: `cost` | `accuracy` | `balanced`.  
   Models marked `UNREVIEWED` are never selected.

## Run locally

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
