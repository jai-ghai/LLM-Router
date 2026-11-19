// Callers: App.tsx when active === 'docs'. In-app user guide (also mirrored in docs/USER_GUIDE.md).
// No prior Docs screen. User: write docs with details, integration, where useful, screenshots.
import { Box, Stack, Typography } from '@mui/material'
import type { ReactNode } from 'react'
import { PageHead, Panel } from '../components/Primitives'
import { PALETTE } from '../theme'

function H({ children }: { children: string }) {
  return (
    <Typography sx={{ fontSize: 15, fontWeight: 700, color: PALETTE.text, mt: 2.5, mb: 1 }}>
      {children}
    </Typography>
  )
}

function P({ children }: { children: ReactNode }) {
  return (
    <Typography sx={{ fontSize: 13.5, color: PALETTE.textMuted, lineHeight: 1.55, mb: 1 }}>
      {children}
    </Typography>
  )
}

function Code({ children }: { children: string }) {
  return (
    <Box
      component="pre"
      sx={{
        m: 0,
        mb: 1.5,
        p: 1.5,
        borderRadius: '8px',
        bgcolor: PALETTE.raised,
        border: `1px solid ${PALETTE.hairline}`,
        fontFamily: 'JetBrains Mono, ui-monospace, monospace',
        fontSize: 12,
        color: PALETTE.text,
        overflowX: 'auto',
        whiteSpace: 'pre-wrap',
      }}
    >
      {children}
    </Box>
  )
}

function Shot({ src, caption }: { src: string; caption: string }) {
  return (
    <Box sx={{ my: 1.5 }}>
      <Box
        component="img"
        src={src}
        alt={caption}
        onError={(e) => {
          ;(e.currentTarget as HTMLImageElement).style.display = 'none'
        }}
        sx={{
          display: 'block',
          width: '100%',
          maxWidth: 880,
          borderRadius: '8px',
          border: `1px solid ${PALETTE.hairline}`,
          bgcolor: PALETTE.surface,
        }}
      />
      <Typography sx={{ fontSize: 12, color: PALETTE.textFaint, mt: 0.75 }}>{caption}</Typography>
    </Box>
  )
}

/**
 * Operator-facing guide. Source of truth for GitHub readers is also docs/USER_GUIDE.md;
 * keep the two aligned when you edit either.
 */
export function Docs() {
  return (
    <Box>
      <PageHead
        title="Documentation"
        lede="How Gradist picks a model tier, how to run it, and how to call it from another app."
      />

      <Panel title="What Gradist is">
        <P>
          Gradist chooses which <em>tier</em> and which <em>model</em> should handle a request from
          its metadata (and optional text or image). It does <strong>not</strong> run the model.
          Your own application calls the provider after Gradist returns a decision.
        </P>
        <P>
          Routing rules, quality ratings, and cost vs accuracy preferences stay in Gradist. Provider
          credentials and inference stay in your application.
        </P>
        <Shot src="/docs/images/overview.png" caption="Overview — request volume and how decisions were made (rules → classifier → fallback)." />
      </Panel>

      <Panel title="Where it is useful">
        <P>Use Gradist when you run more than one model tier and need a consistent, auditable pick:</P>
        <Stack component="ul" sx={{ pl: 2.5, m: 0, mb: 1, '& li': { fontSize: 13.5, color: PALETTE.textMuted, mb: 0.5 } }}>
          <li>Workloads that already carry structured metadata (task, channel, content type, …)</li>
          <li>Defaulting to lower-cost models and raising the bar only when quality ratings support it</li>
          <li>Letting operators rate outcomes and feeding those ratings back into selection</li>
          <li>Reviewing newly imported catalog models before they can be routed to</li>
        </Stack>
        <P>
          If you only ever use a single model, or you need a full request proxy with streaming and
          provider failover, Gradist alone is not that layer — pair it with whatever you already use
          to call models.
        </P>
      </Panel>

      <Panel title="How a request is resolved">
        <P>Order is fixed:</P>
        <Code>{`1. Rules — if metadata matches a rule → that tier (and optional optimize_for)
2. Classifier — if trained and confidence ≥ min_confidence → predicted tier
3. Fallback — low-confidence / untrained → configured fallback model
4. Catalog pick — within the tier, pick a concrete model by optimize_for:
     cost | accuracy | balanced
   UNREVIEWED models are never selected.`}</Code>
        <Shot src="/docs/images/playground.png" caption="Playground — try a resolve with metadata, optional image, and optimize_for." />
        <Shot src="/docs/images/request-logs.png" caption="Request logs — every resolve is stored for audit and rating." />
      </Panel>

      <Panel title="Operators, catalog, quality">
        <P>
          Admins manage operators (who may sign in). Members can resolve, rate, and judge; only admins
          assign catalog tiers and manage users.
        </P>
        <Shot src="/docs/images/operators.png" caption="Operators — who can sign in, assign catalog tiers, and who is attributed on ratings." />
        <Shot src="/docs/images/catalog.png" caption="Catalog — new models stay under review until an admin assigns a real tier." />
        <P>
          Quality ratings (1–5) key off a configured metadata field (for example task). Accuracy and
          balanced modes use those averages; without ratings, selection falls back to cost.
        </P>
      </Panel>

      <Panel title="Integrate from another application">
        <H>Contract</H>
        <P>
          Call POST /v1/resolve as multipart/form-data (same shape the Playground uses). You must
          send a session cookie from a prior POST /v1/auth/login, or run a trusted service account
          login once and reuse the cookie jar.
        </P>
        <Code>{`curl -c cookies.txt -X POST http://localhost:8000/v1/auth/login \\
  -H "Content-Type: application/json" \\
  -d '{"username":"svc","password":"..."}'

curl -b cookies.txt -X POST http://localhost:8000/v1/resolve \\
  -F 'metadata={"task":"summarise"}' \\
  -F 'text=example request text' \\
  -F 'optimize_for=cost'

# Response (Decision):
# {
#   "tier": "small",
#   "model_name": "<selected-model>",
#   "confidence": null,
#   "reason": "...",
#   "rule_outcome": "...",
#   "classifier_outcome": null,
#   "fallback_outcome": null,
#   "optimize_for": "cost"
# }`}</Code>
        <P>Then call your model provider with <code>model_name</code>. Gradist does not store provider API keys.</P>
        <H>Python sketch</H>
        <Code>{`import httpx

with httpx.Client(base_url="http://localhost:8000", timeout=30.0) as client:
    client.post("/v1/auth/login", json={"username": "svc", "password": "..."})
    r = client.post(
        "/v1/resolve",
        data={
            "metadata": '{"task":"news"}',
            "text": "summarise the bulletin",
            "optimize_for": "balanced",
        },
    )
    r.raise_for_status()
    decision = r.json()
    model = decision["model_name"]
    # call your provider with model`}</Code>
        <H>Syncing the catalog</H>
        <Code>{`python experiments/sync_catalog.py \\
  --out config/catalog.yaml \\
  --providers openai,anthropic,google \\
  --add-new          # new rows land as UNREVIEWED
# or --only-existing  # refresh cost/modality facts only`}</Code>
      </Panel>
    </Box>
  )
}
