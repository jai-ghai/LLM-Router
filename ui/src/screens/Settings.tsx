// Callers: App.tsx when active === 'settings'. No prior Settings screen.
// Data: GET /v1/status (+ auth me). No secrets. User asked for Settings if useful.
import { Alert, Box, Stack, Typography } from '@mui/material'
import { useEffect, useState } from 'react'
import { api, describeError } from '../api'
import { useAuth } from '../auth'
import { PageHead, Panel } from '../components/Primitives'
import { PALETTE } from '../theme'
import type { Status, TiersResponse } from '../types'

function Row({ label, value }: { label: string; value: string }) {
  return (
    <Stack direction="row" sx={{ justifyContent: 'space-between', gap: 2, py: 0.75 }}>
      <Typography sx={{ fontSize: 13, color: PALETTE.textMuted }}>{label}</Typography>
      <Typography sx={{ fontSize: 13, fontWeight: 600, color: PALETTE.text, textAlign: 'right' }}>
        {value}
      </Typography>
    </Stack>
  )
}

/**
 * Read-only deployment facts from the live API. Env vars (session secret, paths) stay
 * out of the UI on purpose — change those in the process environment and restart.
 */
export function Settings() {
  const { state } = useAuth()
  const user = state.status === 'authenticated' ? state.user : null
  const [status, setStatus] = useState<Status | null>(null)
  const [tiers, setTiers] = useState<TiersResponse | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    void (async () => {
      try {
        const [s, t] = await Promise.all([api.status(), api.tiers()])
        setStatus(s)
        setTiers(t)
      } catch (err) {
        setError(describeError(err))
      }
    })()
  }, [])

  return (
    <Box>
      <PageHead
        title="Settings"
        lede="What this Gradist deployment is running with. Secrets and file paths are set via environment variables, not here."
      />
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}
      <Stack direction={{ xs: 'column', md: 'row' }} spacing={2.5} sx={{ alignItems: 'stretch' }}>
        <Box sx={{ flex: 1, minWidth: 0 }}>
          <Panel title="Deployment">
            {status === null && !error ? (
              <Typography sx={{ fontSize: 13, color: PALETTE.textMuted }}>Loading…</Typography>
            ) : status ? (
              <Box>
                <Row
                  label="Model catalog"
                  value={status.catalog_configured ? 'configured' : 'not set (tiers.yaml only)'}
                />
                <Row
                  label="Models awaiting review"
                  value={String(status.unreviewed_count ?? 0)}
                />
                <Row
                  label="Quality field"
                  value={status.quality_field ? status.quality_field : 'not set (accuracy mode limited)'}
                />
                <Row
                  label="Classifier"
                  value={
                    status.trained
                      ? `trained · ${status.n_labels_last_trained_on ?? 0} labels${status.uses_images ? ' · images on' : ''}`
                      : 'not trained'
                  }
                />
                <Row
                  label="Quality head (plain text)"
                  value={status.quality_head ? 'loaded · xRouteBench outcomes' : 'not loaded — text uses the classifier'}
                />
                <Row
                  label="QC-risk escalation"
                  value={status.qc_risk ? 'on · riskiest ~20% of sub-stories → large' : 'off'}
                />
                <Row
                  label="Unauthenticated reads"
                  value={status.allow_read_without_auth ? 'allowed' : 'denied'}
                />
                {tiers && (
                  <>
                    <Row label="Tiers" value={tiers.tiers.map((t) => t.name).join(' · ')} />
                    <Row label="Low-confidence fallback" value={tiers.low_confidence_fallback} />
                    <Row
                      label="Providers with tier defaults"
                      value={String(new Set(tiers.tiers.flatMap((t) => Object.keys(t.models))).size)}
                    />
                  </>
                )}
              </Box>
            ) : null}
          </Panel>
        </Box>
        <Box sx={{ width: { xs: '100%', md: 320 }, flexShrink: 0 }}>
          <Panel title="Signed-in operator">
            <Row label="Username" value={user?.username ?? '—'} />
            <Row label="Role" value={user?.role ?? '—'} />
          </Panel>
        </Box>
      </Stack>
    </Box>
  )
}
