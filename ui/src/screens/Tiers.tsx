import { ArrowRight, Download } from '@phosphor-icons/react'
import { Alert, Box, Button, MenuItem, Stack, TextField, Typography } from '@mui/material'
import { InfoBanner, Mono, PageHead, Panel, StateTag, StepFlow } from '../components/Primitives'
import { PALETTE, tierToneByPosition } from '../theme'
import type { TierSpec, TiersResponse } from '../types'

/** Read-only here: editing tiers.yaml is not built yet — see README.md. The layout matches
 * the Figma reference, where each tier becomes a self-contained card with per-provider
 * model mappings. The yellow banner at the top appears only when some mapping is still
 * REPLACE_ME, which is the exact state that blocks a real deployment. */
export function Tiers({ tiers }: { tiers: TiersResponse | null }) {
  if (!tiers) return <Alert severity="info">Loading…</Alert>
  const providers = Array.from(new Set(tiers.tiers.flatMap((t) => Object.keys(t.models))))
  const tierNames = tiers.tiers.map((t) => t.name)
  const unmapped = tiers.tiers.reduce((acc, t) => acc + Object.values(t.models).filter((m) => m === 'REPLACE_ME' || !m).length, 0)
  const totalMappings = tiers.tiers.reduce((acc, t) => acc + Object.keys(t.models).length, 0)

  return (
    <>
      <PageHead
        title="Model tiers"
        lede="Map each tier to the right model for every provider."
        actions={
          <Button variant="contained" startIcon={<Download size={14} weight="bold" />}>
            Export configuration
          </Button>
        }
      />

      <Stack spacing={2}>
        {unmapped > 0 && (
          <InfoBanner
            tone="warning"
            badge="!"
            title="Finish your model mappings"
            note={`The example configuration uses REPLACE_ME. Assign a real model name to each tier and provider before use.`}
            actions={
              <Typography sx={{ fontSize: 12, fontWeight: 600, color: PALETTE.warningText }}>
                {unmapped} of {totalMappings} mappings remaining
              </Typography>
            }
          />
        )}

        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: `repeat(${Math.min(tiers.tiers.length, 3)}, 1fr)` }, gap: 2 }}>
          {tiers.tiers.map((tier, i) => (
            <TierCard key={tier.name} tier={tier} index={i} providers={providers} tierNames={tierNames} />
          ))}
        </Box>

        <Panel dense>
          <Stack
            direction={{ xs: 'column', md: 'row' }}
            sx={{ justifyContent: 'space-between', alignItems: { xs: 'flex-start', md: 'center' }, px: 2.5, py: 2, gap: 1.5 }}
          >
            <Box>
              <Typography sx={{ fontSize: 14, fontWeight: 650, color: PALETTE.text }}>Default fallback tier</Typography>
              <Typography sx={{ fontSize: 12.5, color: PALETTE.textMuted }}>
                Used only when neither a rule nor the classifier produces a decision.
              </Typography>
            </Box>
            <TextField select size="small" value={tiers.low_confidence_fallback} disabled sx={{ minWidth: 200 }}>
              {tiers.tiers.map((t) => (
                <MenuItem key={t.name} value={t.name}>{t.name}</MenuItem>
              ))}
            </TextField>
          </Stack>
        </Panel>

        <Panel title="One decision. One configuration lookup." note="The classifier returns a tier, not a model. The resolver uses your provider-specific mappings to select model_name.">
          <StepFlow
            steps={[
              { label: 'Input', note: 'metadata + text' },
              { label: 'Decision', note: `tier: ${tiers.tiers[0]?.name ?? '—'}` },
              { label: 'Lookup', note: `provider: ${tiers.default_provider}` },
              { label: 'Output', note: 'model_name' },
            ]}
          />
          <Typography sx={{ fontSize: 11.5, color: PALETTE.textFaint, mt: 1.5 }}>
            Configuration preview · Example provider labels · No model is invoked
          </Typography>
        </Panel>
      </Stack>
    </>
  )
}

function TierCard({
  tier,
  index,
  providers,
  tierNames,
}: {
  tier: TierSpec
  index: number
  providers: string[]
  tierNames: string[]
}) {
  const tone = tierToneByPosition(tier.name, tierNames)
  const unmapped = Object.values(tier.models).filter((m) => m === 'REPLACE_ME' || !m).length
  const total = Object.keys(tier.models).length
  return (
    <Box
      sx={{
        borderRadius: '12px',
        border: `1px solid ${PALETTE.hairline}`,
        bgcolor: PALETTE.surface,
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <Box sx={{ p: 2.5, pb: 1.5 }}>
        <StateTag label={`TIER ${String(index + 1).padStart(2, '0')}`} tone={tone} size="small" />
        <Typography sx={{ fontSize: 22, fontWeight: 700, letterSpacing: '-0.02em', color: PALETTE.text, mt: 1, textTransform: 'capitalize' }}>
          {tier.name}
        </Typography>
        <Typography sx={{ fontSize: 13, color: PALETTE.textMuted, mt: 0.5, minHeight: 36 }}>
          {tier.description}
        </Typography>
      </Box>

      <Box sx={{ px: 2.5, pb: 2 }}>
        <Typography sx={{ fontSize: 11, fontWeight: 650, letterSpacing: '0.05em', color: PALETTE.textMuted, textTransform: 'uppercase', mb: 1 }}>
          Model per provider
        </Typography>
        <Stack spacing={1.25}>
          {providers.map((provider) => (
            <Box key={provider}>
              <Typography sx={{ fontSize: 12, color: PALETTE.textMuted, mb: 0.4 }}>{provider}</Typography>
              <TextField
                select
                size="small"
                fullWidth
                value={tier.models[provider] ?? ''}
                disabled
                slotProps={{ input: { sx: { fontSize: 13 } } }}
              >
                <MenuItem value={tier.models[provider] ?? ''}>
                  {tier.models[provider] === 'REPLACE_ME' ? (
                    <Mono size={12.5} muted>REPLACE_ME</Mono>
                  ) : (
                    <Mono size={12.5}>{tier.models[provider] ?? '—'}</Mono>
                  )}
                </MenuItem>
              </TextField>
            </Box>
          ))}
        </Stack>
      </Box>

      <Box
        sx={{
          px: 2.5,
          py: 1.25,
          borderTop: `1px solid ${PALETTE.hairline}`,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <Typography sx={{ fontSize: 11.5, color: unmapped === 0 ? PALETTE.goodText : PALETTE.textMuted, fontWeight: 600 }}>
          {unmapped === 0 ? 'Configured' : `Not configured (${unmapped}/${total})`}
        </Typography>
        <Button size="small" endIcon={<ArrowRight size={12} />} disabled>
          Edit tier
        </Button>
      </Box>
    </Box>
  )
}
