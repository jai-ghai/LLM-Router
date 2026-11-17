import { MagnifyingGlass, Plus, ArrowRight } from '@phosphor-icons/react'
import {
  Alert,
  Box,
  Button,
  InputAdornment,
  MenuItem,
  Skeleton,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from '@mui/material'
import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { api, describeError } from '../api'
import { MetricStrip, Mono, PageHead, Panel, StateTag, type Metric } from '../components/Primitives'
import { PALETTE, tierTone } from '../theme'
import type { CatalogEntry, QualityResponse } from '../types'

/**
 * The selectable model catalog: every candidate model this deployment knows about. The
 * table on the left is the reference list, filtered by search / provider / tier; the
 * right panel is the full details of whichever one is selected, matching the Figma
 * master-detail layout. Checking a row is kept from the original design because Playground
 * reads from this selection — it was more of a developer tool than a visual element, so it
 * is now folded into the details panel.
 */
export function Catalog({
  selected,
  onChange,
}: {
  selected: Set<string> | null
  onChange: (next: Set<string> | null) => void
}) {
  const [entries, setEntries] = useState<CatalogEntry[] | null>(null)
  const [error, setError] = useState('')
  const [quality, setQuality] = useState<QualityResponse | null>(null)
  const [activeEntry, setActiveEntry] = useState<CatalogEntry | null>(null)
  const [search, setSearch] = useState('')
  const [providerFilter, setProviderFilter] = useState<string>('')
  const [tierFilter, setTierFilter] = useState<string>('')

  useEffect(() => {
    void (async () => {
      try {
        const [catalogRes, qualityRes] = await Promise.all([api.catalog(), api.quality()])
        setEntries(catalogRes.catalog)
        setActiveEntry((prev) => prev ?? catalogRes.catalog[0] ?? null)
        setQuality(qualityRes)
      } catch (err) {
        setError(describeError(err))
        setEntries([])
      }
    })()
  }, [])

  const allIds = (entries ?? []).map((e) => e.id)
  const usingAll = selected === null
  const activeIds = usingAll ? new Set(allIds) : new Set(selected ?? [])

  function toggle(id: string) {
    const base = new Set(selected ?? allIds)
    if (base.has(id)) base.delete(id)
    else base.add(id)
    onChange(base.size === allIds.length ? null : base)
  }

  const providers = useMemo(() => Array.from(new Set((entries ?? []).map((e) => e.provider))), [entries])
  const tiers = useMemo(() => Array.from(new Set((entries ?? []).map((e) => e.tier))), [entries])

  const filtered = useMemo(() => {
    const term = search.trim().toLowerCase()
    return (entries ?? []).filter((e) => {
      if (providerFilter && e.provider !== providerFilter) return false
      if (tierFilter && e.tier !== tierFilter) return false
      if (term && !(e.id.toLowerCase().includes(term) || e.label.toLowerCase().includes(term) || e.provider.toLowerCase().includes(term))) return false
      return true
    })
  }, [entries, search, providerFilter, tierFilter])

  const metrics: Metric[] = [
    { label: 'Catalog entries', value: entries?.length ?? '—', tone: 'brand', note: 'Reference entries' },
    { label: 'Providers', value: providers.length || '—', note: providers.join(' · ') || 'No providers configured' },
    {
      label: 'Active selections',
      value: entries === null ? '—' : `${activeIds.size} / ${entries.length}`,
      note: usingAll ? 'All models selectable in Playground' : 'Subset selectable in Playground',
    },
  ]

  const activeScores = activeEntry && quality?.field ? quality.scores?.filter((s) => s.model_id === activeEntry.id) ?? [] : []

  return (
    <>
      <PageHead
        title="Model catalog"
        lede="A reference catalog for model names and tier mappings."
        actions={
          <Button variant="contained" startIcon={<Plus size={14} weight="bold" />} disabled>
            Add model
          </Button>
        }
      />
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Stack spacing={2}>
        <MetricStrip metrics={metrics} />

        {entries === null ? (
          <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1.4fr 1fr' }, gap: 2 }}>
            <Panel title="Available model references" dense>
              <Stack spacing={1} sx={{ p: 2.5 }}>
                {[0, 1, 2, 3].map((i) => (
                  <Skeleton key={i} variant="rectangular" height={32} sx={{ borderRadius: '8px' }} />
                ))}
              </Stack>
            </Panel>
            <Panel title="Model reference">
              <Skeleton variant="text" height={28} width="60%" />
              <Skeleton variant="text" height={20} />
              <Skeleton variant="text" height={20} />
            </Panel>
          </Box>
        ) : entries.length === 0 && !error ? (
          <Alert severity="info">
            No catalog configured for this deployment (TIERROUTER_CATALOG is unset) — Playground uses tiers.yaml's single
            model per tier instead.
          </Alert>
        ) : (
          <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1.4fr 1fr' }, gap: 2, alignItems: 'start' }}>
            <Panel
              title="Available model references"
              dense
              actions={
                <Stack direction="row" spacing={1}>
                  <TextField
                    size="small"
                    placeholder="Search model name or provider…"
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    sx={{ minWidth: 220 }}
                    slotProps={{ input: { startAdornment: <InputAdornment position="start"><MagnifyingGlass size={14} /></InputAdornment> } }}
                  />
                  <TextField
                    select
                    size="small"
                    value={providerFilter}
                    onChange={(e) => setProviderFilter(e.target.value)}
                    sx={{ minWidth: 130 }}
                  >
                    <MenuItem value="">All providers</MenuItem>
                    {providers.map((p) => (
                      <MenuItem key={p} value={p}>{p}</MenuItem>
                    ))}
                  </TextField>
                  <TextField
                    select
                    size="small"
                    value={tierFilter}
                    onChange={(e) => setTierFilter(e.target.value)}
                    sx={{ minWidth: 110 }}
                  >
                    <MenuItem value="">All tiers</MenuItem>
                    {tiers.map((t) => (
                      <MenuItem key={t} value={t}>{t}</MenuItem>
                    ))}
                  </TextField>
                </Stack>
              }
            >
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Model name</TableCell>
                    <TableCell>Provider</TableCell>
                    <TableCell>Suggested tier</TableCell>
                    <TableCell align="right">Cost</TableCell>
                    <TableCell align="center">Mapping</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {filtered.length === 0 && (
                    <TableRow>
                      <TableCell colSpan={5}>
                        <Typography sx={{ fontSize: 13, color: PALETTE.textMuted, p: 1 }}>No models match this filter.</Typography>
                      </TableCell>
                    </TableRow>
                  )}
                  {filtered.map((entry) => (
                    <TableRow
                      key={entry.id}
                      hover
                      onClick={() => setActiveEntry(entry)}
                      sx={{ cursor: 'pointer', bgcolor: activeEntry?.id === entry.id ? PALETTE.brandSoft : undefined }}
                    >
                      <TableCell sx={{ fontSize: 13, fontWeight: 600 }}>
                        <Mono size={13}>{entry.label}</Mono>
                      </TableCell>
                      <TableCell>{entry.provider}</TableCell>
                      <TableCell>
                        <StateTag label={entry.tier} tone={tierTone(entry.tier)} size="small" />
                      </TableCell>
                      <TableCell align="right">
                        <Mono>{entry.cost}</Mono>
                      </TableCell>
                      <TableCell align="center">
                        <Typography sx={{ fontSize: 11.5, color: activeIds.has(entry.id) ? PALETTE.goodText : PALETTE.textMuted, fontWeight: 600 }}>
                          {activeIds.has(entry.id) ? 'Selectable' : 'Unassigned'}
                        </Typography>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
              <Box sx={{ px: 2.5, py: 1.25, borderTop: `1px solid ${PALETTE.hairline}`, display: 'flex', justifyContent: 'space-between' }}>
                <Typography sx={{ fontSize: 11.5, color: PALETTE.textMuted }}>
                  Showing {filtered.length} of {entries.length} reference entries.
                </Typography>
                <Button size="small" variant="outlined" onClick={() => onChange(null)} disabled={usingAll}>
                  Use all models
                </Button>
              </Box>
            </Panel>

            <Panel
              title="Model reference"
              actions={activeEntry && <StateTag label={activeEntry.provider} tone="brand" size="small" />}
            >
              {!activeEntry ? (
                <Typography sx={{ fontSize: 13, color: PALETTE.textMuted }}>Select a model to see its details.</Typography>
              ) : (
                <Stack spacing={1.5}>
                  <Box>
                    <Typography sx={{ fontSize: 22, fontWeight: 700, color: PALETTE.text, letterSpacing: '-0.02em' }}>
                      <Mono size={22}>{activeEntry.label}</Mono>
                    </Typography>
                    <Typography sx={{ fontSize: 12.5, color: PALETTE.textMuted, mt: 0.25 }}>
                      Suggested for the <Box component="span" sx={{ fontWeight: 650, textTransform: 'capitalize' }}>{activeEntry.tier}</Box> tier.
                    </Typography>
                  </Box>

                  <Stack spacing={1}>
                    <DetailRow label="Model identifier" value={<Mono>{activeEntry.id}</Mono>} />
                    <DetailRow label="Provider key" value={activeEntry.provider} />
                    <DetailRow label="Suggested tier" value={<StateTag label={activeEntry.tier} tone={tierTone(activeEntry.tier)} size="small" />} />
                    <DetailRow label="Relative cost" value={<Mono>{activeEntry.cost}</Mono>} />
                    <DetailRow label="Accepts an image" value={activeEntry.supports_image ? 'Yes' : 'No'} />
                    <DetailRow label="Current mapping" value={activeIds.has(activeEntry.id) ? 'Selectable' : 'Not assigned'} />
                  </Stack>

                  <Box sx={{ p: 1.25, borderRadius: '10px', bgcolor: PALETTE.warningSoft, border: `1px solid ${PALETTE.warningEdge}` }}>
                    <Typography sx={{ fontSize: 12, fontWeight: 650, color: PALETTE.warningText }}>Reference only</Typography>
                    <Typography sx={{ fontSize: 11.5, color: PALETTE.warningText, opacity: 0.85 }}>
                      Adding a catalog entry does not configure a tier or call a provider.
                    </Typography>
                  </Box>

                  <Button
                    variant="contained"
                    endIcon={<ArrowRight size={14} />}
                    onClick={() => toggle(activeEntry.id)}
                    fullWidth
                  >
                    {activeIds.has(activeEntry.id) ? 'Remove from selection' : 'Configure tier mapping'}
                  </Button>

                  {quality?.field && (
                    <Box>
                      <Typography sx={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.04em', textTransform: 'uppercase', color: PALETTE.textMuted, mb: 0.5 }}>
                        Human quality score (by {quality.field})
                      </Typography>
                      {activeScores.length === 0 ? (
                        <Typography sx={{ fontSize: 12, color: PALETTE.textFaint }}>No ratings yet — rate real output on Request logs.</Typography>
                      ) : (
                        <Stack spacing={0.5}>
                          {activeScores.map((s) => (
                            <Stack key={s.value} direction="row" sx={{ justifyContent: 'space-between' }}>
                              <Mono size={12}>{s.value}</Mono>
                              <Typography sx={{ fontSize: 12.5, fontWeight: 600 }}>{s.average.toFixed(1)} / 5</Typography>
                            </Stack>
                          ))}
                        </Stack>
                      )}
                    </Box>
                  )}
                </Stack>
              )}
            </Panel>
          </Box>
        )}

        <Panel title="Your tier mappings remain the source of truth." note="Use the catalog to organize model identifiers. The router can resolve a tier through its configuration without a catalog." dense>
          <Stack direction={{ xs: 'column', md: 'row' }} sx={{ p: 2.25, gap: 1.5, justifyContent: 'space-between' }}>
            <Typography sx={{ fontSize: 12.5, color: PALETTE.textMuted }}>
              Catalog = reference metadata
            </Typography>
            <Typography sx={{ fontSize: 12.5, fontWeight: 600, color: PALETTE.brandText }}>
              Tier configuration = provider-specific model selection
            </Typography>
          </Stack>
        </Panel>
      </Stack>
    </>
  )
}

function DetailRow({ label, value }: { label: string; value: ReactNode }) {
  return (
    <Stack direction="row" sx={{ justifyContent: 'space-between', alignItems: 'center', py: 0.4, borderBottom: `1px dashed ${PALETTE.hairline}` }}>
      <Typography sx={{ fontSize: 12, color: PALETTE.textMuted }}>{label}</Typography>
      <Box sx={{ fontSize: 13, color: PALETTE.text }}>{value}</Box>
    </Stack>
  )
}
