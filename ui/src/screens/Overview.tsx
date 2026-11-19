import { Play } from '@phosphor-icons/react'
import { Box, Button, Skeleton, Stack, Table, TableBody, TableCell, TableHead, TableRow, Typography } from '@mui/material'
import { BarChart } from '@mui/x-charts/BarChart'
import { useEffect, useState } from 'react'
import { api, describeError } from '../api'
import { Mono, MetricStrip, Panel, PageHead, StateTag, StepFlow, type Metric, type FlowStep } from '../components/Primitives'
import { PALETTE, TONE_FILL, tierTone, tierToneByPosition } from '../theme'
import type { HistoryEntry, TiersResponse } from '../types'
import type { PageKey } from '../Nav'

// Three green shades used for the stacked request-volume bar chart — same ordered
// family as the tier-pill tones, so a viewer who glances between the chart and the
// table does not have to re-learn the colour language on each view.
const TIER_CHART_COLORS = [PALETTE.tier1Fill, PALETTE.accentSoft, PALETTE.accent]
const TIER_CHART_EDGES = [PALETTE.tier1Text, PALETTE.good, PALETTE.brand]

export function Overview({
  tiers,
  onlineStatus,
  onNavigate,
}: {
  tiers: TiersResponse | null
  onlineStatus: 'checking' | 'online' | 'offline'
  onNavigate: (page: PageKey) => void
}) {
  const [history, setHistory] = useState<HistoryEntry[] | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    void (async () => {
      try {
        setHistory((await api.history(500)).history)
      } catch (err) {
        setError(describeError(err))
        setHistory([])
      }
    })()
  }, [])

  const loading = history === null
  const rows = history ?? []
  const total = rows.length
  const ruleCount = rows.filter((h) => h.rule_outcome !== 'no matching rule').length
  const classifierCount = rows.filter((h) => h.rule_outcome === 'no matching rule' && h.fallback_outcome === 'not needed').length
  const fallbackCount = total - ruleCount - classifierCount

  // Mean classifier confidence (ignoring the rule-match rows whose confidence is 1.0 by
  // definition — a rule hit says nothing about whether the classifier is calibrated).
  const classifierRows = rows.filter((r) => r.rule_outcome === 'no matching rule' && typeof r.confidence === 'number')
  const meanConfidence = classifierRows.length === 0
    ? null
    : classifierRows.reduce((s, r) => s + (r.confidence ?? 0), 0) / classifierRows.length

  const tierNames = tiers?.tiers.map((t) => t.name) ?? []

  const metrics: Metric[] = [
    {
      label: 'Total requests',
      value: total.toLocaleString(),
      tone: 'brand',
      note: total === 0 ? 'No resolved requests yet.' : 'Resolved via /v1/resolve this session.',
    },
    {
      label: 'Rule-matched',
      value: `${total === 0 ? '—' : ((ruleCount / total) * 100).toFixed(0)}%`,
      note: `${ruleCount.toLocaleString()} of ${total.toLocaleString()} requests`,
    },
    {
      label: 'Classifier-decided',
      value: `${total === 0 ? '—' : ((classifierCount / total) * 100).toFixed(0)}%`,
      note: `${classifierCount.toLocaleString()} of ${total.toLocaleString()} requests`,
    },
    {
      label: 'Mean confidence',
      value: meanConfidence === null ? '—' : `${(meanConfidence * 100).toFixed(1)}%`,
      note: meanConfidence === null ? 'No classifier decisions yet.' : 'Classifier-decided rows only.',
    },
  ]

  // Per-tier histogram across up to 12 time buckets, so the "Request volume" chart has
  // enough bars to feel like a timeseries without requiring a real backend aggregation.
  const chart = buildVolumeChart(rows, tierNames)

  const tierBreakdown = tierNames.map((name) => {
    const count = rows.filter((r) => r.tier === name).length
    const pct = total === 0 ? 0 : (count / total) * 100
    return { name, count, pct, tone: tierToneByPosition(name, tierNames) }
  })

  const resolutionSteps: FlowStep[] = [
    {
      label: 'Rules',
      note: `Match metadata & conditions · ${total > 0 ? `${((ruleCount / total) * 100).toFixed(1)}% of requests` : 'no data'}`,
      active: ruleCount > 0 && ruleCount >= classifierCount && ruleCount >= fallbackCount,
    },
    {
      label: 'Classifier',
      note: `Predict tier from text + metadata · ${total > 0 ? `${((classifierCount / total) * 100).toFixed(1)}% of requests` : 'no data'}`,
      active: classifierCount > 0 && classifierCount > ruleCount && classifierCount >= fallbackCount,
    },
    {
      label: 'Fallback',
      note: `Use the configured default tier · ${total > 0 ? `${((fallbackCount / total) * 100).toFixed(1)}% of requests` : 'no data'}`,
      active: fallbackCount > 0 && fallbackCount > ruleCount && fallbackCount > classifierCount,
    },
  ]

  return (
    <>
      <PageHead
        title="Routing overview"
        lede="The right model for every request. Visibility into every decision."
        actions={
          <Button
            variant="contained"
            startIcon={<Play size={14} weight="fill" />}
            onClick={() => onNavigate('resolve')}
          >
            Open playground
          </Button>
        }
      />

      <Stack spacing={2}>
        <MetricStrip metrics={metrics} />

        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1.6fr 1fr' }, gap: 2, alignItems: 'stretch' }}>
          <Panel
            title="Request volume"
            note={error || `Requests routed across all tiers · ${onlineStatus === 'online' ? 'Live traffic' : 'Service offline'}`}
          >
            {loading ? (
              <Skeleton variant="rectangular" height={240} sx={{ borderRadius: '10px' }} />
            ) : total === 0 ? (
              <EmptyChartHint />
            ) : (
              <>
                <BarChart
                  series={chart.series}
                  xAxis={[{ data: chart.labels, scaleType: 'band' }]}
                  height={240}
                  hideLegend
                  borderRadius={5}
                  margin={{ left: 36, right: 12, top: 12, bottom: 28 }}
                  colors={TIER_CHART_EDGES}
                />
                <Stack direction="row" spacing={2} sx={{ mt: 1, alignItems: 'center', flexWrap: 'wrap' }}>
                  {tierNames.map((name, i) => (
                    <Stack key={name} direction="row" spacing={0.75} sx={{ alignItems: 'center' }}>
                      <Box sx={{ width: 10, height: 10, borderRadius: '3px', bgcolor: TIER_CHART_EDGES[i] ?? PALETTE.textFaint }} />
                      <Typography sx={{ fontSize: 12, color: PALETTE.textMuted, textTransform: 'capitalize' }}>{name}</Typography>
                    </Stack>
                  ))}
                </Stack>
              </>
            )}
          </Panel>

          <Panel title="Tier distribution" note="Efficient by default. Powerful when needed.">
            {loading ? (
              <Skeleton variant="rectangular" height={200} sx={{ borderRadius: '10px' }} />
            ) : total === 0 ? (
              <EmptyChartHint />
            ) : (
              <Stack spacing={1.5}>
                <Stack direction="row" sx={{ height: 10, borderRadius: '999px', overflow: 'hidden', bgcolor: PALETTE.raised }}>
                  {tierBreakdown.map((t) => (
                    <Box
                      key={t.name}
                      sx={{
                        width: `${t.pct}%`,
                        bgcolor: TONE_FILL[t.tone] === PALETTE.hairlineStrong ? PALETTE.textFaint : TONE_FILL[t.tone],
                        transition: 'width 220ms ease',
                      }}
                      aria-label={`${t.name}: ${t.pct.toFixed(0)}%`}
                    />
                  ))}
                </Stack>
                <Stack spacing={1}>
                  {tierBreakdown.map((t) => (
                    <Stack key={t.name} direction="row" spacing={1.5} sx={{ alignItems: 'center' }}>
                      <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: TONE_FILL[t.tone] === PALETTE.hairlineStrong ? PALETTE.textFaint : TONE_FILL[t.tone], flex: '0 0 auto' }} />
                      <Typography sx={{ fontSize: 13, fontWeight: 600, color: PALETTE.text, textTransform: 'capitalize', minWidth: 80 }}>{t.name}</Typography>
                      <Typography sx={{ fontSize: 11.5, color: PALETTE.textMuted, flex: 1 }}>
                        {t.count.toLocaleString()} {t.count === 1 ? 'request' : 'requests'}
                      </Typography>
                      <Typography sx={{ fontSize: 13, fontWeight: 650, color: PALETTE.text, fontVariantNumeric: 'tabular-nums' }}>
                        {t.pct.toFixed(0)}%
                      </Typography>
                    </Stack>
                  ))}
                </Stack>
              </Stack>
            )}
          </Panel>
        </Box>

        <Panel title="How requests are resolved" note="First match wins.">
          <StepFlow steps={resolutionSteps} />
        </Panel>

        <Panel
          title="Recent decisions"
          note="This session only — there is no persistent analytics store."
          actions={
            <Button size="small" onClick={() => onNavigate('history')}>
              View request logs →
            </Button>
          }
          dense
        >
          {rows.length === 0 ? (
            <Typography sx={{ fontSize: 13, color: PALETTE.textMuted, p: 2.5 }}>
              Nothing resolved yet — use Playground to send a request.
            </Typography>
          ) : (
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Request / time</TableCell>
                  <TableCell>Resolution</TableCell>
                  <TableCell>Tier</TableCell>
                  <TableCell>Model</TableCell>
                  <TableCell align="right">Confidence</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {rows.slice(0, 6).map((entry, i) => (
                  <TableRow key={`${entry.timestamp}-${i}`}>
                    <TableCell>
                      <Typography sx={{ fontSize: 12.5, color: PALETTE.text, fontWeight: 500 }}>
                        {entry.text ? entry.text.slice(0, 32) + (entry.text.length > 32 ? '…' : '') : '— no text —'}
                      </Typography>
                      <Typography sx={{ fontSize: 11, color: PALETTE.textMuted }}>
                        {new Date(entry.timestamp).toLocaleTimeString()}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography sx={{ fontSize: 12.5, color: PALETTE.textMuted }}>
                        {entry.rule_outcome !== 'no matching rule' ? 'Rule match' : entry.fallback_outcome === 'not needed' ? 'Classifier' : 'Fallback'}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <StateTag label={entry.tier} tone={tierTone(entry.tier)} size="small" />
                    </TableCell>
                    <TableCell>
                      <Mono size={12}>{entry.model_name}</Mono>
                    </TableCell>
                    <TableCell align="right">
                      <Typography sx={{ fontSize: 12, color: PALETTE.textMuted, fontVariantNumeric: 'tabular-nums' }}>
                        {entry.confidence !== null ? `${(entry.confidence * 100).toFixed(1)}%` : '—'}
                      </Typography>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </Panel>
      </Stack>
    </>
  )
}

function EmptyChartHint() {
  return (
    <Stack sx={{ height: 200, alignItems: 'center', justifyContent: 'center' }}>
      <Typography sx={{ fontSize: 12.5, color: PALETTE.textFaint }}>
        Resolve a request in Playground to see this chart fill in.
      </Typography>
    </Stack>
  )
}

/** Build a stacked-bar timeseries from the raw history rows. Buckets by hour when there
 * are enough rows to make hourly bars informative; otherwise falls back to a per-row index
 * so a thin history still shows something. */
function buildVolumeChart(rows: HistoryEntry[], tierNames: string[]) {
  if (rows.length === 0) return { labels: [] as string[], series: [] as { data: number[]; label: string; stack: string; color: string }[] }
  const BUCKETS = Math.min(12, Math.max(4, rows.length))
  const buckets: number[][] = Array.from({ length: BUCKETS }, () => tierNames.map(() => 0))
  const sorted = [...rows].sort((a, b) => a.timestamp.localeCompare(b.timestamp))
  const first = new Date(sorted[0].timestamp).getTime()
  const last = new Date(sorted[sorted.length - 1].timestamp).getTime()
  const span = Math.max(1, last - first)
  for (const row of sorted) {
    const t = new Date(row.timestamp).getTime()
    const bucket = Math.min(BUCKETS - 1, Math.floor(((t - first) / span) * BUCKETS))
    const tierIdx = tierNames.indexOf(row.tier)
    if (tierIdx >= 0) buckets[bucket][tierIdx] += 1
  }
  const labels = buckets.map((_, i) => {
    const t = first + (i / BUCKETS) * span
    return new Date(t).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  })
  const series = tierNames.map((name, tierIdx) => ({
    data: buckets.map((b) => b[tierIdx]),
    label: name,
    stack: 'all',
    color: TIER_CHART_COLORS[tierIdx] ?? PALETTE.textFaint,
  }))
  return { labels, series }
}
