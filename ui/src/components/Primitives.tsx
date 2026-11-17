import { ArrowDown, ArrowRight, ArrowUp } from '@phosphor-icons/react'
import { Box, Card, Stack, Tooltip, Typography } from '@mui/material'
import { useEffect, useState, type ReactNode } from 'react'
import { MONO_STACK, PALETTE, TONE_COLOR, TONE_EDGE, TONE_SOFT, type Tone } from '../theme'

/** True once, cheaply — this never changes during a session. */
const prefersReducedMotion =
  typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

/* ---------------------------------------------------------------- motion ---- */

/**
 * Fade-and-rise mount animation, staggered by index. Entrance only (per the
 * web-animation-design guidance: ease-out, under 300ms, and every animated
 * element gets a reduced-motion escape hatch).
 */
export function Reveal({ children, index = 0 }: { children: ReactNode; index?: number }) {
  const [shown, setShown] = useState(prefersReducedMotion)

  useEffect(() => {
    if (prefersReducedMotion) return
    const id = window.setTimeout(() => setShown(true), 20 + Math.min(index, 8) * 35)
    return () => window.clearTimeout(id)
  }, [index])

  return (
    <Box
      sx={{
        opacity: shown ? 1 : 0,
        transform: shown ? 'translateY(0)' : 'translateY(6px)',
        transition: 'opacity 240ms cubic-bezier(0.165,0.84,0.44,1), transform 240ms cubic-bezier(0.165,0.84,0.44,1)',
      }}
    >
      {children}
    </Box>
  )
}

/** Fades a screen in on route/tab change, keyed by the caller. Exit is implicit (unmount). */
export function PageTransition({ routeKey, children }: { routeKey: string; children: ReactNode }) {
  const [shown, setShown] = useState(prefersReducedMotion)
  useEffect(() => {
    setShown(prefersReducedMotion)
    if (prefersReducedMotion) return
    const id = window.setTimeout(() => setShown(true), 10)
    return () => window.clearTimeout(id)
  }, [routeKey])
  return (
    <Box
      key={routeKey}
      sx={{
        opacity: shown ? 1 : 0,
        transform: shown ? 'translateY(0)' : 'translateY(4px)',
        transition: 'opacity 200ms ease-out, transform 200ms ease-out',
      }}
    >
      {children}
    </Box>
  )
}

/* ------------------------------------------------------------- page head ---- */

export function PageHead({
  title,
  lede,
  actions,
}: {
  title: string
  lede?: string
  actions?: ReactNode
}) {
  return (
    <Stack
      direction="row"
      sx={{ alignItems: 'flex-start', justifyContent: 'space-between', gap: 2.5, mb: 2.5, flexWrap: 'wrap' }}
    >
      <Box sx={{ minWidth: 0, flex: '1 1 380px' }}>
        <Typography
          sx={{ fontSize: 26, fontWeight: 700, letterSpacing: '-0.025em', color: PALETTE.text, m: 0, lineHeight: 1.15 }}
        >
          {title}
        </Typography>
        {lede && (
          <Typography sx={{ fontSize: 13.5, lineHeight: 1.55, color: PALETTE.textMuted, mt: 0.5, maxWidth: '72ch' }}>
            {lede}
          </Typography>
        )}
      </Box>
      {actions && (
        <Stack direction="row" spacing={1} sx={{ alignItems: 'center', flex: '0 0 auto' }}>
          {actions}
        </Stack>
      )}
    </Stack>
  )
}

/* ------------------------------------------------------------------ panel --- */

/** A titled plane. Groups a table, a form, or a result without decorating it. */
export function Panel({
  title,
  note,
  actions,
  children,
  dense = false,
}: {
  title?: ReactNode
  note?: ReactNode
  actions?: ReactNode
  children: ReactNode
  dense?: boolean
}) {
  return (
    <Card variant="outlined" sx={{ overflow: 'hidden', borderRadius: '12px' }}>
      {title && (
        <Stack
          direction="row"
          sx={{
            alignItems: 'flex-start',
            justifyContent: 'space-between',
            gap: 1.5,
            px: 2.5,
            pt: 2,
            pb: 1.5,
            borderBottom: `1px solid ${PALETTE.hairline}`,
          }}
        >
          <Box sx={{ minWidth: 0 }}>
            <Typography sx={{ fontSize: 15, fontWeight: 650, letterSpacing: '-0.01em', color: PALETTE.text }}>
              {title}
            </Typography>
            {note && (
              <Typography sx={{ fontSize: 12.5, color: PALETTE.textMuted, mt: 0.4, maxWidth: '78ch' }}>
                {note}
              </Typography>
            )}
          </Box>
          {actions && <Box sx={{ flex: '0 0 auto' }}>{actions}</Box>}
        </Stack>
      )}
      <Box sx={{ p: dense ? 0 : 2.5 }}>{children}</Box>
    </Card>
  )
}

/* -------------------------------------------------------------- state tag --- */

/**
 * The one way a status is rendered: a dot and a word, colour as reinforcement
 * rather than the only signal, so it doesn't rely on colour vision alone. Tier tones
 * (tier1/tier2/tier3) suppress the dot — the pill itself already carries the colour.
 */
export function StateTag({
  label,
  tone,
  size = 'default',
  title,
}: {
  label: string
  tone: Tone
  size?: 'default' | 'small'
  title?: string
}) {
  const small = size === 'small'
  const isTier = tone === 'tier1' || tone === 'tier2' || tone === 'tier3'
  const edge = TONE_EDGE[tone] ?? `${TONE_COLOR[tone]}33`
  const body = (
    <Box
      component="span"
      sx={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: small ? 0.5 : 0.7,
        px: small ? 0.9 : 1.1,
        py: small ? 0.2 : 0.3,
        borderRadius: '999px',
        bgcolor: TONE_SOFT[tone],
        color: TONE_COLOR[tone],
        border: '1px solid',
        borderColor: edge,
        fontSize: small ? 11 : 12,
        fontWeight: 600,
        whiteSpace: 'nowrap',
        lineHeight: 1.6,
        textTransform: isTier ? 'capitalize' : 'none',
      }}
    >
      {!isTier && (
        <Box
          component="span"
          sx={{
            width: small ? 5 : 6,
            height: small ? 5 : 6,
            borderRadius: '50%',
            bgcolor: tone === 'default' ? PALETTE.textFaint : TONE_COLOR[tone],
            flex: '0 0 auto',
          }}
        />
      )}
      {label}
    </Box>
  )
  return title ? <Tooltip title={title}>{body}</Tooltip> : body
}

/* ------------------------------------------------------------------- mono --- */

/** Monospaced value — model ids, example ids, confidences. Never used for prose. */
export function Mono({
  children,
  muted = false,
  size = 12.5,
}: {
  children: ReactNode
  muted?: boolean
  size?: number
}) {
  return (
    <Box
      component="span"
      sx={{ fontFamily: MONO_STACK, fontSize: size, color: muted ? PALETTE.textMuted : PALETTE.text }}
    >
      {children}
    </Box>
  )
}

/* --------------------------------------------------------------- section --- */

export function SectionHead({ title, note }: { title: string; note?: string }) {
  return (
    <Box sx={{ mb: 0.5 }}>
      <Typography sx={{ fontSize: 13.5, fontWeight: 650, letterSpacing: '-0.01em', color: PALETTE.text }}>
        {title}
      </Typography>
      {note && (
        <Typography sx={{ fontSize: 12, color: PALETTE.textMuted, mt: 0.3, maxWidth: '80ch' }}>{note}</Typography>
      )}
    </Box>
  )
}

/* ------------------------------------------------------------- info banner -- */

/**
 * The numbered-badge info banner used across pages (Rules "Rules always run first",
 * Playground "Explore a sample decision", Logs "Decisions, not model traces", Tiers
 * "Finish your model mappings"). One component so these callouts never drift apart
 * visually, and so the tone→background mapping goes through `TONE_SOFT` exactly once.
 */
export function InfoBanner({
  tone = 'brand',
  badge,
  title,
  note,
  actions,
}: {
  tone?: Tone
  /** Small corner label — "01", "PREVIEW", "!"… if omitted, a dot in the tone colour is used. */
  badge?: ReactNode
  title: ReactNode
  note?: ReactNode
  actions?: ReactNode
}) {
  return (
    <Stack
      direction={{ xs: 'column', sm: 'row' }}
      sx={{
        alignItems: { xs: 'flex-start', sm: 'center' },
        gap: 1.5,
        px: 2,
        py: 1.5,
        borderRadius: '10px',
        border: '1px solid',
        borderColor: TONE_EDGE[tone] ?? PALETTE.hairline,
        bgcolor: TONE_SOFT[tone],
        color: TONE_COLOR[tone],
      }}
    >
      <Box
        sx={{
          flex: '0 0 auto',
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          minWidth: 28,
          height: 22,
          px: 0.75,
          borderRadius: '6px',
          bgcolor: PALETTE.surface,
          color: TONE_COLOR[tone],
          border: `1px solid ${TONE_EDGE[tone] ?? PALETTE.hairline}`,
          fontFamily: MONO_STACK,
          fontSize: 11,
          fontWeight: 700,
          letterSpacing: '0.04em',
        }}
      >
        {badge ?? '•'}
      </Box>
      <Box sx={{ minWidth: 0, flex: 1 }}>
        <Typography sx={{ fontSize: 13.5, fontWeight: 650, color: TONE_COLOR[tone] }}>{title}</Typography>
        {note && (
          <Typography sx={{ fontSize: 12.5, color: TONE_COLOR[tone], opacity: 0.85, mt: 0.2 }}>{note}</Typography>
        )}
      </Box>
      {actions && <Box sx={{ flex: '0 0 auto' }}>{actions}</Box>}
    </Stack>
  )
}

/* ---------------------------------------------------------------- step flow -- */

export interface FlowStep {
  label: string
  note?: ReactNode
  /** When true, the step is drawn as "matched" — bolder weight, brand-tinted border. */
  active?: boolean
}

/**
 * The "01 → 02 → 03" numbered horizontal track used on Overview ("How requests are
 * resolved"), Rules bottom ("Predictable routing, by design"), Tiers bottom ("One
 * decision. One configuration lookup."). One primitive so these three never diverge.
 */
export function StepFlow({ steps, dense = false }: { steps: FlowStep[]; dense?: boolean }) {
  return (
    <Stack
      direction={{ xs: 'column', md: 'row' }}
      sx={{ gap: { xs: 1.25, md: 0 }, alignItems: { xs: 'stretch', md: 'center' } }}
      role="list"
    >
      {steps.map((step, i) => (
        <Stack key={`${step.label}-${i}`} direction="row" sx={{ alignItems: 'center', flex: 1 }}>
          <Box
            role="listitem"
            sx={{
              flex: 1,
              display: 'flex',
              alignItems: 'center',
              gap: 1.25,
              px: dense ? 1.25 : 1.75,
              py: dense ? 1 : 1.5,
              borderRadius: '10px',
              bgcolor: step.active ? PALETTE.brandSoft : PALETTE.subtle,
              border: `1px solid ${step.active ? PALETTE.brandEdge : PALETTE.hairline}`,
              transition: 'background-color 120ms ease, border-color 120ms ease',
            }}
          >
            <Box
              sx={{
                flex: '0 0 auto',
                width: 26,
                height: 26,
                borderRadius: '8px',
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                bgcolor: step.active ? PALETTE.brand : PALETTE.surface,
                color: step.active ? PALETTE.onBrand : PALETTE.textMuted,
                border: `1px solid ${step.active ? PALETTE.brand : PALETTE.hairline}`,
                fontFamily: MONO_STACK,
                fontSize: 11,
                fontWeight: 700,
              }}
            >
              {String(i + 1).padStart(2, '0')}
            </Box>
            <Box sx={{ minWidth: 0, flex: 1 }}>
              <Typography
                sx={{
                  fontSize: 13,
                  fontWeight: step.active ? 650 : 600,
                  color: step.active ? PALETTE.brandText : PALETTE.text,
                  lineHeight: 1.35,
                }}
              >
                {step.label}
              </Typography>
              {step.note && (
                <Typography
                  sx={{
                    fontSize: 11.5,
                    color: step.active ? PALETTE.brandText : PALETTE.textMuted,
                    opacity: step.active ? 0.9 : 1,
                    mt: 0.1,
                    lineHeight: 1.35,
                  }}
                >
                  {step.note}
                </Typography>
              )}
            </Box>
          </Box>
          {i < steps.length - 1 && (
            <Box
              aria-hidden
              sx={{
                display: { xs: 'none', md: 'flex' },
                alignItems: 'center',
                justifyContent: 'center',
                px: 0.75,
                color: PALETTE.textFaint,
              }}
            >
              <ArrowRight size={14} />
            </Box>
          )}
        </Stack>
      ))}
    </Stack>
  )
}

/* --------------------------------------------------------------- resolution -- */

/**
 * The precedence trace every `Decision` carries: rule → classifier → fallback, which step
 * actually decided, and what the others would have said. One component so "Try it" and
 * "Request logs" never disagree on how this is rendered.
 */
export function ResolutionPath({
  ruleOutcome,
  classifierOutcome,
  fallbackOutcome,
}: {
  ruleOutcome: string
  classifierOutcome: string | null
  fallbackOutcome: string | null
}) {
  const ruleMatched = ruleOutcome !== 'no matching rule'
  const classifierMatched = !ruleMatched && fallbackOutcome === 'not needed'
  const fallbackMatched = !ruleMatched && !classifierMatched

  const steps: { label: string; result: string; matched: boolean }[] = [
    { label: 'Rules', result: ruleOutcome, matched: ruleMatched },
    { label: 'Classifier', result: classifierOutcome ?? 'not reached', matched: classifierMatched },
    { label: 'Fallback', result: fallbackOutcome ?? 'not used', matched: fallbackMatched },
  ]

  return (
    <Stack spacing={0.25} role="list" aria-label="Resolution path">
      {steps.map((step, i) => (
        <Stack
          key={step.label}
          direction="row"
          spacing={1.25}
          role="listitem"
          sx={{
            alignItems: 'flex-start',
            py: 0.9,
            px: 1,
            borderRadius: '8px',
            bgcolor: step.matched ? PALETTE.brandSoft : 'transparent',
          }}
        >
          <Typography
            sx={{
              fontSize: 11,
              fontWeight: 700,
              fontFamily: MONO_STACK,
              color: step.matched ? PALETTE.brand : PALETTE.textFaint,
              minWidth: 22,
              mt: 0.2,
            }}
          >
            {String(i + 1).padStart(2, '0')}
          </Typography>
          <Stack sx={{ minWidth: 0, flex: 1 }}>
            <Typography sx={{ fontSize: 12.5, fontWeight: step.matched ? 650 : 500, color: step.matched ? PALETTE.brandText : PALETTE.textMuted }}>
              {step.label}
            </Typography>
            <Typography sx={{ fontSize: 12, color: step.matched ? PALETTE.brandText : PALETTE.textFaint, wordBreak: 'break-word' }}>
              {step.result}
            </Typography>
          </Stack>
        </Stack>
      ))}
    </Stack>
  )
}

/* ------------------------------------------------------------------ metric -- */

export interface Metric {
  label: string
  value: ReactNode
  note?: ReactNode
  tone?: Tone
  /** Trend delta, e.g. { direction: 'up', value: '12.8%', note: 'vs. previous 24 hours', good: true }. */
  trend?: {
    direction: 'up' | 'down'
    value: ReactNode
    note?: ReactNode
    /** Whether this direction is a good thing (an up-arrow on "cost saved" is good; on "latency" it is bad). */
    good?: boolean
  }
}

/** The KPI row. Clean card, big figure, trend arrow underneath — matches Figma's Overview tiles. */
export function MetricStrip({ metrics }: { metrics: Metric[] }) {
  return (
    <Box
      sx={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
        gap: 1.5,
      }}
      role="group"
      aria-label="Key figures"
    >
      {metrics.map((metric, i) => (
        <Reveal key={metric.label} index={i}>
          <MetricTile metric={metric} />
        </Reveal>
      ))}
    </Box>
  )
}

function MetricTile({ metric }: { metric: Metric }) {
  const tone = metric.tone ?? 'default'
  const trend = metric.trend
  const trendColor = trend
    ? trend.good === false
      ? PALETTE.trendDown
      : trend.good === true || trend.direction === 'up'
        ? PALETTE.trendUp
        : PALETTE.trendDown
    : undefined
  const TrendIcon = trend?.direction === 'down' ? ArrowDown : ArrowUp
  return (
    <Card
      variant="outlined"
      sx={{
        px: 2,
        py: 1.75,
        borderRadius: '12px',
        minHeight: 108,
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        gap: 0.4,
      }}
    >
      <Typography sx={{ fontSize: 11.5, fontWeight: 600, letterSpacing: '0.02em', color: PALETTE.textMuted }}>
        {metric.label}
      </Typography>
      <Typography
        sx={{
          fontSize: 28,
          fontWeight: 700,
          letterSpacing: '-0.025em',
          color: tone === 'brand' ? PALETTE.brand : PALETTE.text,
          fontVariantNumeric: 'tabular-nums',
          lineHeight: 1.1,
        }}
      >
        {metric.value}
      </Typography>
      {trend ? (
        <Stack direction="row" spacing={0.6} sx={{ alignItems: 'center' }}>
          <Box sx={{ display: 'inline-flex', alignItems: 'center', color: trendColor }}>
            <TrendIcon size={12} weight="bold" />
          </Box>
          <Typography sx={{ fontSize: 11.5, fontWeight: 650, color: trendColor }}>{trend.value}</Typography>
          {trend.note && (
            <Typography sx={{ fontSize: 11.5, color: PALETTE.textMuted }}>{trend.note}</Typography>
          )}
        </Stack>
      ) : (
        <Typography sx={{ fontSize: 11.5, color: PALETTE.textMuted, minHeight: 15 }}>
          {metric.note ?? ' '}
        </Typography>
      )}
    </Card>
  )
}
