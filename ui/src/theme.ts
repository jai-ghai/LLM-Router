import { createTheme, type Theme } from '@mui/material/styles'

/**
 * One theme, set once at the app root. No page or component overrides it.
 *
 * Figma-driven: dark teal-green brand (sidebar logo, headings emphasis), a brighter
 * CTA green for primary buttons, and three tier pill tones (mint / periwinkle / peach)
 * for Economy / Balanced / Premium — mapped by name so the same tones work whether a
 * deployment's tiers.yaml uses the original (small/medium/large) or the Figma (Economy/
 * Balanced/Premium) naming.
 */

export const PALETTE = {
  // Page ground vs. card plane, so a white card visibly sits above the canvas.
  canvas: '#F6F7F9',
  surface: '#FFFFFF',
  subtle: '#F7F8FA',
  raised: '#EEF1F3',
  chrome: '#FFFFFF',

  hairline: '#E4E7EB',
  hairlineStrong: '#C9CFD6',

  // Brand = dark forest green. Used for the logomark, the matched-step indicators in the
  // resolution trace, and headings-level emphasis. It is the *text* colour variant of the
  // palette: >12:1 contrast on white, safe at small sizes.
  brand: '#0E7A52',
  brandAlt: '#0A6341',
  brandText: '#0E7A52',
  brandSoft: '#E8F3EE',
  brandEdge: '#B8DBC6',
  onBrand: '#FFFFFF',

  // Accent = the brighter CTA green ("Open playground", "Save rule", "Resolve tier").
  // It is the *fill* colour variant — pairs with white text on top, not used for text.
  accent: '#16A866',
  accentAlt: '#138958',
  accentSoft: '#E3F6EC',
  accentEdge: '#B2E2C4',
  onAccent: '#FFFFFF',

  text: '#0F1A24',
  textSecondary: '#42515F',
  textMuted: '#67748A',
  textFaint: '#99A4B1',

  // Status. Universal semantics; the brand's green is close to "good" — so "good" here is
  // deliberately a slightly darker, more muted green than `accent` so success states don't
  // read as a second CTA.
  good: '#1F8B57',
  goodText: '#116A3F',
  goodSoft: '#E6F4EC',
  warning: '#CC8800',
  warningText: '#946300',
  warningSoft: '#FEF4D9',
  warningEdge: '#F4D9A2',
  critical: '#D14343',
  criticalText: '#A62E2E',
  criticalSoft: '#FDECEC',
  info: '#2563EB',
  infoText: '#1D4ED8',
  infoSoft: '#EAF0FE',

  // Tier pill tones — three distinct pastels, one per tier position (first / middle / last).
  // Matched by name in `tierTone()` so switching tier names between deployments doesn't
  // require a UI change.
  tier1Fill: '#D7F0DB',
  tier1Text: '#1F7A3A',
  tier1Edge: '#B7DEBE',
  tier2Fill: '#DADEF7',
  tier2Text: '#3B50C2',
  tier2Edge: '#BCC3EC',
  tier3Fill: '#F8D6C4',
  tier3Text: '#AF542A',
  tier3Edge: '#E6BBA3',

  // Up/down trend arrows on metric tiles (good = up, bad = down — not the same as
  // success/critical, because a lower latency is a *good* down).
  trendUp: '#1F8B57',
  trendDown: '#D14343',
} as const

export const FONT_STACK =
  '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif'

/** Strictly for quantitative/technical values — model ids, example ids, confidences. */
export const MONO_STACK =
  '"JetBrains Mono", ui-monospace, "Cascadia Mono", "SF Mono", Menlo, monospace'

export const TYPE = {
  metric: { fontSize: 30, fontWeight: 700, lineHeight: 1.05, letterSpacing: '-0.025em' },
  metricSm: { fontSize: 22, fontWeight: 700, lineHeight: 1.15, letterSpacing: '-0.02em' },
  widgetTitle: { fontSize: 15, fontWeight: 600, lineHeight: 1.35, letterSpacing: '-0.01em' },
  body: { fontSize: 13.5, fontWeight: 400, lineHeight: 1.55 },
  micro: { fontSize: 11, fontWeight: 650, lineHeight: 1.4, letterSpacing: '0.06em' },
} as const

export type Tone =
  | 'default'
  | 'good'
  | 'warning'
  | 'critical'
  | 'info'
  | 'brand'
  | 'accent'
  | 'tier1'
  | 'tier2'
  | 'tier3'

export const TONE_COLOR: Record<Tone, string> = {
  default: PALETTE.textMuted,
  good: PALETTE.goodText,
  warning: PALETTE.warningText,
  critical: PALETTE.criticalText,
  info: PALETTE.infoText,
  brand: PALETTE.brandText,
  accent: PALETTE.accentAlt,
  tier1: PALETTE.tier1Text,
  tier2: PALETTE.tier2Text,
  tier3: PALETTE.tier3Text,
}

export const TONE_FILL: Record<Tone, string> = {
  default: PALETTE.hairlineStrong,
  good: PALETTE.good,
  warning: PALETTE.warning,
  critical: PALETTE.critical,
  info: PALETTE.info,
  brand: PALETTE.brand,
  accent: PALETTE.accent,
  tier1: PALETTE.tier1Fill,
  tier2: PALETTE.tier2Fill,
  tier3: PALETTE.tier3Fill,
}

export const TONE_SOFT: Record<Tone, string> = {
  default: PALETTE.raised,
  good: PALETTE.goodSoft,
  warning: PALETTE.warningSoft,
  critical: PALETTE.criticalSoft,
  info: PALETTE.infoSoft,
  brand: PALETTE.brandSoft,
  accent: PALETTE.accentSoft,
  tier1: PALETTE.tier1Fill,
  tier2: PALETTE.tier2Fill,
  tier3: PALETTE.tier3Fill,
}

/** Edge colour for tier pills — hair-thin, same hue family. */
export const TONE_EDGE: Partial<Record<Tone, string>> = {
  tier1: PALETTE.tier1Edge,
  tier2: PALETTE.tier2Edge,
  tier3: PALETTE.tier3Edge,
  warning: PALETTE.warningEdge,
  accent: PALETTE.accentEdge,
  brand: PALETTE.brandEdge,
}

/**
 * The one place a tier name becomes a colour. Maps both the original backend names
 * (small/medium/large, from config/tiers.example.yaml) and the Figma names
 * (Economy/Balanced/Premium) to the same three pill tones. Any other name falls back
 * to the neutral `default` tone, which still renders legibly.
 */
const TIER_TONE_MAP: Record<string, Tone> = {
  small: 'tier1',
  medium: 'tier2',
  large: 'tier3',
  economy: 'tier1',
  balanced: 'tier2',
  premium: 'tier3',
  // Common alternates seen in configs out in the wild — kept here so a rename in
  // tiers.yaml doesn't quietly turn a tier pill grey.
  cheap: 'tier1',
  fast: 'tier1',
  standard: 'tier2',
  pro: 'tier3',
  heavy: 'tier3',
  // P1 — UNREVIEWED is catalog-only / never routable; keep it on the neutral tone so it
  // never looks like a declared small/medium/large choice.
  unreviewed: 'default',
}

export function tierTone(name: string): Tone {
  return TIER_TONE_MAP[name.toLowerCase()] ?? 'default'
}

/**
 * Position-aware variant for when the caller has the full ordered tier list (e.g. Model
 * tiers, Overview charts). First tier → `tier1`, last tier → `tier3`, everything in
 * between → `tier2`. Name-map result wins over position so a custom name that happens to
 * be well-known still gets its canonical colour.
 */
export function tierToneByPosition(name: string, orderedNames: string[]): Tone {
  const byName = TIER_TONE_MAP[name.toLowerCase()]
  if (byName) return byName
  const i = orderedNames.findIndex((n) => n.toLowerCase() === name.toLowerCase())
  if (i === -1) return 'default'
  if (orderedNames.length === 1) return 'tier2'
  if (i === 0) return 'tier1'
  if (i === orderedNames.length - 1) return 'tier3'
  return 'tier2'
}

export const theme = createTheme({
  palette: {
    mode: 'light',
    primary: { main: PALETTE.accent, dark: PALETTE.accentAlt, contrastText: PALETTE.onAccent },
    secondary: { main: PALETTE.brand, contrastText: PALETTE.onBrand },
    success: { main: PALETTE.good },
    warning: { main: PALETTE.warning },
    error: { main: PALETTE.critical },
    info: { main: PALETTE.info },
    background: { default: PALETTE.canvas, paper: PALETTE.surface },
    text: { primary: PALETTE.text, secondary: PALETTE.textSecondary },
    divider: PALETTE.hairline,
  },
  shape: { borderRadius: 10 },
  typography: {
    fontFamily: FONT_STACK,
    fontSize: 13.5,
    button: { textTransform: 'none', fontWeight: 600 },
    h6: { fontWeight: 650, letterSpacing: '-0.02em' },
    subtitle2: { fontWeight: 600 },
  },
  shadows: Array(25).fill('none') as unknown as Theme['shadows'],
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        '::selection': { backgroundColor: PALETTE.accentSoft },
        '*': { scrollbarColor: `${PALETTE.hairlineStrong} transparent` },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: PALETTE.chrome,
          color: PALETTE.text,
          borderBottom: `1px solid ${PALETTE.hairline}`,
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: { backgroundColor: PALETTE.chrome, borderRight: `1px solid ${PALETTE.hairline}` },
      },
    },
    MuiListItemButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          marginInline: 10,
          marginBottom: 2,
          minHeight: 36,
          paddingInline: 10,
          transition: 'background-color 120ms ease',
          '&.Mui-selected': {
            backgroundColor: PALETTE.brandSoft,
            color: PALETTE.brandText,
            '& .MuiListItemIcon-root': { color: PALETTE.brand },
            '&:hover': { backgroundColor: PALETTE.brandSoft },
          },
          '&:hover': { backgroundColor: PALETTE.subtle },
          '&:active': { transform: 'scale(0.99)' },
        },
      },
    },
    MuiListItemIcon: { styleOverrides: { root: { minWidth: 32, color: PALETTE.textMuted } } },
    MuiCard: {
      styleOverrides: {
        root: {
          border: `1px solid ${PALETTE.hairline}`,
          boxShadow: 'none',
          backgroundImage: 'none',
          transition: 'border-color 150ms ease',
        },
      },
    },
    MuiPaper: { styleOverrides: { root: { backgroundImage: 'none' } } },
    MuiTableCell: {
      styleOverrides: {
        root: { padding: '9px 14px', fontSize: 13, borderColor: PALETTE.hairline },
        head: {
          backgroundColor: PALETTE.subtle,
          color: PALETTE.textMuted,
          fontSize: 11,
          fontWeight: 650,
          letterSpacing: '0.05em',
          textTransform: 'uppercase',
        },
      },
    },
    MuiTableRow: {
      styleOverrides: {
        root: {
          transition: 'background-color 120ms ease',
          '&:hover td': { backgroundColor: PALETTE.subtle },
        },
      },
    },
    MuiButton: {
      defaultProps: { size: 'small', disableElevation: true },
      styleOverrides: {
        root: {
          borderRadius: 8,
          paddingInline: 14,
          fontWeight: 600,
          transition: 'transform 100ms ease, background-color 120ms ease, border-color 120ms ease',
          '&:active': { transform: 'scale(0.97)' },
        },
        outlined: { borderColor: PALETTE.hairlineStrong, color: PALETTE.text },
        contained: {
          '&.MuiButton-colorPrimary': {
            backgroundColor: PALETTE.accent,
            '&:hover': { backgroundColor: PALETTE.accentAlt },
          },
        },
      },
    },
    MuiIconButton: {
      styleOverrides: {
        root: {
          transition: 'transform 100ms ease, background-color 120ms ease',
          '&:active': { transform: 'scale(0.92)' },
        },
      },
    },
    MuiTextField: { defaultProps: { size: 'small' } },
    MuiChip: {
      defaultProps: { size: 'small' },
      styleOverrides: { root: { borderRadius: 6, fontWeight: 600 } },
    },
    MuiAlert: { styleOverrides: { root: { borderRadius: 10 } } },
    MuiCheckbox: {
      styleOverrides: {
        root: {
          color: PALETTE.hairlineStrong,
          transition: 'transform 100ms ease',
          '&:active': { transform: 'scale(0.9)' },
        },
      },
    },
    MuiSwitch: {
      defaultProps: { size: 'small' },
      styleOverrides: {
        root: {
          '& .MuiSwitch-switchBase.Mui-checked': { color: PALETTE.accent },
          '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': { backgroundColor: PALETTE.accent, opacity: 0.6 },
        },
      },
    },
  },
})
