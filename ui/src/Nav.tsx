import {
  Brain,
  CaretDown,
  ChartBar,
  CheckSquare,
  ClipboardText,
  Database,
  Flask,
  FlowArrow,
  SquaresFour,
  Stack as StackIcon,
  Storefront,
} from '@phosphor-icons/react'
import { Badge, Box, List, ListItemButton, ListItemIcon, ListItemText, Typography } from '@mui/material'
import type { ReactNode } from 'react'
import { PALETTE } from './theme'

export type PageKey = 'overview' | 'rules' | 'tiers' | 'catalog' | 'examples' | 'judging' | 'train' | 'evaluate' | 'history' | 'resolve'

interface Item {
  key: PageKey
  label: string
  icon: ReactNode
}

interface Group {
  label: string
  items: Item[]
}

const ICON_SIZE = 18
const ICON_PROPS = { size: ICON_SIZE, weight: 'regular' as const }

const GROUPS: Group[] = [
  {
    label: 'Platform',
    items: [
      { key: 'overview', label: 'Overview', icon: <SquaresFour {...ICON_PROPS} /> },
      { key: 'rules', label: 'Routing rules', icon: <FlowArrow {...ICON_PROPS} /> },
      { key: 'tiers', label: 'Model tiers', icon: <StackIcon {...ICON_PROPS} /> },
      { key: 'catalog', label: 'Model catalog', icon: <Storefront {...ICON_PROPS} /> },
      { key: 'history', label: 'Request logs', icon: <ClipboardText {...ICON_PROPS} /> },
      { key: 'resolve', label: 'Playground', icon: <Flask {...ICON_PROPS} /> },
    ],
  },
  {
    label: 'Training',
    items: [
      { key: 'examples', label: 'Examples', icon: <Database {...ICON_PROPS} /> },
      { key: 'judging', label: 'Human judging', icon: <CheckSquare {...ICON_PROPS} /> },
      { key: 'train', label: 'Train', icon: <Brain {...ICON_PROPS} /> },
      { key: 'evaluate', label: 'Evaluate', icon: <ChartBar {...ICON_PROPS} /> },
    ],
  },
]

/**
 * A flat, grouped destination list — one full page per item (see App.tsx). The `Platform`
 * group mirrors the Figma reference exactly (six items: overview → playground). The
 * `Training` group is this project's addition for the human-judged-labels workflow that
 * the reference deck doesn't depict but the service actually ships (examples + judging +
 * train + evaluate, all wired through `/v1/*` already).
 */
export function Nav({
  active,
  onNavigate,
  historyCount,
  workspaceName,
  workspaceTag,
  onWorkspaceClick,
}: {
  active: PageKey
  onNavigate: (key: PageKey) => void
  historyCount: number
  workspaceName: string
  workspaceTag?: string
  onWorkspaceClick?: () => void
}) {
  return (
    <>
      <Box sx={{ px: 2, pt: 1.5, pb: 1 }}>
        <Typography
          sx={{
            fontSize: 10.5,
            fontWeight: 700,
            letterSpacing: '0.08em',
            textTransform: 'uppercase',
            color: PALETTE.textFaint,
            mb: 0.75,
          }}
        >
          Workspace
        </Typography>
        <Box
          role="button"
          onClick={onWorkspaceClick}
          sx={{
            display: 'flex',
            alignItems: 'center',
            gap: 1,
            px: 1,
            py: 0.85,
            borderRadius: '8px',
            border: `1px solid ${PALETTE.hairline}`,
            bgcolor: PALETTE.surface,
            cursor: onWorkspaceClick ? 'pointer' : 'default',
            transition: 'background-color 120ms ease, border-color 120ms ease',
            '&:hover': onWorkspaceClick ? { bgcolor: PALETTE.subtle, borderColor: PALETTE.hairlineStrong } : undefined,
          }}
        >
          <Box
            sx={{
              flex: '0 0 auto',
              width: 22,
              height: 22,
              borderRadius: '6px',
              bgcolor: PALETTE.brandSoft,
              color: PALETTE.brand,
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 11,
              fontWeight: 700,
              letterSpacing: '0.02em',
            }}
            aria-hidden
          >
            {workspaceName.slice(0, 1).toUpperCase()}
          </Box>
          <Box sx={{ minWidth: 0, flex: 1 }}>
            <Typography
              noWrap
              sx={{ fontSize: 13, fontWeight: 600, color: PALETTE.text, lineHeight: 1.2 }}
            >
              {workspaceName}
            </Typography>
            {workspaceTag && (
              <Typography sx={{ fontSize: 10.5, color: PALETTE.textFaint, lineHeight: 1.2 }}>
                {workspaceTag}
              </Typography>
            )}
          </Box>
          <CaretDown size={12} color={PALETTE.textFaint} />
        </Box>
      </Box>

      <List component="nav" aria-label="Sections" dense sx={{ py: 1 }}>
        {GROUPS.map((group) => (
          <List key={group.label} component="div" dense disablePadding sx={{ mb: 0.5 }}>
            <Typography
              sx={{
                fontSize: 10.5,
                fontWeight: 700,
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
                color: PALETTE.textFaint,
                px: 2,
                pt: 1,
                pb: 0.5,
              }}
            >
              {group.label}
            </Typography>
            {group.items.map((item) => (
              <ListItemButton key={item.key} selected={active === item.key} onClick={() => onNavigate(item.key)}>
                <ListItemIcon>{item.icon}</ListItemIcon>
                <ListItemText primary={item.label} slotProps={{ primary: { sx: { fontSize: 13.5, fontWeight: active === item.key ? 650 : 500 } } }} />
                {item.key === 'history' && historyCount > 0 && (
                  <Badge
                    badgeContent={historyCount}
                    max={999}
                    sx={{ '& .MuiBadge-badge': { position: 'static', transform: 'none', bgcolor: PALETTE.raised, color: PALETTE.textMuted, fontWeight: 600, borderRadius: '6px' } }}
                  />
                )}
              </ListItemButton>
            ))}
          </List>
        ))}
      </List>
    </>
  )
}
