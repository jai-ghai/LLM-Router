import { ClipboardText, Flask, FlowArrow, SquaresFour } from '@phosphor-icons/react'
import { Badge, List, ListItemButton, ListItemIcon, ListItemText, Typography } from '@mui/material'
import type { ReactNode } from 'react'
import { PALETTE } from './theme'
import type { UserRole } from './types'

export type PageKey =
  | 'overview'
  | 'rules'
  | 'examples'
  | 'judging'
  | 'train'
  | 'evaluate'
  | 'history'
  | 'resolve'
  | 'users'
  | 'settings'
  | 'docs'

interface Item {
  key: PageKey
  label: string
  icon: ReactNode
  adminOnly?: boolean
  /** Hide when a model catalog is configured (tier-ladder / classifier unused on resolve). */
  hideWhenCatalog?: boolean
}

interface Group {
  label: string
  items: Item[]
  adminOnly?: boolean
  hideWhenCatalog?: boolean
}

const ICON_SIZE = 18
const ICON_PROPS = { size: ICON_SIZE, weight: 'regular' as const }

const GROUPS: Group[] = [
  {
    label: 'Platform',
    items: [
      { key: 'overview', label: 'Overview', icon: <SquaresFour {...ICON_PROPS} /> },
      { key: 'rules', label: 'Routing rules', icon: <FlowArrow {...ICON_PROPS} />, hideWhenCatalog: true },
      { key: 'history', label: 'Request logs', icon: <ClipboardText {...ICON_PROPS} /> },
      { key: 'resolve', label: 'Playground', icon: <Flask {...ICON_PROPS} /> },
    ],
  },
  // Training is an operator workflow, not something end users need; the pages still exist.
  // {
  //   label: 'Training',
  //   hideWhenCatalog: true,
  //   items: [
  //     { key: 'examples', label: 'Examples', icon: <Database {...ICON_PROPS} />, hideWhenCatalog: true },
  //     { key: 'judging', label: 'Human judging', icon: <CheckSquare {...ICON_PROPS} />, hideWhenCatalog: true },
  //     { key: 'train', label: 'Train', icon: <Brain {...ICON_PROPS} />, hideWhenCatalog: true },
  //     { key: 'evaluate', label: 'Evaluate', icon: <ChartBar {...ICON_PROPS} />, hideWhenCatalog: true },
  //   ],
  // },
  // Admin (Operators) and Help (Documentation, Settings) live in the user menu.
  // {
  //   label: 'Admin',
  //   adminOnly: true,
  //   items: [
  //     { key: 'users', label: 'Operators', icon: <UsersThree {...ICON_PROPS} />, adminOnly: true },
  //   ],
  // },
  // {
  //   label: 'Help',
  //   items: [
  //     { key: 'docs', label: 'Documentation', icon: <BookOpen {...ICON_PROPS} /> },
  //     { key: 'settings', label: 'Settings', icon: <GearSix {...ICON_PROPS} /> },
  //   ],
  // },
]

/**
 * Grouped destination list — one full page per item (see App.tsx). No workspace /
 * environment chrome here: those were Figma placeholders with no backend model.
 */
export function Nav({
  active,
  onNavigate,
  historyCount,
  role,
  /** null = status not loaded yet — omit ladder items so they don't flash then vanish. */
  catalogConfigured = null,
}: {
  active: PageKey
  onNavigate: (key: PageKey) => void
  historyCount: number
  role: UserRole | null
  catalogConfigured?: boolean | null
}) {
  // Hybrid resolve (content→tier→catalog): always show Training / Rules / Tiers.
  // App.tsx renders <Nav />. User: generalized app — don't hide ladder when catalog on.
  void catalogConfigured
  const hideLadder = false
  const visibleGroups = GROUPS.map((g) => ({
    ...g,
    items: g.items.filter(
      (item) => (!item.adminOnly || role === 'admin') && !(hideLadder && item.hideWhenCatalog),
    ),
  })).filter(
    (g) => (!g.adminOnly || role === 'admin') && !(hideLadder && g.hideWhenCatalog) && g.items.length > 0,
  )
  return (
    <List component="nav" aria-label="Sections" dense sx={{ py: 1 }}>
      {visibleGroups.map((group) => (
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
  )
}
