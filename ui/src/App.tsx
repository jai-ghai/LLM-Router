import { Bell, MagnifyingGlass, GearSix, BookOpen, SignOut } from '@phosphor-icons/react'
import {
  Alert,
  AppBar,
  Avatar,
  Box,
  Divider,
  Drawer,
  IconButton,
  Menu,
  MenuItem,
  Stack,
  Toolbar,
  Typography,
} from '@mui/material'
import { useEffect, useState, type MouseEvent } from 'react'
import { api, describeError } from './api'
import { Nav, type PageKey } from './Nav'
import { PageTransition, StateTag } from './components/Primitives'
import { Catalog } from './screens/Catalog'
import { Evaluate } from './screens/Evaluate'
import { Examples } from './screens/Examples'
import { History } from './screens/History'
import { HumanJudging } from './screens/HumanJudging'
import { Overview } from './screens/Overview'
import { Resolve } from './screens/Resolve'
import { Rules } from './screens/Rules'
import { Tiers } from './screens/Tiers'
import { Train } from './screens/Train'
import { PALETTE } from './theme'
import type { Status, TiersResponse, TrainResult } from './types'

const DRAWER_WIDTH = 248

const PAGE_TITLES: Record<PageKey, string> = {
  overview: 'Overview',
  rules: 'Routing rules',
  tiers: 'Model tiers',
  catalog: 'Model catalog',
  examples: 'Examples',
  judging: 'Human judging',
  train: 'Train',
  evaluate: 'Evaluate',
  history: 'Request logs',
  resolve: 'Playground',
}

/**
 * Sidebar-routed console: one full page per nav item (see Nav.tsx). Matches the Figma
 * reference layout — persistent left drawer with a workspace selector + grouped sections
 * + status footer; top app bar with breadcrumb + environment pill + search/notifications
 * + user menu. Page-shared state (the tier table, catalog selection, last training
 * result, history badge count) is lifted here and threaded down.
 */
export default function App() {
  const [active, setActive] = useState<PageKey>('overview')
  const [online, setOnline] = useState<'checking' | 'online' | 'offline'>('checking')
  const [error, setError] = useState('')
  const [tiers, setTiers] = useState<TiersResponse | null>(null)
  const [status, setStatus] = useState<Status | null>(null)
  const [lastTrainResult, setLastTrainResult] = useState<TrainResult | null>(null)
  const [selectedModels, setSelectedModels] = useState<Set<string> | null>(null)
  const [historyRefresh, setHistoryRefresh] = useState(0)
  const [historyCount, setHistoryCount] = useState(0)
  const [userMenuAnchor, setUserMenuAnchor] = useState<HTMLElement | null>(null)

  async function check() {
    setOnline('checking')
    try {
      await api.health()
      setTiers(await api.tiers())
      setStatus(await api.status())
      setOnline('online')
    } catch (err) {
      setError(describeError(err))
      setOnline('offline')
    }
  }

  useEffect(() => {
    void check()
  }, [])

  useEffect(() => {
    void (async () => {
      try {
        setHistoryCount((await api.history(1000)).history.length)
      } catch {
        // The nav badge is cosmetic; a failed count just stays at its last known value.
      }
    })()
  }, [historyRefresh])

  function onResolved() {
    setHistoryRefresh((n) => n + 1)
  }

  async function onTrained(result: TrainResult) {
    setLastTrainResult(result)
    try {
      setStatus(await api.status())
    } catch {
      // Status refresh is informational only.
    }
  }

  // Environment pill: "Production" when the service is up, "Offline" when it is not,
  // "Preview" while still checking — matches the badge in the Figma header without
  // needing extra config.
  const envTone = online === 'online' ? 'good' : online === 'offline' ? 'critical' : 'default'
  const envLabel = online === 'online' ? 'Production' : online === 'offline' ? 'Offline' : 'Preview'

  return (
    <Box sx={{ display: 'flex', minHeight: '100dvh', bgcolor: PALETTE.canvas }}>
      <Drawer
        variant="permanent"
        sx={{ width: DRAWER_WIDTH, flexShrink: 0, [`& .MuiDrawer-paper`]: { width: DRAWER_WIDTH, boxSizing: 'border-box', border: 'none' } }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.25, px: 2, py: 2 }}>
          <Box
            sx={{
              width: 30,
              height: 30,
              borderRadius: '9px',
              bgcolor: PALETTE.brand,
              color: PALETTE.onBrand,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flex: '0 0 auto',
              fontSize: 15,
              fontWeight: 800,
              letterSpacing: '-0.02em',
            }}
            aria-hidden
          >
            tr
          </Box>
          <Typography sx={{ fontWeight: 700, fontSize: 16, letterSpacing: '-0.015em', color: PALETTE.text }}>
            tierrouter
          </Typography>
        </Box>
        <Divider />
        <Box sx={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column' }}>
          <Nav
            active={active}
            onNavigate={setActive}
            historyCount={historyCount}
            workspaceName="Acme AI"
            workspaceTag={online === 'offline' ? 'Service offline' : 'Sample workspace'}
          />
        </Box>

        <Divider />
        <Box sx={{ px: 2, py: 1.5 }}>
          <Stack direction="row" spacing={0.75} sx={{ alignItems: 'center', mb: 0.75 }}>
            <Box
              sx={{
                width: 7,
                height: 7,
                borderRadius: '50%',
                bgcolor: online === 'online' ? PALETTE.good : online === 'checking' ? PALETTE.textFaint : PALETTE.critical,
                flex: '0 0 auto',
              }}
              aria-hidden
            />
            <Typography sx={{ fontSize: 12, fontWeight: 600, color: PALETTE.text }}>
              {online === 'online' ? 'All systems operational' : online === 'checking' ? 'Checking service…' : 'Service offline'}
            </Typography>
          </Stack>
          <Stack direction="row" sx={{ justifyContent: 'space-between' }}>
            <Typography sx={{ fontSize: 11, color: PALETTE.textMuted }}>FastAPI service</Typography>
            <Typography sx={{ fontSize: 11, color: online === 'online' ? PALETTE.goodText : PALETTE.textMuted, fontWeight: 600 }}>
              {online === 'online' ? 'Connected' : online === 'offline' ? 'Offline' : '—'}
            </Typography>
          </Stack>
          <Stack direction="row" sx={{ justifyContent: 'space-between', mt: 0.25 }}>
            <Typography sx={{ fontSize: 11, color: PALETTE.textMuted }}>Classifier</Typography>
            <Typography sx={{ fontSize: 11, color: PALETTE.textMuted, fontWeight: 600 }}>
              {status?.trained ? `trained · ${status.n_labels_last_trained_on ?? 0} labels` : 'not trained'}
            </Typography>
          </Stack>
        </Box>
        <Divider />
        <Box sx={{ px: 1, py: 1 }}>
          <Stack
            component="a"
            href="https://github.com/"
            target="_blank"
            rel="noreferrer"
            direction="row"
            spacing={1.25}
            sx={{
              alignItems: 'center',
              px: 1.25,
              py: 0.85,
              borderRadius: '8px',
              textDecoration: 'none',
              color: PALETTE.textMuted,
              '&:hover': { bgcolor: PALETTE.subtle, color: PALETTE.text },
            }}
          >
            <BookOpen size={16} />
            <Typography sx={{ fontSize: 13, fontWeight: 500 }}>Documentation</Typography>
          </Stack>
          <Stack
            direction="row"
            spacing={1.25}
            sx={{
              alignItems: 'center',
              px: 1.25,
              py: 0.85,
              borderRadius: '8px',
              color: PALETTE.textMuted,
              cursor: 'pointer',
              '&:hover': { bgcolor: PALETTE.subtle, color: PALETTE.text },
            }}
          >
            <GearSix size={16} />
            <Typography sx={{ fontSize: 13, fontWeight: 500 }}>Settings</Typography>
          </Stack>
        </Box>
      </Drawer>

      <Box component="main" sx={{ flexGrow: 1, minWidth: 0 }}>
        <AppBar position="sticky" sx={{ zIndex: (t) => t.zIndex.drawer - 1 }} elevation={0}>
          <Toolbar variant="dense" sx={{ gap: 1.5 }}>
            <Typography sx={{ fontSize: 12.5, color: PALETTE.textMuted, flex: 1, minWidth: 0 }}>
              Workspace <Box component="span" sx={{ mx: 0.75, color: PALETTE.textFaint }}>/</Box>
              <Box component="span" sx={{ color: PALETTE.text, fontWeight: 600 }}>{PAGE_TITLES[active]}</Box>
            </Typography>
            <StateTag label={envLabel} tone={envTone} size="small" />
            <IconButton size="small" aria-label="Search" sx={{ color: PALETTE.textMuted }}>
              <MagnifyingGlass size={16} />
            </IconButton>
            <IconButton size="small" aria-label="Notifications" sx={{ color: PALETTE.textMuted }}>
              <Bell size={16} />
            </IconButton>
            <Stack
              direction="row"
              spacing={0.75}
              onClick={(e: MouseEvent<HTMLDivElement>) => setUserMenuAnchor(e.currentTarget)}
              sx={{
                alignItems: 'center',
                px: 1,
                py: 0.4,
                borderRadius: '8px',
                cursor: 'pointer',
                '&:hover': { bgcolor: PALETTE.subtle },
              }}
              aria-haspopup="menu"
              aria-expanded={Boolean(userMenuAnchor)}
            >
              <Avatar sx={{ width: 22, height: 22, bgcolor: PALETTE.brandSoft, color: PALETTE.brand, fontSize: 11, fontWeight: 700 }}>
                A
              </Avatar>
              <Typography sx={{ fontSize: 12.5, fontWeight: 600, color: PALETTE.text }}>ACME</Typography>
            </Stack>
            <Menu
              anchorEl={userMenuAnchor}
              open={Boolean(userMenuAnchor)}
              onClose={() => setUserMenuAnchor(null)}
              anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
              transformOrigin={{ vertical: 'top', horizontal: 'right' }}
              slotProps={{ paper: { sx: { minWidth: 180, mt: 0.5 } } }}
            >
              <MenuItem onClick={() => setUserMenuAnchor(null)} sx={{ fontSize: 13 }}>
                <BookOpen size={14} style={{ marginRight: 8 }} />
                Documentation
              </MenuItem>
              <MenuItem onClick={() => setUserMenuAnchor(null)} sx={{ fontSize: 13 }}>
                <GearSix size={14} style={{ marginRight: 8 }} />
                Settings
              </MenuItem>
              <Divider />
              <MenuItem onClick={() => setUserMenuAnchor(null)} sx={{ fontSize: 13 }}>
                <SignOut size={14} style={{ marginRight: 8 }} />
                Sign out
              </MenuItem>
            </Menu>
          </Toolbar>
        </AppBar>

        <Box sx={{ p: 3, maxWidth: 1360, mx: 'auto' }}>
          {online === 'offline' && (
            <Alert severity="error" sx={{ mb: 2 }}>
              Cannot reach the service at {import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'}: {error}. Start it
              with TIERROUTER_TIERS and TIERROUTER_DATA set, then uvicorn tierrouter.service.main:create --factory.
            </Alert>
          )}
          <PageTransition routeKey={active}>
            {active === 'overview' && <Overview tiers={tiers} onlineStatus={online} onNavigate={setActive} />}
            {active === 'rules' && <Rules tierNames={tiers?.tiers.map((t) => t.name) ?? []} />}
            {active === 'tiers' && <Tiers tiers={tiers} />}
            {active === 'catalog' && <Catalog selected={selectedModels} onChange={setSelectedModels} />}
            {active === 'examples' && <Examples />}
            {active === 'judging' && <HumanJudging tierNames={tiers?.tiers.map((t) => t.name) ?? []} />}
            {active === 'train' && <Train onTrained={(r) => void onTrained(r)} result={lastTrainResult} />}
            {active === 'evaluate' && <Evaluate result={lastTrainResult} />}
            {active === 'history' && <History refreshKey={historyRefresh} />}
            {active === 'resolve' && <Resolve tiers={tiers} selectedModels={selectedModels} onResolved={onResolved} />}
          </PageTransition>
        </Box>
      </Box>
    </Box>
  )
}
