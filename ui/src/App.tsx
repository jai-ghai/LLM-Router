import { Bell, BookOpen, GearSix, MagnifyingGlass, SignOut } from '@phosphor-icons/react'
import {
  Alert,
  AppBar,
  Avatar,
  Badge,
  Box,
  Dialog,
  DialogContent,
  Divider,
  Drawer,
  IconButton,
  List,
  ListItemButton,
  ListItemText,
  Menu,
  MenuItem,
  Stack,
  TextField,
  Toolbar,
  Typography,
} from '@mui/material'
import { useEffect, useMemo, useState, type MouseEvent } from 'react'
import { api, describeError } from './api'
import { AuthProvider, useAuth } from './auth'
import { Nav, type PageKey } from './Nav'
import { PageTransition } from './components/Primitives'
import { Docs } from './screens/Docs'
import { Evaluate } from './screens/Evaluate'
import { Examples } from './screens/Examples'
import { History } from './screens/History'
import { HumanJudging } from './screens/HumanJudging'
import { Login } from './screens/Login'
import { Overview } from './screens/Overview'
import { Resolve } from './screens/Resolve'
import { Rules } from './screens/Rules'
import { Settings } from './screens/Settings'
import { Train } from './screens/Train'
import { Users } from './screens/Users'
import { PALETTE } from './theme'
import type { Status, TiersResponse, TrainResult } from './types'

const DRAWER_WIDTH = 248

const PAGE_TITLES: Record<PageKey, string> = {
  overview: 'Overview',
  rules: 'Routing rules',
  examples: 'Examples',
  judging: 'Human judging',
  train: 'Train',
  evaluate: 'Evaluate',
  history: 'Request logs',
  resolve: 'Playground',
  users: 'Operators',
  settings: 'Settings',
  docs: 'Documentation',
}

const LADDER_PAGES = new Set<PageKey>(['rules', 'examples', 'judging', 'train', 'evaluate'])

const SEARCHABLE: { key: PageKey; label: string; adminOnly?: boolean; hideWhenCatalog?: boolean }[] = (
  Object.entries(PAGE_TITLES) as [PageKey, string][]
).map(([key, label]) => ({
  key,
  label,
  adminOnly: key === 'users',
  hideWhenCatalog: LADDER_PAGES.has(key),
}))

/**
 * Entry component — wraps everything in AuthProvider so every screen can call useAuth().
 * Rendering is split: anonymous state renders Login, authenticated renders the main shell.
 */
export default function App() {
  return (
    <AuthProvider>
      <AuthGate />
    </AuthProvider>
  )
}

function AuthGate() {
  const { state } = useAuth()
  if (state.status === 'loading') {
    return (
      <Box sx={{ minHeight: '100dvh', display: 'flex', alignItems: 'center', justifyContent: 'center', bgcolor: PALETTE.canvas }}>
        <Typography sx={{ fontSize: 13, color: PALETTE.textMuted }}>Loading…</Typography>
      </Box>
    )
  }
  if (state.status === 'anonymous') return <Login />
  return <AuthenticatedShell />
}

/**
 * Sidebar-routed console shown once the user is authenticated.
 */
function AuthenticatedShell() {
  const { state, logout } = useAuth()
  const user = state.status === 'authenticated' ? state.user : null
  const [active, setActive] = useState<PageKey>('overview')
  const [online, setOnline] = useState<'checking' | 'online' | 'offline'>('checking')
  const [error, setError] = useState('')
  const [tiers, setTiers] = useState<TiersResponse | null>(null)
  const [status, setStatus] = useState<Status | null>(null)
  const [lastTrainResult, setLastTrainResult] = useState<TrainResult | null>(null)
  const [historyRefresh, setHistoryRefresh] = useState(0)
  const [historyCount, setHistoryCount] = useState(0)
  const [userMenuAnchor, setUserMenuAnchor] = useState<HTMLElement | null>(null)
  const [bellAnchor, setBellAnchor] = useState<HTMLElement | null>(null)
  const [searchOpen, setSearchOpen] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')

  const unreviewed = status?.unreviewed_count ?? 0
  // null until /v1/status returns — avoids sidebar showing Training then yanking it away.
  const catalogConfigured: boolean | null = status == null ? null : Boolean(status.catalog_configured)
  const catalogOn = catalogConfigured === true
  const searchHits = useMemo(() => {
    const q = searchQuery.trim().toLowerCase()
    return SEARCHABLE.filter((item) => {
      if (item.adminOnly && user?.role !== 'admin') return false
      // Match Nav: hide ladder pages until we know catalog is off.
      if (catalogConfigured !== false && item.hideWhenCatalog) return false
      return !q || item.label.toLowerCase().includes(q)
    })
  }, [searchQuery, user?.role, catalogConfigured])

  // Hybrid: catalog no longer hides ladder pages — keep user's active route.
  void LADDER_PAGES

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
            gd
          </Box>
          <Typography sx={{ fontWeight: 700, fontSize: 16, letterSpacing: '-0.015em', color: PALETTE.text }}>
            Gradist
          </Typography>
        </Box>
        <Divider />
        <Box sx={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column' }}>
          <Nav
            active={active}
            onNavigate={setActive}
            historyCount={historyCount}
            role={user?.role ?? null}
            catalogConfigured={catalogConfigured}
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
          {/* Sidebar footer — AuthenticatedShell only. Status from /v1/status. */}
          {status != null && !status.catalog_configured && (
            <Stack direction="row" sx={{ justifyContent: 'space-between', mt: 0.25 }}>
              <Typography sx={{ fontSize: 11, color: PALETTE.textMuted }}>Classifier</Typography>
              <Typography sx={{ fontSize: 11, color: PALETTE.textMuted, fontWeight: 600 }}>
                {status.trained
                  ? `trained · ${status.n_labels_last_trained_on ?? 0}${status.uses_images ? ' · images' : ''}`
                  : 'not trained'}
              </Typography>
            </Stack>
          )}
          {catalogOn && (
            <Stack direction="row" sx={{ justifyContent: 'space-between', mt: 0.25 }}>
              <Typography sx={{ fontSize: 11, color: PALETTE.textMuted }}>Catalog</Typography>
              <Typography sx={{ fontSize: 11, color: PALETTE.textMuted, fontWeight: 600 }}>configured</Typography>
            </Stack>
          )}
        </Box>
      </Drawer>

      <Box component="main" sx={{ flexGrow: 1, minWidth: 0 }}>
        <AppBar position="sticky" sx={{ zIndex: (t) => t.zIndex.drawer - 1 }} elevation={0}>
          <Toolbar variant="dense" sx={{ gap: 1 }}>
            <Typography sx={{ fontSize: 12.5, fontWeight: 600, color: PALETTE.text, flex: 1, minWidth: 0 }}>
              {PAGE_TITLES[active]}
            </Typography>
            <IconButton
              size="small"
              aria-label="Search pages"
              onClick={() => {
                setSearchQuery('')
                setSearchOpen(true)
              }}
              sx={{ color: PALETTE.textMuted }}
            >
              <MagnifyingGlass size={16} />
            </IconButton>
            {/* Catalog-only: UNREVIEWED model alerts. Hidden in ladder mode (noise). */}
            {catalogOn && (
              <>
                <IconButton
                  size="small"
                  aria-label="Alerts"
                  onClick={(e) => {
                    setBellAnchor(e.currentTarget)
                    void api.status().then(setStatus).catch(() => undefined)
                  }}
                  sx={{ color: PALETTE.textMuted }}
                >
                  <Badge
                    color="error"
                    badgeContent={unreviewed}
                    max={99}
                    invisible={unreviewed === 0}
                    sx={{ '& .MuiBadge-badge': { fontSize: 10, height: 16, minWidth: 16 } }}
                  >
                    <Bell size={16} />
                  </Badge>
                </IconButton>
                <Menu
                  anchorEl={bellAnchor}
                  open={Boolean(bellAnchor)}
                  onClose={() => setBellAnchor(null)}
                  anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
                  transformOrigin={{ vertical: 'top', horizontal: 'right' }}
                  slotProps={{ paper: { sx: { minWidth: 260, mt: 0.5 } } }}
                >
                  {unreviewed > 0 ? (
                    <MenuItem
                      onClick={() => {
                        setActive('resolve')
                        setBellAnchor(null)
                      }}
                      sx={{ fontSize: 13, whiteSpace: 'normal', alignItems: 'flex-start' }}
                    >
                      <Box>
                        <Typography sx={{ fontSize: 13, fontWeight: 600 }}>
                          {unreviewed} model{unreviewed === 1 ? '' : 's'} still UNREVIEWED
                        </Typography>
                        <Typography sx={{ fontSize: 11.5, color: PALETTE.textMuted }}>
                          Assign tiers when creating a task in Playground.
                        </Typography>
                      </Box>
                    </MenuItem>
                  ) : (
                    <Box sx={{ px: 2, py: 1.5 }}>
                      <Typography sx={{ fontSize: 13, color: PALETTE.textMuted }}>No pending alerts.</Typography>
                    </Box>
                  )}
                </Menu>
              </>
            )}
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
                {(user?.username ?? '?').slice(0, 1).toUpperCase()}
              </Avatar>
              <Typography sx={{ fontSize: 12.5, fontWeight: 600, color: PALETTE.text }}>
                {user?.username ?? '—'}
              </Typography>
            </Stack>
            <Menu
              anchorEl={userMenuAnchor}
              open={Boolean(userMenuAnchor)}
              onClose={() => setUserMenuAnchor(null)}
              anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
              transformOrigin={{ vertical: 'top', horizontal: 'right' }}
              slotProps={{ paper: { sx: { minWidth: 200, mt: 0.5 } } }}
            >
              <Box sx={{ px: 2, py: 1 }}>
                <Typography sx={{ fontSize: 12.5, fontWeight: 600, color: PALETTE.text }}>
                  {user?.username ?? '—'}
                </Typography>
                <Typography sx={{ fontSize: 11, color: PALETTE.textMuted }}>
                  {user?.role ?? 'unknown role'}
                </Typography>
              </Box>
              <Divider />
              <MenuItem
                onClick={() => {
                  setActive('docs')
                  setUserMenuAnchor(null)
                }}
                sx={{ fontSize: 13 }}
              >
                <BookOpen size={14} style={{ marginRight: 8 }} />
                Documentation
              </MenuItem>
              <MenuItem
                onClick={() => {
                  setActive('settings')
                  setUserMenuAnchor(null)
                }}
                sx={{ fontSize: 13 }}
              >
                <GearSix size={14} style={{ marginRight: 8 }} />
                Settings
              </MenuItem>
              {user?.role === 'admin' && (
                <MenuItem
                  onClick={() => {
                    setActive('users')
                    setUserMenuAnchor(null)
                  }}
                  sx={{ fontSize: 13 }}
                >
                  <GearSix size={14} style={{ marginRight: 8 }} />
                  Operators
                </MenuItem>
              )}
              <Divider />
              <MenuItem
                onClick={() => {
                  setUserMenuAnchor(null)
                  void logout()
                }}
                sx={{ fontSize: 13 }}
              >
                <SignOut size={14} style={{ marginRight: 8 }} />
                Sign out
              </MenuItem>
            </Menu>
          </Toolbar>
        </AppBar>

        <Dialog open={searchOpen} onClose={() => setSearchOpen(false)} fullWidth maxWidth="xs">
          <DialogContent sx={{ pt: 2 }}>
            <TextField
              autoFocus
              fullWidth
              size="small"
              placeholder="Jump to a page…"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              slotProps={{ htmlInput: { 'aria-label': 'Search pages' } }}
            />
            <List dense sx={{ mt: 1, maxHeight: 320, overflowY: 'auto' }}>
              {searchHits.map((item) => (
                <ListItemButton
                  key={item.key}
                  onClick={() => {
                    setActive(item.key)
                    setSearchOpen(false)
                  }}
                >
                  <ListItemText primary={item.label} slotProps={{ primary: { sx: { fontSize: 13.5 } } }} />
                </ListItemButton>
              ))}
              {searchHits.length === 0 && (
                <Typography sx={{ px: 2, py: 1.5, fontSize: 13, color: PALETTE.textMuted }}>No pages match.</Typography>
              )}
            </List>
          </DialogContent>
        </Dialog>

        <Box sx={{ p: 3, maxWidth: 1360, mx: 'auto' }}>
          {online === 'offline' && (
            <Alert severity="error" sx={{ mb: 2 }}>
              Cannot reach the Gradist service{import.meta.env.VITE_API_BASE ? ` at ${import.meta.env.VITE_API_BASE}` : ''}.
              {error ? ` ${error}` : ''} Check that the API is running, then refresh.
            </Alert>
          )}
          <PageTransition routeKey={active}>
            {active === 'overview' && <Overview tiers={tiers} onlineStatus={online} onNavigate={setActive} />}
            {active === 'rules' && <Rules tierNames={tiers?.tiers.map((t) => t.name) ?? []} />}
            {active === 'examples' && <Examples />}
            {active === 'judging' && <HumanJudging tierNames={tiers?.tiers.map((t) => t.name) ?? []} />}
            {active === 'train' && <Train onTrained={(r) => void onTrained(r)} result={lastTrainResult} />}
            {active === 'evaluate' && <Evaluate result={lastTrainResult} />}
            {active === 'history' && <History refreshKey={historyRefresh} />}
            {active === 'resolve' && (
              <Resolve tiers={tiers} catalogConfigured={catalogOn} onResolved={onResolved} />
            )}
            {active === 'users' && <Users />}
            {active === 'settings' && <Settings />}
            {active === 'docs' && <Docs />}
          </PageTransition>
        </Box>
      </Box>
    </Box>
  )
}
