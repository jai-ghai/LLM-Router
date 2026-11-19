import { ArrowRight, ArrowCounterClockwise, Copy, MagnifyingGlass, Plus } from '@phosphor-icons/react'
import {
  Alert,
  Box,
  Button,
  Checkbox,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  InputAdornment,
  MenuItem,
  Slider,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TablePagination,
  TableRow,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from '@mui/material'
import { useEffect, useMemo, useState } from 'react'
import { api, describeError } from '../api'
import { ImagePicker } from '../components/ImagePicker'
import { MetadataFieldsEditor } from '../components/MetadataFieldsEditor'
import { Mono, PageHead, Panel, StateTag } from '../components/Primitives'
import { PALETTE, TONE_SOFT, tierTone } from '../theme'
import type { CatalogEntry, Decision, OptimizeFor, TiersResponse } from '../types'

/**
 * Playground — ladder (rules→classifier→fallback) or catalog task-pool pick.
 * Media: image or video (browser extracts a still). No LLM call.
 */
export function Resolve({
  tiers,
  catalogConfigured = false,
  onResolved,
}: {
  tiers: TiersResponse | null
  catalogConfigured?: boolean
  onResolved: () => void
}) {
  // App.tsx renders <Resolve />; mediaKind/source drive ImagePicker video→still UI.
  const [metadata, setMetadata] = useState<Record<string, string>>({})
  const [text, setText] = useState('')
  const [image, setImage] = useState<File | null>(null)
  const [mediaKind, setMediaKind] = useState<'image' | 'video' | null>(null)
  const [mediaSourceName, setMediaSourceName] = useState<string | null>(null)
  const [provider, setProvider] = useState('')
  const [minConfidence, setMinConfidence] = useState(0)
  const [minQuality, setMinQuality] = useState(0)
  const [qualityField, setQualityField] = useState<string | null>(null)
  const [optimizeFor, setOptimizeFor] = useState<OptimizeFor>('cost')
  const [decision, setDecision] = useState<Decision | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [copied, setCopied] = useState(false)
  const [pools, setPools] = useState<Record<string, string[]>>({})
  const [task, setTask] = useState('')
  const [scope, setScope] = useState<'all' | 'pool' | 'custom'>('all')
  const [customIds, setCustomIds] = useState<Set<string>>(() => new Set())
  const [customPickerOpen, setCustomPickerOpen] = useState(false)

  const COMMON_TASKS = [
    'video_processing',
    'v3_single',
    'headline_similarity',
    'spell_batch',
    'playout_observation',
    'qc_corrections',
  ]

  const [createOpen, setCreateOpen] = useState(false)
  const [pickerOpen, setPickerOpen] = useState(false)
  const [newTaskName, setNewTaskName] = useState('')
  const [pickedIds, setPickedIds] = useState<Set<string>>(() => new Set())
  const [createError, setCreateError] = useState('')
  const [savingTask, setSavingTask] = useState(false)

  const [catalogProviders, setCatalogProviders] = useState<string[]>([])
  const providers = (
    catalogProviders.length
      ? catalogProviders
      : tiers
        ? Array.from(new Set(tiers.tiers.flatMap((t) => Object.keys(t.models))))
        : []
  ).sort()
  const taskNames = Object.keys(pools).sort()
  const poolCount = task && pools[task] ? pools[task].length : null
  const providerSuffix = provider ? ` · ${provider} only` : ''
  const scopeSummary =
    scope === 'pool'
      ? poolCount != null
        ? `${poolCount} models in “${task}”${providerSuffix}`
        : 'Select a task pool'
      : scope === 'custom'
        ? `${customIds.size} picked model(s)${providerSuffix}`
        : `Whole catalog${providerSuffix}`

  useEffect(() => {
    void (async () => {
      try {
        const [quality, taskPools, catalog] = await Promise.all([
          api.quality(),
          api.taskPools(),
          api.catalog().catch(() => ({ catalog: [] as CatalogEntry[] })),
        ])
        setQualityField(quality.field)
        setPools(taskPools.pools)
        const fromCatalog = Array.from(
          new Set(catalog.catalog.map((e) => e.provider).filter(Boolean)),
        )
        setCatalogProviders(fromCatalog)
      } catch {
        // Optional surfaces — leave disabled if the calls fail.
      }
    })()
  }, [])

  function resetInputs() {
    setMetadata({})
    setText('')
    setImage(null)
    setMediaKind(null)
    setMediaSourceName(null)
    setProvider('')
    setMinConfidence(0)
    setMinQuality(0)
    setOptimizeFor('cost')
    setTask('')
    setScope('all')
    setCustomIds(new Set())
    setDecision(null)
    setError('')
  }

  function onMediaChange(
    file: File | null,
    meta?: { kind: 'image' | 'video'; sourceName: string } | null,
  ) {
    setImage(file)
    if (!file || !meta) {
      setMediaKind(null)
      setMediaSourceName(null)
      setMetadata((prev) => {
        const next = { ...prev }
        delete next.modality
        return next
      })
      return
    }
    setMediaKind(meta.kind)
    setMediaSourceName(meta.sourceName)
    setMetadata((prev) => ({
      ...prev,
      modality: meta.kind === 'video' ? 'video' : 'image',
    }))
  }

  function openCreateTask() {
    setNewTaskName('')
    setPickedIds(new Set())
    setCreateError('')
    setCreateOpen(true)
  }

  async function saveTask() {
    const name = newTaskName.trim()
    if (!name) {
      setCreateError('Enter a task name.')
      return
    }
    if (pickedIds.size === 0) {
      setCreateError('Select at least one model.')
      return
    }
    setSavingTask(true)
    setCreateError('')
    try {
      const res = await api.saveTaskPool(name, Array.from(pickedIds))
      setPools(res.pools)
      setTask(name)
      setScope('pool')
      setCreateOpen(false)
      setPickerOpen(false)
    } catch (err) {
      setCreateError(describeError(err))
    } finally {
      setSavingTask(false)
    }
  }

  async function run() {
    setLoading(true)
    setError('')
    setDecision(null)
    setCopied(false)
    try {
      const usePool = !catalogConfigured || scope === 'pool'
      const useCustom = catalogConfigured && scope === 'custom'
      if (catalogConfigured && scope === 'pool' && !task) throw new Error('Choose a task pool.')
      if (useCustom && customIds.size === 0) throw new Error('Select at least one model.')
      const meta = { ...metadata }
      if (usePool && task) meta.task = task
      setDecision(
        await api.resolve({
          metadata: meta,
          text,
          image,
          provider: provider || undefined,
          minConfidence,
          task: usePool && task ? task : undefined,
          selectedModels: useCustom ? Array.from(customIds) : null,
          minQuality: qualityField ? minQuality : undefined,
          optimizeFor,
        }),
      )
      onResolved()
    } catch (err) {
      setError(describeError(err))
    } finally {
      setLoading(false)
    }
  }

  async function copyDecision() {
    if (!decision) return
    try {
      await navigator.clipboard.writeText(JSON.stringify(decision, null, 2))
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1200)
    } catch {
      // Clipboard access can be denied; the decision is still visible on screen.
    }
  }

  return (
    <>
      <PageHead
        title="Playground"
        lede={
          catalogConfigured
            ? 'Content decides the tier, then the best model is chosen from your scope: whole catalog, a task pool, hand-picked models — optionally one provider. No LLM call.'
            : 'Rules → classifier → fallback. Attach a thumbnail or video clip (still extracted locally).'
        }
        actions={
          <Button startIcon={<ArrowCounterClockwise size={14} />} onClick={resetInputs}>
            Reset
          </Button>
        }
      />

      <Stack spacing={1.5}>
        <Panel title="Request">
          <Stack spacing={2.25}>
            <Box
              sx={{
                display: 'grid',
                gridTemplateColumns: { xs: '1fr', md: 'auto minmax(0, 1fr) auto' },
                gap: 1.5,
                alignItems: 'center',
              }}
            >
              {catalogConfigured ? (
                <ToggleButtonGroup
                  size="small"
                  exclusive
                  value={scope}
                  onChange={(_, v: 'all' | 'pool' | 'custom' | null) => {
                    if (v) setScope(v)
                  }}
                  aria-label="Candidate models"
                  sx={{
                    height: 40,
                    '& .MuiToggleButton-root': { fontSize: 12.5, textTransform: 'none', px: 1.5, fontWeight: 600, lineHeight: 1 },
                  }}
                >
                  <ToggleButton value="all">All models</ToggleButton>
                  <ToggleButton value="pool">Task pool</ToggleButton>
                  <ToggleButton value="custom">Pick models</ToggleButton>
                </ToggleButtonGroup>
              ) : (
                <Box />
              )}
              <TextField
                select
                size="small"
                label="Provider"
                value={provider}
                onChange={(e) => setProvider(e.target.value)}
                fullWidth
              >
                <MenuItem value="">Any provider</MenuItem>
                {providers.map((p) => (
                  <MenuItem key={p} value={p}>
                    {p}
                  </MenuItem>
                ))}
              </TextField>
              {catalogConfigured && (
                <ToggleButtonGroup
                  size="small"
                  exclusive
                  value={optimizeFor}
                  onChange={(_, v: OptimizeFor | null) => {
                    if (v) setOptimizeFor(v)
                  }}
                  aria-label="Optimise for"
                  sx={{
                    height: 40,
                    '& .MuiToggleButton-root': {
                      fontSize: 12.5,
                      textTransform: 'none',
                      px: 1.5,
                      fontWeight: 600,
                      lineHeight: 1,
                    },
                  }}
                >
                  <ToggleButton value="cost">Cost</ToggleButton>
                  <ToggleButton value="balanced">Balanced</ToggleButton>
                  <ToggleButton value="accuracy">Accuracy</ToggleButton>
                </ToggleButtonGroup>
              )}
            </Box>

            {(!catalogConfigured || scope === 'pool') && (
              <Stack direction="row" spacing={1.5} sx={{ alignItems: 'center' }}>
                <TextField
                  select
                  size="small"
                  label={catalogConfigured ? 'Task pool' : 'Task (rules)'}
                  value={task}
                  onChange={(e) => setTask(e.target.value)}
                  fullWidth
                >
                  <MenuItem value="">
                    <em>{catalogConfigured ? 'Select a task pool…' : 'None — classifier / fallback'}</em>
                  </MenuItem>
                  {(catalogConfigured ? taskNames : Array.from(new Set([...COMMON_TASKS, ...taskNames]))).map((name) => (
                    <MenuItem key={name} value={name}>
                      {name}
                      {catalogConfigured ? ` (${pools[name]?.length ?? 0})` : ''}
                    </MenuItem>
                  ))}
                </TextField>
                {catalogConfigured && (
                  <Button
                    variant="outlined"
                    startIcon={<Plus size={16} />}
                    onClick={openCreateTask}
                    sx={{ height: 40, px: 1.75, whiteSpace: 'nowrap', flexShrink: 0 }}
                  >
                    Create task
                  </Button>
                )}
              </Stack>
            )}

            {catalogConfigured && scope === 'custom' && (
              <Stack direction="row" spacing={1.5} sx={{ alignItems: 'center' }}>
                <Button variant="outlined" onClick={() => setCustomPickerOpen(true)} sx={{ height: 40, whiteSpace: 'nowrap' }}>
                  Select models{customIds.size > 0 ? ` (${customIds.size})` : ''}
                </Button>
                <Typography sx={{ fontSize: 12.5, color: PALETTE.textMuted, minWidth: 0 }} noWrap>
                  {customIds.size === 0
                    ? 'Router will only choose among the models you tick.'
                    : Array.from(customIds).slice(0, 4).join(', ') + (customIds.size > 4 ? ` +${customIds.size - 4} more` : '')}
                </Typography>
              </Stack>
            )}

            <TextField
              value={text}
              onChange={(e) => setText(e.target.value)}
              multiline
              rows={3}
              size="small"
              fullWidth
              label="Request text"
              placeholder="e.g. channel=Aaj Tak; genre=News — or paste segment text"
            />

            <Box
              sx={{
                display: 'grid',
                gridTemplateColumns: { xs: '1fr', md: '1fr auto' },
                gap: 1.5,
                alignItems: 'start',
              }}
            >
              <MetadataFieldsEditor value={metadata} onChange={setMetadata} size="small" />
              <ImagePicker
                value={image}
                onChange={onMediaChange}
                sourceKind={mediaKind}
                sourceName={mediaSourceName}
                hint="Image or video — video → one still in-browser"
              />
            </Box>

            <Stack
              direction={{ xs: 'column', sm: 'row' }}
              spacing={1.5}
              sx={{ alignItems: { sm: 'center' }, justifyContent: 'space-between' }}
            >
              <Stack direction="row" spacing={2} sx={{ flex: 1, alignItems: 'center', flexWrap: 'wrap', minHeight: 40 }}>
                <Stack direction="row" spacing={1} sx={{ alignItems: 'center', minWidth: 200, flex: 1 }}>
                  <Typography sx={{ fontSize: 12, color: PALETTE.textMuted, whiteSpace: 'nowrap' }}>
                    Min conf {minConfidence.toFixed(2)}
                  </Typography>
                  <Slider
                    size="small"
                    value={minConfidence}
                    onChange={(_, v) => setMinConfidence(v as number)}
                    min={0}
                    max={1}
                    step={0.05}
                    aria-label="Minimum confidence"
                  />
                </Stack>
                {qualityField && catalogConfigured && (
                  <Stack direction="row" spacing={1} sx={{ alignItems: 'center', minWidth: 200, flex: 1 }}>
                    <Typography sx={{ fontSize: 12, color: PALETTE.textMuted, whiteSpace: 'nowrap' }}>
                      Quality {minQuality.toFixed(1)}
                    </Typography>
                    <Slider
                      size="small"
                      value={minQuality}
                      onChange={(_, v) => setMinQuality(v as number)}
                      min={0}
                      max={5}
                      step={0.5}
                      aria-label="Minimum quality"
                    />
                  </Stack>
                )}
                <Typography sx={{ fontSize: 12.5, color: PALETTE.textMuted }}>
                  {catalogConfigured
                    ? scopeSummary
                    : mediaKind === 'video'
                      ? 'Ladder · still from video'
                      : image
                        ? 'Ladder · thumbnail attached'
                        : 'Ladder · rules → classifier → fallback'}
                </Typography>
              </Stack>
              <Button
                variant="contained"
                onClick={() => void run()}
                disabled={
                  loading ||
                  (catalogConfigured && scope === 'pool' && !task) ||
                  (catalogConfigured && scope === 'custom' && customIds.size === 0)
                }
                endIcon={<ArrowRight size={16} />}
                sx={{ flexShrink: 0, height: 40, px: 2.25, textTransform: 'none' }}
              >
                {loading ? 'Picking…' : 'Pick best model'}
              </Button>
            </Stack>
          </Stack>
        </Panel>

        <Panel title="Decision" note={decision ? undefined : 'Router only — no Gemini call.'}>
          {error && (
            <Alert severity="error" sx={{ mb: decision ? 2 : 0 }}>
              {error}
            </Alert>
          )}
          {!decision && !error ? (
            <Typography sx={{ fontSize: 13, color: PALETTE.textMuted, py: 1 }}>
              {catalogConfigured
                ? 'Add text, metadata or media, choose the model scope, then pick.'
                : 'Set a task (e.g. video_processing), optional clip/thumbnail, then pick.'}
            </Typography>
          ) : null}
          {decision ? (
            <Stack spacing={1.75}>
              <Box
                sx={{
                  p: 2,
                  borderRadius: '10px',
                  bgcolor: TONE_SOFT[tierTone(decision.tier)],
                  border: `1px solid ${PALETTE.hairline}`,
                }}
              >
                <Typography sx={{ fontSize: 11, fontWeight: 650, letterSpacing: '0.04em', textTransform: 'uppercase', color: PALETTE.textMuted }}>
                  Best model
                </Typography>
                <Typography sx={{ fontSize: 22, fontWeight: 700, letterSpacing: '-0.02em', mt: 0.5 }}>
                  <Mono size={22}>{decision.model_name}</Mono>
                </Typography>
                <Stack direction="row" spacing={1} sx={{ mt: 1, alignItems: 'center', flexWrap: 'wrap' }}>
                  <StateTag label={decision.tier} tone={tierTone(decision.tier)} size="small" />
                  <StateTag label={decision.optimize_for} tone="default" size="small" />
                  {decision.confidence != null && (
                    <StateTag label={`conf ${decision.confidence.toFixed(2)}`} tone="default" size="small" />
                  )}
                </Stack>
              </Box>

              <Box>
                <Typography sx={{ fontSize: 11, fontWeight: 650, letterSpacing: '0.04em', textTransform: 'uppercase', color: PALETTE.textMuted, mb: 0.5 }}>
                  Why
                </Typography>
                <Typography sx={{ fontSize: 12.5, color: PALETTE.textSecondary }}>{decision.reason}</Typography>
              </Box>

              <Button variant="outlined" startIcon={<Copy size={14} />} onClick={() => void copyDecision()} sx={{ alignSelf: 'flex-start' }}>
                {copied ? 'Copied' : 'Copy decision'}
              </Button>
            </Stack>
          ) : null}
        </Panel>
      </Stack>

      <Dialog open={createOpen} onClose={() => !savingTask && setCreateOpen(false)} fullWidth maxWidth="sm">
        <DialogTitle sx={{ fontSize: 17, fontWeight: 700 }}>Create task</DialogTitle>
        <DialogContent>
          <Stack spacing={2} sx={{ pt: 0.5 }}>
            {createError && <Alert severity="error">{createError}</Alert>}
            <TextField
              autoFocus
              size="small"
              label="Task name"
              placeholder="e.g. video_processing"
              value={newTaskName}
              onChange={(e) => setNewTaskName(e.target.value)}
              fullWidth
            />
            <Stack direction="row" spacing={1.25} sx={{ alignItems: 'center' }}>
              <Button variant="outlined" onClick={() => setPickerOpen(true)}>
                Select models{pickedIds.size > 0 ? ` (${pickedIds.size})` : ''}
              </Button>
              <Typography sx={{ fontSize: 12.5, color: PALETTE.textMuted }}>
                {pickedIds.size === 0 ? 'No models selected yet' : `${pickedIds.size} model(s) in this task`}
              </Typography>
            </Stack>
          </Stack>
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button onClick={() => setCreateOpen(false)} disabled={savingTask}>
            Cancel
          </Button>
          <Button variant="contained" onClick={() => void saveTask()} disabled={savingTask || !newTaskName.trim() || pickedIds.size === 0}>
            {savingTask ? 'Saving…' : 'Save task'}
          </Button>
        </DialogActions>
      </Dialog>

      <ModelPickerDialog
        open={pickerOpen}
        selected={pickedIds}
        onChange={setPickedIds}
        onClose={() => setPickerOpen(false)}
      />
      <ModelPickerDialog
        open={customPickerOpen}
        selected={customIds}
        onChange={setCustomIds}
        onClose={() => setCustomPickerOpen(false)}
      />
    </>
  )
}

/** Nested modal: searchable catalog checklist for Create task. */
function ModelPickerDialog({
  open,
  selected,
  onChange,
  onClose,
}: {
  open: boolean
  selected: Set<string>
  onChange: (next: Set<string>) => void
  onClose: () => void
}) {
  const [entries, setEntries] = useState<CatalogEntry[] | null>(null)
  const [error, setError] = useState('')
  const [query, setQuery] = useState('')
  const [provider, setProvider] = useState('')
  const [page, setPage] = useState(0)
  const [rowsPerPage, setRowsPerPage] = useState(25)

  useEffect(() => {
    if (!open) return
    void (async () => {
      try {
        setEntries((await api.catalog()).catalog)
        setError('')
      } catch (err) {
        setError(describeError(err))
        setEntries([])
      }
    })()
  }, [open])

  useEffect(() => {
    setPage(0)
  }, [query, provider])

  const providers = useMemo(() => Array.from(new Set((entries ?? []).map((e) => e.provider))).sort(), [entries])
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    return (entries ?? []).filter((e) => {
      if (provider && e.provider !== provider) return false
      if (!q) return true
      return e.label.toLowerCase().includes(q) || e.id.toLowerCase().includes(q)
    })
  }, [entries, query, provider])

  const pageRows = filtered.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)

  function toggle(id: string) {
    const next = new Set(selected)
    if (next.has(id)) next.delete(id)
    else next.add(id)
    onChange(next)
  }

  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="md">
      <DialogTitle sx={{ fontSize: 17, fontWeight: 700 }}>Select models</DialogTitle>
      <DialogContent dividers>
        {error && (
          <Alert severity="error" sx={{ mb: 1.5 }}>
            {error}
          </Alert>
        )}
        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} sx={{ mb: 1.5 }}>
          <TextField
            size="small"
            fullWidth
            placeholder="Search model or id…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            slotProps={{
              input: {
                startAdornment: (
                  <InputAdornment position="start">
                    <MagnifyingGlass size={14} />
                  </InputAdornment>
                ),
              },
            }}
          />
          <TextField select size="small" label="Provider" value={provider} onChange={(e) => setProvider(e.target.value)} sx={{ minWidth: 160 }}>
            <MenuItem value="">All</MenuItem>
            {providers.map((p) => (
              <MenuItem key={p} value={p}>
                {p}
              </MenuItem>
            ))}
          </TextField>
          <Button size="small" variant="outlined" onClick={() => onChange(new Set(filtered.map((e) => e.id)))} disabled={filtered.length === 0}>
            Check filtered
          </Button>
          <Button size="small" onClick={() => onChange(new Set())} disabled={selected.size === 0}>
            Clear
          </Button>
        </Stack>
        <Typography sx={{ fontSize: 12, color: PALETTE.textMuted, mb: 1 }}>
          {selected.size} selected · {filtered.length} shown
        </Typography>
        <Box sx={{ maxHeight: 420, overflow: 'auto', border: `1px solid ${PALETTE.hairline}`, borderRadius: '8px' }}>
          <Table size="small" stickyHeader>
            <TableHead>
              <TableRow>
                <TableCell padding="checkbox" />
                <TableCell>Model</TableCell>
                <TableCell>ID</TableCell>
                <TableCell>Provider</TableCell>
                <TableCell>Label</TableCell>
                <TableCell align="right">Cost</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {entries === null && (
                <TableRow>
                  <TableCell colSpan={6}>
                    <Typography sx={{ fontSize: 13, color: PALETTE.textMuted, p: 1 }}>Loading catalog…</Typography>
                  </TableCell>
                </TableRow>
              )}
              {pageRows.map((entry) => {
                const checked = selected.has(entry.id)
                return (
                  <TableRow key={entry.id} hover selected={checked} onClick={() => toggle(entry.id)} sx={{ cursor: 'pointer' }}>
                    <TableCell padding="checkbox">
                      <Checkbox size="small" checked={checked} slotProps={{ input: { 'aria-label': `Select ${entry.id}` } }} />
                    </TableCell>
                    <TableCell>
                      <Mono size={12}>{entry.label}</Mono>
                    </TableCell>
                    <TableCell>
                      <Mono size={11} muted>
                        {entry.id}
                      </Mono>
                    </TableCell>
                    <TableCell>{entry.provider}</TableCell>
                    <TableCell>
                      <StateTag label={entry.tier} tone={tierTone(entry.tier)} size="small" />
                    </TableCell>
                    <TableCell align="right">
                      <Mono size={12}>{entry.cost}</Mono>
                    </TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        </Box>
        <TablePagination
          component="div"
          count={filtered.length}
          page={page}
          onPageChange={(_, next) => setPage(next)}
          rowsPerPage={rowsPerPage}
          onRowsPerPageChange={(e) => {
            setRowsPerPage(parseInt(e.target.value, 10))
            setPage(0)
          }}
          rowsPerPageOptions={[25, 50, 100]}
        />
      </DialogContent>
      <DialogActions sx={{ px: 3, pb: 2 }}>
        <Button variant="contained" onClick={onClose}>
          Done ({selected.size})
        </Button>
      </DialogActions>
    </Dialog>
  )
}
