import { DotsThreeVertical, FloppyDisk, MagnifyingGlass, Plus, Trash } from '@phosphor-icons/react'
import {
  Alert,
  Box,
  Button,
  IconButton,
  InputAdornment,
  MenuItem,
  Skeleton,
  Stack,
  Switch,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from '@mui/material'
import { useEffect, useMemo, useState } from 'react'
import { api, describeError } from '../api'
import { MetadataFieldsEditor } from '../components/MetadataFieldsEditor'
import { InfoBanner, Mono, PageHead, Panel, SectionHead, StateTag, StepFlow } from '../components/Primitives'
import { PALETTE, tierTone } from '../theme'
import type { RuleEntry } from '../types'

function describeWhen(when: Record<string, string | string[]>): string {
  return Object.entries(when)
    .map(([k, v]) => `${k}=${Array.isArray(v) ? v.join('|') : v}`)
    .join(' and ')
}

/** Splits a comma-separated "Value" into an OR-match list, or keeps a single value as a string. */
function toWhenValue(raw: string): string | string[] {
  const parts = raw
    .split(',')
    .map((p) => p.trim())
    .filter(Boolean)
  return parts.length > 1 ? parts : raw.trim()
}

function fromWhenValue(value: string | string[]): string {
  return Array.isArray(value) ? value.join(', ') : value
}

/**
 * Master-detail: the table on the left is every stored rule; the panel on the right edits
 * whichever one is selected, or builds a new one when nothing is. "Rules always run first" —
 * the most specific match wins, priority breaks a tie between equally specific rules.
 */
export function Rules({ tierNames }: { tierNames: string[] }) {
  const [rules, setRules] = useState<RuleEntry[] | null>(null)
  const [editingIndex, setEditingIndex] = useState<number | null>(null)
  const [search, setSearch] = useState('')
  const [tierFilter, setTierFilter] = useState<string>('')
  const [when, setWhen] = useState<Record<string, string>>({})
  const [tier, setTier] = useState('')
  const [priority, setPriority] = useState('0')
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)

  async function load() {
    try {
      setRules((await api.rules()).rules)
    } catch (err) {
      setError(describeError(err))
    }
  }

  useEffect(() => {
    void load()
  }, [])

  function resetForm() {
    setEditingIndex(null)
    setWhen({})
    setTier('')
    setPriority('0')
  }

  function selectRule(index: number) {
    const rule = rules?.[index]
    if (!rule) return
    setEditingIndex(index)
    setWhen(Object.fromEntries(Object.entries(rule.when).map(([k, v]) => [k, fromWhenValue(v)])))
    setTier(rule.tier)
    setPriority(String(rule.priority ?? 0))
  }

  async function save(next: RuleEntry[]) {
    setSaving(true)
    setError('')
    try {
      setRules((await api.saveRules(next)).rules)
    } catch (err) {
      setError(describeError(err))
    } finally {
      setSaving(false)
    }
  }

  async function saveForm() {
    if (Object.keys(when).length === 0 || !tier) return
    const whenWithOr = Object.fromEntries(Object.entries(when).map(([k, v]) => [k, toWhenValue(v)]))
    const parsedPriority = Number.parseInt(priority, 10) || 0
    const entry: RuleEntry = { when: whenWithOr, tier, priority: parsedPriority }
    const current = rules ?? []
    const next = editingIndex === null ? [...current, entry] : current.map((r, i) => (i === editingIndex ? entry : r))
    await save(next)
    resetForm()
  }

  async function removeRule(index: number) {
    if (!rules) return
    await save(rules.filter((_, i) => i !== index))
    if (editingIndex === index) resetForm()
  }

  const filtered = useMemo(() => {
    const term = search.trim().toLowerCase()
    return (rules ?? [])
      .map((rule, index) => ({ rule, index }))
      .filter(({ rule }) => (tierFilter ? rule.tier === tierFilter : true))
      .filter(({ rule }) => (term ? describeWhen(rule.when).toLowerCase().includes(term) : true))
  }, [rules, search, tierFilter])

  return (
    <>
      <PageHead
        title="Routing rules"
        lede="Define deterministic routing before the classifier takes over."
        actions={
          <Button
            variant="contained"
            startIcon={<Plus size={14} weight="bold" />}
            onClick={resetForm}
          >
            Create rule
          </Button>
        }
      />
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Stack spacing={2}>
        <InfoBanner
          tone="brand"
          badge="01"
          title="Rules always run first"
          note="Ordered by priority. The first matching rule determines the tier; unmatched inputs continue to the classifier."
        />

        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1.4fr 1fr' }, gap: 2, alignItems: 'start' }}>
          <Panel
            title={rules === null ? 'Loading…' : `Routing rules · ${filtered.length} of ${rules.length}`}
            dense
            actions={
              <Stack direction="row" spacing={1}>
                <TextField
                  size="small"
                  placeholder="Search by rule name or metadata field…"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  sx={{ minWidth: 240 }}
                  slotProps={{ input: { startAdornment: <InputAdornment position="start"><MagnifyingGlass size={14} /></InputAdornment> } }}
                />
                <TextField
                  select
                  size="small"
                  value={tierFilter}
                  onChange={(e) => setTierFilter(e.target.value)}
                  sx={{ minWidth: 140 }}
                >
                  <MenuItem value="">All tiers</MenuItem>
                  {tierNames.map((name) => (
                    <MenuItem key={name} value={name}>{name}</MenuItem>
                  ))}
                </TextField>
              </Stack>
            }
          >
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>#</TableCell>
                  <TableCell>Rule / condition</TableCell>
                  <TableCell>Target tier</TableCell>
                  <TableCell align="right">Priority</TableCell>
                  <TableCell align="center">Status</TableCell>
                  <TableCell align="right" />
                </TableRow>
              </TableHead>
              <TableBody>
                {rules === null &&
                  [0, 1, 2, 3].map((i) => (
                    <TableRow key={i}>
                      <TableCell colSpan={6}>
                        <Skeleton variant="text" height={28} />
                      </TableCell>
                    </TableRow>
                  ))}
                {rules !== null && filtered.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={6}>
                      <Typography sx={{ fontSize: 13, color: PALETTE.textMuted, p: 1 }}>
                        {rules.length === 0 ? 'No rules yet — add one on the right to steer traffic deterministically.' : 'No rules match this filter.'}
                      </Typography>
                    </TableCell>
                  </TableRow>
                )}
                {filtered.map(({ rule, index }, i) => (
                  <TableRow
                    key={JSON.stringify(rule.when) + index}
                    hover
                    onClick={() => selectRule(index)}
                    sx={{ cursor: 'pointer', bgcolor: editingIndex === index ? PALETTE.brandSoft : undefined }}
                  >
                    <TableCell sx={{ width: 32 }}>
                      <Mono size={11} muted>{String(i + 1).padStart(2, '0')}</Mono>
                    </TableCell>
                    <TableCell>
                      <Stack direction="row" spacing={0.5} sx={{ flexWrap: 'wrap' }}>
                        {Object.entries(rule.when).map(([k, v]) => (
                          <Mono key={k} size={12.5}>
                            {k}={Array.isArray(v) ? v.join('|') : v}
                          </Mono>
                        ))}
                      </Stack>
                    </TableCell>
                    <TableCell>
                      <StateTag label={rule.tier} tone={tierTone(rule.tier)} size="small" />
                    </TableCell>
                    <TableCell align="right">
                      <Mono size={12} muted={!rule.priority}>
                        {rule.priority ?? 0}
                      </Mono>
                    </TableCell>
                    <TableCell align="center">
                      <Switch checked onClick={(e) => e.stopPropagation()} disabled />
                    </TableCell>
                    <TableCell align="right" sx={{ width: 40 }}>
                      <IconButton
                        size="small"
                        aria-label={`Remove rule ${describeWhen(rule.when)}`}
                        onClick={(e) => {
                          e.stopPropagation()
                          void removeRule(index)
                        }}
                        disabled={saving}
                      >
                        <DotsThreeVertical size={16} />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            <Box sx={{ px: 2.5, py: 1.25, borderTop: `1px solid ${PALETTE.hairline}`, display: 'flex', justifyContent: 'space-between' }}>
              <Typography sx={{ fontSize: 11.5, color: PALETTE.textMuted }}>Priority is evaluated from top to bottom.</Typography>
              <Typography sx={{ fontSize: 11.5, color: PALETTE.textMuted }}>
                {rules?.length ?? 0} {rules?.length === 1 ? 'rule' : 'rules'}
              </Typography>
            </Box>
          </Panel>

          <Panel
            title={editingIndex === null ? 'Rule details · new rule' : `Rule details · priority ${priority || '0'}`}
            note={editingIndex === null ? 'Add a new deterministic rule — matches bypass the classifier.' : 'First match wins. Rule matches bypass the classifier.'}
            actions={<StateTag label={editingIndex === null ? 'Draft' : 'Enabled'} tone={editingIndex === null ? 'default' : 'good'} size="small" />}
          >
            <Stack spacing={1.75}>
              <TextField
                label="Rule name"
                value={tier ? `${tier.charAt(0).toUpperCase() + tier.slice(1)} routing` : ''}
                onChange={() => { /* label is derived from tier; kept visible for parity with Figma */ }}
                fullWidth
                slotProps={{ input: { readOnly: true } }}
                helperText="Derived from the target tier."
              />
              <SectionHead
                title="When all conditions match"
                note="Comma-separate a value for an OR match, e.g. qc_review, audit."
              />
              <MetadataFieldsEditor value={when} onChange={setWhen} />
              <TextField
                select
                label="Route to tier"
                value={tier}
                onChange={(e) => setTier(e.target.value)}
                fullWidth
              >
                {tierNames.map((name) => (
                  <MenuItem key={name} value={name}>
                    {name}
                  </MenuItem>
                ))}
              </TextField>
              <TextField
                label="Priority"
                type="number"
                value={priority}
                onChange={(e) => setPriority(e.target.value)}
                sx={{ maxWidth: 160 }}
                slotProps={{ htmlInput: { min: 0 } }}
                helperText="Breaks a tie between equally specific rules. Default 0."
              />
              <Typography sx={{ fontSize: 12, color: PALETTE.textMuted }}>
                Rule matches bypass the classifier.
              </Typography>
              <Stack direction="row" spacing={1} sx={{ pt: 1 }}>
                {editingIndex !== null && (
                  <Button onClick={resetForm} disabled={saving} startIcon={<Trash size={14} />}>
                    Discard
                  </Button>
                )}
                <Button
                  variant="contained"
                  onClick={() => void saveForm()}
                  startIcon={<FloppyDisk size={14} />}
                  disabled={saving || Object.keys(when).length === 0 || !tier}
                >
                  {editingIndex === null ? 'Add rule' : 'Save rule'}
                </Button>
              </Stack>
            </Stack>
          </Panel>
        </Box>

        <Panel title="Predictable routing, by design" note="No matching rule? The feature builder encodes metadata and text before tier classification." dense>
          <Box sx={{ p: 2.25 }}>
            <StepFlow
              steps={[
                { label: 'Rule match', note: 'Operator-defined overrides.', active: editingIndex !== null },
                { label: 'Classifier prediction', note: 'Learned tier for metadata + text.', active: false },
                { label: 'Configured fallback', note: 'Used only when neither rule nor classifier decide.', active: false },
              ]}
              dense
            />
          </Box>
        </Panel>
      </Stack>
    </>
  )
}
