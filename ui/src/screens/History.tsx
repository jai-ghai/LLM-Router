import { ArrowClockwise, Copy, Image as ImageIcon, MagnifyingGlass, Scales, Star } from '@phosphor-icons/react'
import {
  Alert,
  Autocomplete,
  Box,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  InputAdornment,
  MenuItem,
  Rating,
  Skeleton,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from '@mui/material'
import { useEffect, useMemo, useState } from 'react'
import { api, describeError } from '../api'
import { useUser } from '../auth'
import { InfoBanner, Mono, PageHead, Panel, ResolutionPath, StateTag } from '../components/Primitives'
import { PALETTE, tierTone } from '../theme'
import type { HistoryEntry, PreferenceOutcome, Skill } from '../types'
import { SKILL_FAULTS, SKILL_LABELS, SkillPicker } from '../components/SkillPicker'

function averageRating(entry: HistoryEntry): number | null {
  const ratings = entry.ratings ?? []
  return ratings.length ? ratings.reduce((sum, r) => sum + r.rating, 0) / ratings.length : null
}

/**
 * Every real resolve() call made from this console, newest first — a master-detail layout:
 * the table is the list, the right panel is the full trace for whichever row is selected.
 * Rating a row's output is how this project's only "accuracy" signal gets created (see
 * quality.py) — a model nobody has rated is never penalized at resolve time.
 *
 * The Rate modal no longer asks for a name — judged_by is the session
 * user (P0). When GRADIST_QUALITY_FIELD is configured, the modal shows that field's
 * value from the row's metadata (read-only; Q5.5.b = no re-tag at rating time).
 */
export function History({ refreshKey }: { refreshKey: number }) {
  const user = useUser()
  const [entries, setEntries] = useState<HistoryEntry[] | null>(null)
  const [error, setError] = useState('')
  const [selected, setSelected] = useState<HistoryEntry | null>(null)
  const [rateOpen, setRateOpen] = useState(false)
  const [rating, setRating] = useState<number | null>(null)
  const [wrong, setWrong] = useState<Skill[]>([])
  const [notes, setNotes] = useState('')
  const [compareOpen, setCompareOpen] = useState(false)
  const [otherModel, setOtherModel] = useState('')
  const [outcome, setOutcome] = useState<PreferenceOutcome | null>(null)
  const [compareSkills, setCompareSkills] = useState<Skill[]>([])
  const [rateError, setRateError] = useState('')
  const [saving, setSaving] = useState(false)
  const [qualityField, setQualityField] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [tierFilter, setTierFilter] = useState('')
  const [pathFilter, setPathFilter] = useState('')
  const [copied, setCopied] = useState(false)
  const isAdmin = user?.role === 'admin'
  const [ownerFilter, setOwnerFilter] = useState('')
  const [operators, setOperators] = useState<string[]>([])

  async function load() {
    try {
      const rows = (await api.history(100, undefined, ownerFilter || undefined)).history
      setEntries(rows)
      setSelected((prev) => rows.find((r) => prev && r.timestamp === prev.timestamp) ?? rows[0] ?? null)
    } catch (err) {
      setError(describeError(err))
      setEntries([])
    }
  }

  useEffect(() => {
    void load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refreshKey, ownerFilter])

  useEffect(() => {
    if (!isAdmin) return
    void (async () => {
      try {
        setOperators((await api.listUsers()).users.map((u) => u.username).sort())
      } catch {
        // Without the user list the table still shows who made each request.
      }
    })()
  }, [isAdmin])

  useEffect(() => {
    void (async () => {
      try {
        setQualityField((await api.qualityField()).field)
      } catch {
        // Quality field is optional — leave hidden if the endpoint is unreachable.
        setQualityField(null)
      }
    })()
  }, [])

  function openRate() {
    setRating(null)
    setWrong([])
    setNotes('')
    setRateError('')
    setRateOpen(true)
  }

  function openCompare() {
    setOtherModel(selected?.shortlist?.find((id) => id !== selected.model_name) ?? '')
    setOutcome(null)
    setCompareSkills([])
    setNotes('')
    setRateError('')
    setCompareOpen(true)
  }

  async function submitCompare() {
    if (!selected?.request_id || !otherModel || !outcome) return
    setSaving(true)
    setRateError('')
    try {
      await api.comparePick({ requestId: selected.request_id, otherModel, outcome, skills: compareSkills, notes })
      setCompareOpen(false)
      await load()
    } catch (err) {
      setRateError(describeError(err))
    } finally {
      setSaving(false)
    }
  }

  async function submitRating() {
    if (!selected || !rating) return
    setSaving(true)
    setRateError('')
    try {
      await api.rateQuality({
        modelId: selected.model_name,
        metadata: selected.metadata,
        rating,
        notes,
        requestId: selected.request_id,
        wrongSkills: rating <= 3 ? wrong : [],
      })
      setRateOpen(false)
      await load()
    } catch (err) {
      setRateError(describeError(err))
    } finally {
      setSaving(false)
    }
  }

  const tiers = useMemo(() => Array.from(new Set((entries ?? []).map((e) => e.tier))), [entries])
  const filtered = useMemo(() => {
    const term = search.trim().toLowerCase()
    return (entries ?? []).filter((e) => {
      if (tierFilter && e.tier !== tierFilter) return false
      const path = e.rule_outcome !== 'no matching rule' ? 'Rule match' : e.fallback_outcome === 'not needed' ? 'Classifier' : 'Fallback'
      if (pathFilter && path !== pathFilter) return false
      if (term && !(e.model_name.toLowerCase().includes(term) || e.text.toLowerCase().includes(term) || e.reason.toLowerCase().includes(term))) return false
      return true
    })
  }, [entries, search, tierFilter, pathFilter])

  async function copyDecision() {
    if (!selected) return
    try {
      await navigator.clipboard.writeText(JSON.stringify(selected, null, 2))
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1200)
    } catch {
      // Clipboard denied — selection is still visible on screen.
    }
  }

  return (
    <>
      <PageHead
        title="Request logs"
        lede={
          isAdmin
            ? 'Inspect routing decisions, confidence, and resolution reasons — for every operator.'
            : 'Your routing decisions, confidence, and resolution reasons. Other operators cannot see these.'
        }
        actions={
          <Button startIcon={<ArrowClockwise size={14} />} onClick={() => void load()}>
            Refresh
          </Button>
        }
      />
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Stack spacing={2}>
        <Stack direction={{ xs: 'column', md: 'row' }} spacing={1}>
          <TextField
            size="small"
            placeholder="Search request text, model, or reason…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            sx={{ flex: 1, minWidth: 220 }}
            slotProps={{ input: { startAdornment: <InputAdornment position="start"><MagnifyingGlass size={14} /></InputAdornment> } }}
          />
          <TextField
            select
            size="small"
            value={tierFilter}
            onChange={(e) => setTierFilter(e.target.value)}
            sx={{ minWidth: 140 }}
            label="Tier"
          >
            <MenuItem value="">All tiers</MenuItem>
            {tiers.map((t) => (
              <MenuItem key={t} value={t}>{t}</MenuItem>
            ))}
          </TextField>
          <TextField
            select
            size="small"
            value={pathFilter}
            onChange={(e) => setPathFilter(e.target.value)}
            sx={{ minWidth: 140 }}
            label="Resolution path"
          >
            <MenuItem value="">All paths</MenuItem>
            <MenuItem value="Rule match">Rule match</MenuItem>
            <MenuItem value="Classifier">Classifier</MenuItem>
            <MenuItem value="Fallback">Fallback</MenuItem>
          </TextField>
          {isAdmin && (
            <TextField
              select
              size="small"
              value={ownerFilter}
              onChange={(e) => setOwnerFilter(e.target.value)}
              sx={{ minWidth: 160 }}
              label="Requested by"
            >
              <MenuItem value="">Everyone</MenuItem>
              {operators.map((name) => (
                <MenuItem key={name} value={name}>{name}</MenuItem>
              ))}
            </TextField>
          )}
        </Stack>

        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1.4fr 1fr' }, gap: 2, alignItems: 'start' }}>
          <Panel
            title={entries === null ? 'Loading…' : `Routing decisions · ${filtered.length} of ${entries.length}`}
            dense
          >
            {entries === null ? (
              <Stack spacing={1} sx={{ p: 2.5 }}>
                {[0, 1, 2].map((i) => (
                  <Skeleton key={i} variant="rectangular" height={32} sx={{ borderRadius: '8px' }} />
                ))}
              </Stack>
            ) : filtered.length === 0 ? (
              <Typography sx={{ fontSize: 13, color: PALETTE.textMuted, p: 2.5 }}>
                {entries.length === 0 ? 'Nothing resolved yet — use Playground to send a request.' : 'No rows match this filter.'}
              </Typography>
            ) : (
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Request / time</TableCell>
                    {isAdmin && <TableCell>Requested by</TableCell>}
                    <TableCell>Resolution</TableCell>
                    <TableCell>Tier</TableCell>
                    <TableCell align="right">Confidence</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {filtered.map((entry, i) => {
                    const isSelected = selected === entry
                    const resolution = entry.rule_outcome !== 'no matching rule' ? 'Rule match' : entry.fallback_outcome === 'not needed' ? 'Classifier' : 'Fallback'
                    return (
                      <TableRow
                        key={`${entry.timestamp}-${i}`}
                        hover
                        onClick={() => setSelected(entry)}
                        sx={{ cursor: 'pointer', bgcolor: isSelected ? PALETTE.brandSoft : undefined }}
                      >
                        <TableCell>
                          <Mono size={12.5}>{entry.model_name || '—'}</Mono>
                          <Typography sx={{ fontSize: 11, color: PALETTE.textMuted, display: 'flex', alignItems: 'center', gap: 0.4 }}>
                            {new Date(entry.timestamp).toLocaleTimeString()}
                            {entry.has_image && <ImageIcon size={12} color={PALETTE.textMuted} />}
                            {averageRating(entry) !== null && (
                              <Box component="span" sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.25, color: PALETTE.warningText }}>
                                <Star size={11} weight="fill" />
                                {averageRating(entry)!.toFixed(1)}
                              </Box>
                            )}
                          </Typography>
                        </TableCell>
                        {isAdmin && (
                          <TableCell>
                            <Typography sx={{ fontSize: 12.5, color: PALETTE.textSecondary }}>{entry.requested_by ?? '—'}</Typography>
                          </TableCell>
                        )}
                        <TableCell>
                          <Typography sx={{ fontSize: 12.5, color: PALETTE.textMuted }}>{resolution}</Typography>
                        </TableCell>
                        <TableCell>
                          <StateTag label={entry.tier} tone={tierTone(entry.tier)} size="small" />
                        </TableCell>
                        <TableCell align="right">
                          <Typography sx={{ fontSize: 12, color: PALETTE.textMuted, fontVariantNumeric: 'tabular-nums' }}>
                            {entry.confidence !== null ? `${(entry.confidence * 100).toFixed(1)}%` : '—'}
                          </Typography>
                        </TableCell>
                      </TableRow>
                    )
                  })}
                </TableBody>
              </Table>
            )}
          </Panel>

          <Panel
            title="Decision details"
            note={
              selected
                ? `${new Date(selected.timestamp).toLocaleString()}${isAdmin && selected.requested_by ? ` · by ${selected.requested_by}` : ''}`
                : undefined
            }
            actions={selected && (
              <StateTag
                label={selected.rule_outcome !== 'no matching rule' ? 'Rule' : selected.fallback_outcome === 'not needed' ? 'Classifier' : 'Fallback'}
                tone={selected.rule_outcome !== 'no matching rule' ? 'brand' : selected.fallback_outcome === 'not needed' ? 'good' : 'warning'}
                size="small"
              />
            )}
          >
            {!selected ? (
              <Typography sx={{ fontSize: 13, color: PALETTE.textMuted }}>Select a row to see its full trace.</Typography>
            ) : (
              <Stack spacing={1.75}>
                <Box>
                  <Typography sx={{ fontSize: 11, fontWeight: 650, letterSpacing: '0.04em', textTransform: 'uppercase', color: PALETTE.textMuted }}>
                    Resolved tier
                  </Typography>
                  <Stack direction="row" sx={{ alignItems: 'baseline', justifyContent: 'space-between', mt: 0.5 }}>
                    <Typography sx={{ fontSize: 28, fontWeight: 700, color: PALETTE.text, letterSpacing: '-0.02em', textTransform: 'capitalize' }}>
                      {selected.tier}
                    </Typography>
                    {selected.confidence !== null && (
                      <Stack sx={{ alignItems: 'flex-end' }}>
                        <Typography sx={{ fontSize: 11, color: PALETTE.textMuted }}>Confidence</Typography>
                        <Typography sx={{ fontSize: 20, fontWeight: 700, color: PALETTE.goodText, fontVariantNumeric: 'tabular-nums' }}>
                          {(selected.confidence * 100).toFixed(1)}%
                        </Typography>
                      </Stack>
                    )}
                  </Stack>
                </Box>

                {selected.model_name && selected.model_name.includes('REPLACE_ME') && (
                  <Box sx={{ p: 1.25, borderRadius: '10px', bgcolor: PALETTE.warningSoft, border: `1px solid ${PALETTE.warningEdge}` }}>
                    <Typography sx={{ fontSize: 12.5, fontWeight: 650, color: PALETTE.warningText }}>
                      model_name: not configured
                    </Typography>
                    <Typography sx={{ fontSize: 11.5, color: PALETTE.warningText, opacity: 0.85 }}>
                      Tier mapping still uses REPLACE_ME.
                    </Typography>
                  </Box>
                )}

                <Box>
                  <Typography sx={{ fontSize: 11, fontWeight: 650, letterSpacing: '0.04em', textTransform: 'uppercase', color: PALETTE.textMuted, mb: 0.5 }}>
                    Resolution reason
                  </Typography>
                  <Typography sx={{ fontSize: 12.5, color: PALETTE.textSecondary }}>{selected.reason}</Typography>
                </Box>

                {Object.keys(selected.metadata).length > 0 && (
                  <Box>
                    <Typography sx={{ fontSize: 11, fontWeight: 650, letterSpacing: '0.04em', textTransform: 'uppercase', color: PALETTE.textMuted, mb: 0.5 }}>
                      Input metadata
                    </Typography>
                    <Box sx={{ borderRadius: '8px', border: `1px solid ${PALETTE.hairline}` }}>
                      {Object.entries(selected.metadata).map(([k, v], i) => (
                        <Stack
                          key={k}
                          direction="row"
                          sx={{
                            justifyContent: 'space-between',
                            px: 1.25,
                            py: 0.75,
                            borderBottom: i < Object.keys(selected.metadata).length - 1 ? `1px dashed ${PALETTE.hairline}` : 'none',
                          }}
                        >
                          <Typography sx={{ fontSize: 12, color: PALETTE.textMuted }}>{k}</Typography>
                          <Mono size={12}>{v}</Mono>
                        </Stack>
                      ))}
                    </Box>
                  </Box>
                )}

                {selected.skill_needs && Object.keys(selected.skill_needs).length > 0 && (
                  <Box>
                    <Typography sx={{ fontSize: 11, fontWeight: 650, letterSpacing: '0.04em', textTransform: 'uppercase', color: PALETTE.textMuted, mb: 0.5 }}>
                      What this request needed
                    </Typography>
                    <Stack direction="row" sx={{ flexWrap: 'wrap', gap: 0.75 }}>
                      {(Object.entries(selected.skill_needs) as [Skill, number][])
                        .sort(([, a], [, b]) => b - a)
                        .map(([skill, need]) => (
                          <StateTag key={skill} label={`${SKILL_LABELS[skill] ?? skill} ${need.toFixed(1)}`} tone="brand" size="small" />
                        ))}
                      {selected.difficulty != null && (
                        <StateTag label={`Difficulty ${selected.difficulty.toFixed(2)}`} tone="warning" size="small" />
                      )}
                    </Stack>
                    <Typography sx={{ fontSize: 11.5, color: PALETTE.textMuted, mt: 0.5 }}>
                      A rating moves {selected.model_name}'s value for these skills, weighted by how much each was needed.
                    </Typography>
                  </Box>
                )}

                <Box>
                  <Typography sx={{ fontSize: 11, fontWeight: 650, letterSpacing: '0.04em', textTransform: 'uppercase', color: PALETTE.textMuted, mb: 0.5 }}>
                    Ratings
                  </Typography>
                  {selected.ratings?.length ? (
                    <Stack spacing={0.25}>
                      {selected.ratings.map((r, i) => (
                        <Stack key={i} direction="row" spacing={1} sx={{ alignItems: 'center' }}>
                          <Rating value={r.rating} readOnly size="small" />
                          <Typography sx={{ fontSize: 12, color: PALETTE.textMuted }}>by {r.judged_by}</Typography>
                        </Stack>
                      ))}
                    </Stack>
                  ) : (
                    <Typography sx={{ fontSize: 12.5, color: PALETTE.textMuted }}>
                      Not rated yet — use Rate below once you have checked the model's real output.
                    </Typography>
                  )}
                </Box>

                {selected.preferences?.length ? (
                  <Box>
                    <Typography sx={{ fontSize: 11, fontWeight: 650, letterSpacing: '0.04em', textTransform: 'uppercase', color: PALETTE.textMuted, mb: 0.5 }}>
                      Comparisons
                    </Typography>
                    <Stack spacing={0.25}>
                      {selected.preferences.map((p, i) => (
                        <Typography key={i} sx={{ fontSize: 12.5, color: PALETTE.textSecondary }}>
                          {p.outcome === 'tie'
                            ? `Same as ${p.other}`
                            : p.outcome === 'picked'
                              ? `Better than ${p.other}`
                              : `${p.other} was better`}
                          {p.skills.length > 0 && ` · ${p.skills.map((s) => SKILL_LABELS[s] ?? s).join(', ')}`}
                          <Box component="span" sx={{ color: PALETTE.textMuted }}> · by {p.judged_by}</Box>
                        </Typography>
                      ))}
                    </Stack>
                  </Box>
                ) : null}

                {selected.need_label && (
                  <Box>
                    <Typography sx={{ fontSize: 11, fontWeight: 650, letterSpacing: '0.04em', textTransform: 'uppercase', color: PALETTE.textMuted, mb: 0.5 }}>
                      Corrected needs
                    </Typography>
                    <Typography sx={{ fontSize: 12.5, color: PALETTE.textSecondary }}>
                      {(Object.entries(selected.need_label.needs) as [Skill, number][])
                        .filter(([, v]) => v > 0)
                        .sort(([, a], [, b]) => b - a)
                        .map(([s, v]) => `${SKILL_LABELS[s] ?? s} ${v.toFixed(1)}`)
                        .join(' · ') || 'No specific skill'}
                      <Box component="span" sx={{ color: PALETTE.textMuted }}> · by {selected.need_label.judged_by}</Box>
                    </Typography>
                  </Box>
                )}

                <Box>
                  <Typography sx={{ fontSize: 11, fontWeight: 650, letterSpacing: '0.04em', textTransform: 'uppercase', color: PALETTE.textMuted, mb: 0.5 }}>
                    Resolution path
                  </Typography>
                  <ResolutionPath ruleOutcome={selected.rule_outcome} classifierOutcome={selected.classifier_outcome} fallbackOutcome={selected.fallback_outcome} />
                </Box>

                <Stack direction="row" spacing={1}>
                  <Button fullWidth variant="outlined" startIcon={<Copy size={14} />} onClick={() => void copyDecision()}>
                    {copied ? 'Copied' : 'Copy decision'}
                  </Button>
                  <Button variant="outlined" startIcon={<Star size={14} />} onClick={openRate}>
                    {selected.ratings?.some((r) => r.judged_by === user?.username) ? 'Rate again' : 'Rate'}
                  </Button>
                  {selected.request_id && (
                    <Button variant="outlined" startIcon={<Scales size={14} />} onClick={openCompare}>
                      Compare
                    </Button>
                  )}
                </Stack>
              </Stack>
            )}
          </Panel>
        </Box>

        <InfoBanner
          tone="brand"
          badge="i"
          title="Decisions, not model traces"
          note="These records explain tier selection only. They contain no generated responses, model-call traces, verification, or escalation."
        />
      </Stack>

      <Dialog open={rateOpen} onClose={() => setRateOpen(false)} maxWidth="xs" fullWidth>
        {selected && (
          <>
            <DialogTitle sx={{ fontSize: 16, fontWeight: 650 }}>
              Rate <Mono size={14}>{selected.model_name}</Mono>'s output
            </DialogTitle>
            <DialogContent>
              <Stack spacing={1.5} sx={{ pt: 0.5 }}>
                <Typography sx={{ fontSize: 12.5, color: PALETTE.textMuted }}>
                  Was this model's real output actually correct/good for this request? This feeds the quality floor in
                  Model catalog — a model nobody has rated is never penalized.
                  {selected.request_id &&
                    ' It also adjusts the model’s skills that this request needed, so the next similar request is ranked with it.'}
                </Typography>
                {rateError && <Alert severity="error">{rateError}</Alert>}
                <Stack>
                  <Typography sx={{ fontSize: 12, color: PALETTE.textMuted }}>Quality (1 = wrong, 5 = correct)</Typography>
                  <Rating value={rating} onChange={(_, v) => setRating(v)} />
                </Stack>
                {rating !== null && rating <= 3 && selected.request_id && (
                  <SkillPicker label="What went wrong? (optional)" value={wrong} onChange={setWrong} labels={SKILL_FAULTS} />
                )}
                {/* P3: quality-field chip is read-only from the row's metadata (Q5.5.b = no re-tag).
                    Hidden entirely when the deployment has no GRADIST_QUALITY_FIELD. */}
                {qualityField && (
                  <Stack direction="row" spacing={0.75} sx={{ alignItems: 'center', flexWrap: 'wrap' }}>
                    <Typography sx={{ fontSize: 12, color: PALETTE.textMuted }}>{qualityField}:</Typography>
                    <StateTag
                      label={selected.metadata[qualityField] || '(not set on this request)'}
                      tone="default"
                      size="small"
                    />
                  </Stack>
                )}
                <Typography sx={{ fontSize: 11.5, color: PALETTE.textFaint }}>
                  Rating as {user?.username ?? 'you'} — recorded from your session.
                </Typography>
                <TextField label="Notes (optional)" value={notes} onChange={(e) => setNotes(e.target.value)} multiline rows={2} fullWidth />
              </Stack>
            </DialogContent>
            <DialogActions>
              <Button onClick={() => setRateOpen(false)}>Cancel</Button>
              <Button variant="contained" onClick={() => void submitRating()} disabled={!rating || saving}>
                Save rating
              </Button>
            </DialogActions>
          </>
        )}
      </Dialog>

      <Dialog open={compareOpen} onClose={() => setCompareOpen(false)} maxWidth="xs" fullWidth>
        {selected && (
          <>
            <DialogTitle sx={{ fontSize: 16, fontWeight: 650 }}>
              Compare <Mono size={14}>{selected.model_name}</Mono> with another model
            </DialogTitle>
            <DialogContent>
              <Stack spacing={1.5} sx={{ pt: 0.5 }}>
                <Typography sx={{ fontSize: 12.5, color: PALETTE.textMuted }}>
                  Run this request on both models outside the router, then record which output was better. Saved as a
                  comparison for learning later; it does not change picks yet.
                </Typography>
                {rateError && <Alert severity="error">{rateError}</Alert>}
                <Autocomplete
                  freeSolo
                  size="small"
                  options={(selected.shortlist ?? []).filter((id) => id !== selected.model_name)}
                  value={otherModel}
                  onInputChange={(_, v) => setOtherModel(v.trim())}
                  renderInput={(params) => (
                    <TextField {...params} label="Other model" helperText="From this request's shortlist, or any catalog model id" />
                  )}
                />
                <Stack spacing={0.5}>
                  <Typography sx={{ fontSize: 12, color: PALETTE.textMuted }}>Which output was better?</Typography>
                  <ToggleButtonGroup
                    exclusive
                    size="small"
                    value={outcome}
                    onChange={(_, v: PreferenceOutcome | null) => setOutcome(v)}
                    sx={{ flexWrap: 'wrap' }}
                  >
                    <ToggleButton value="picked" sx={{ textTransform: 'none' }}>{selected.model_name}</ToggleButton>
                    <ToggleButton value="tie" sx={{ textTransform: 'none' }}>Same</ToggleButton>
                    <ToggleButton value="other" sx={{ textTransform: 'none' }} disabled={!otherModel}>
                      {otherModel || 'Other model'}
                    </ToggleButton>
                  </ToggleButtonGroup>
                </Stack>
                {outcome && outcome !== 'tie' && (
                  <SkillPicker label="Where did the difference show? (optional)" value={compareSkills} onChange={setCompareSkills} />
                )}
                <TextField label="Notes (optional)" value={notes} onChange={(e) => setNotes(e.target.value)} multiline rows={2} fullWidth />
              </Stack>
            </DialogContent>
            <DialogActions>
              <Button onClick={() => setCompareOpen(false)}>Cancel</Button>
              <Button variant="contained" onClick={() => void submitCompare()} disabled={!otherModel || !outcome || saving}>
                Save comparison
              </Button>
            </DialogActions>
          </>
        )}
      </Dialog>
    </>
  )
}
