import { ArrowRight, ArrowCounterClockwise, Copy } from '@phosphor-icons/react'
import {
  Alert,
  Box,
  Button,
  MenuItem,
  Slider,
  Stack,
  Tab,
  Tabs,
  TextField,
  Typography,
} from '@mui/material'
import { useEffect, useState } from 'react'
import { api, describeError } from '../api'
import { ImagePicker } from '../components/ImagePicker'
import { MetadataFieldsEditor } from '../components/MetadataFieldsEditor'
import { InfoBanner, Mono, PageHead, Panel, ResolutionPath, SectionHead, StateTag, StepFlow } from '../components/Primitives'
import { PALETTE, TONE_SOFT, tierTone } from '../theme'
import type { Decision, TiersResponse } from '../types'

export function Resolve({
  tiers,
  selectedModels,
  onResolved,
}: {
  tiers: TiersResponse | null
  /** From the Model catalog page — `null` means "every model in the catalog". */
  selectedModels: Set<string> | null
  onResolved: () => void
}) {
  const [metadata, setMetadata] = useState<Record<string, string>>({})
  const [text, setText] = useState('')
  const [image, setImage] = useState<File | null>(null)
  const [provider, setProvider] = useState('')
  const [minConfidence, setMinConfidence] = useState(0)
  const [minQuality, setMinQuality] = useState(0)
  const [qualityField, setQualityField] = useState<string | null>(null)
  const [decision, setDecision] = useState<Decision | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [copied, setCopied] = useState(false)
  const [inputTab, setInputTab] = useState<'text' | 'structured'>('text')

  const providers = tiers ? Array.from(new Set(tiers.tiers.flatMap((t) => Object.keys(t.models)))) : []

  useEffect(() => {
    void (async () => {
      try {
        setQualityField((await api.quality()).field)
      } catch {
        // Quality is an optional feature — a failed check here just leaves it disabled.
      }
    })()
  }, [])

  function resetInputs() {
    setMetadata({})
    setText('')
    setImage(null)
    setProvider('')
    setMinConfidence(0)
    setMinQuality(0)
    setDecision(null)
    setError('')
  }

  async function run() {
    setLoading(true)
    setError('')
    setDecision(null)
    setCopied(false)
    try {
      setDecision(
        await api.resolve({
          metadata,
          text,
          image,
          provider: provider || undefined,
          minConfidence,
          selectedModels: selectedModels ? Array.from(selectedModels) : null,
          minQuality: qualityField ? minQuality : undefined,
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

  const modelNameNeedsConfig = decision?.model_name?.includes('REPLACE_ME')

  return (
    <>
      <PageHead
        title="Playground"
        lede="Test a routing decision without calling a model."
        actions={
          <Button startIcon={<ArrowCounterClockwise size={14} />} onClick={resetInputs}>
            Reset inputs
          </Button>
        }
      />

      <Stack spacing={2}>
        <InfoBanner
          tone="info"
          badge="PREVIEW"
          title="Explore a sample decision"
          note="Classifier results are illustrative. Training is blocked until a labeling strategy is selected."
          actions={<Typography sx={{ fontSize: 11.5, fontWeight: 600, color: PALETTE.infoText }}>No model calls</Typography>}
        />

        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1.1fr 1fr' }, gap: 2, alignItems: 'start' }}>
          <Panel
            title="Input"
            actions={
              <Button size="small" startIcon={<ArrowRight size={12} />} disabled>
                Load example
              </Button>
            }
          >
            <Tabs
              value={inputTab}
              onChange={(_, v) => setInputTab(v)}
              sx={{ minHeight: 34, mb: 1.5, '& .MuiTab-root': { minHeight: 34, fontSize: 12.5, fontWeight: 600, textTransform: 'none' } }}
            >
              <Tab value="text" label="Text + metadata" />
              <Tab value="structured" label="Structured input" disabled />
            </Tabs>

            <Stack spacing={1.75}>
              <SectionHead title="Text" />
              <TextField
                value={text}
                onChange={(e) => setText(e.target.value)}
                multiline
                rows={4}
                fullWidth
                placeholder="Compare three cloud database architectures for a multilingual analytics platform…"
                helperText={`${text.length} characters`}
              />

              <SectionHead title="Metadata" note="Caller-defined fields and known values." />
              <MetadataFieldsEditor value={metadata} onChange={setMetadata} />

              <ImagePicker value={image} onChange={setImage} />

              <Stack direction="row" spacing={1.5} sx={{ alignItems: 'center' }}>
                <TextField
                  select
                  label="Provider (optional)"
                  value={provider}
                  onChange={(e) => setProvider(e.target.value)}
                  sx={{ minWidth: 200 }}
                >
                  <MenuItem value="">Default ({tiers?.default_provider ?? '—'})</MenuItem>
                  {providers.map((p) => (
                    <MenuItem key={p} value={p}>
                      {p}
                    </MenuItem>
                  ))}
                </TextField>
              </Stack>

              <Stack>
                <Typography sx={{ fontSize: 12, color: PALETTE.textMuted }}>
                  Minimum confidence to trust the classifier: {minConfidence.toFixed(2)}
                </Typography>
                <Slider
                  value={minConfidence}
                  onChange={(_, v) => setMinConfidence(v as number)}
                  min={0}
                  max={1}
                  step={0.05}
                  aria-label="Minimum confidence"
                />
              </Stack>
              <Stack>
                <Typography sx={{ fontSize: 12, color: PALETTE.textMuted }}>
                  {qualityField
                    ? `Minimum human quality score (bucketed by "${qualityField}"): ${minQuality.toFixed(1)}`
                    : 'Minimum quality score: not enabled for this deployment.'}
                </Typography>
                <Slider
                  value={minQuality}
                  onChange={(_, v) => setMinQuality(v as number)}
                  min={0}
                  max={5}
                  step={0.5}
                  disabled={!qualityField}
                  aria-label="Minimum quality"
                />
              </Stack>

              <Typography sx={{ fontSize: 11.5, color: PALETTE.textFaint }}>
                Candidate models: {selectedModels ? `${selectedModels.size} selected in Model catalog` : 'all models in the catalog (or tiers.yaml, if none is configured)'}
              </Typography>

              <Button
                variant="contained"
                onClick={() => void run()}
                disabled={loading}
                endIcon={<ArrowRight size={14} />}
                sx={{ py: 1.1 }}
              >
                {loading ? 'Resolving…' : 'Resolve tier'}
              </Button>
            </Stack>
          </Panel>

          <Panel
            title="Decision"
            note={decision ? undefined : 'No model calls.'}
            actions={decision && <StateTag label="Sample result" tone="default" size="small" />}
          >
            {error && (
              <Alert severity="error" sx={{ mb: 2 }}>
                {error}
              </Alert>
            )}
            {!decision ? (
              <Stack sx={{ py: 4, alignItems: 'center', justifyContent: 'center' }}>
                <Typography sx={{ fontSize: 13, color: PALETTE.textMuted, textAlign: 'center', maxWidth: '36ch' }}>
                  Fill in text + metadata on the left and click <Box component="span" sx={{ fontWeight: 650 }}>Resolve tier</Box> to see the decision here.
                </Typography>
              </Stack>
            ) : (
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
                    Selected tier
                  </Typography>
                  <Stack direction="row" sx={{ justifyContent: 'space-between', alignItems: 'baseline', mt: 0.5 }}>
                    <Typography sx={{ fontSize: 30, fontWeight: 700, letterSpacing: '-0.025em', color: PALETTE.text, textTransform: 'capitalize' }}>
                      {decision.tier}
                    </Typography>
                    {decision.confidence !== null && (
                      <Stack sx={{ alignItems: 'flex-end' }}>
                        <Typography sx={{ fontSize: 11, color: PALETTE.textMuted }}>Confidence</Typography>
                        <Typography sx={{ fontSize: 20, fontWeight: 700, color: PALETTE.goodText, fontVariantNumeric: 'tabular-nums' }}>
                          {(decision.confidence * 100).toFixed(1)}%
                        </Typography>
                      </Stack>
                    )}
                  </Stack>
                  <Typography sx={{ fontSize: 12, color: PALETTE.textMuted, mt: 0.5 }}>
                    Resolved by {decision.rule_outcome !== 'no matching rule' ? 'rule' : decision.fallback_outcome === 'not needed' ? 'classifier' : 'fallback'}
                  </Typography>
                </Box>

                {modelNameNeedsConfig && (
                  <Box sx={{ p: 1.25, borderRadius: '10px', bgcolor: PALETTE.warningSoft, border: `1px solid ${PALETTE.warningEdge}` }}>
                    <Stack direction="row" sx={{ justifyContent: 'space-between', alignItems: 'center', gap: 1 }}>
                      <Box>
                        <Typography sx={{ fontSize: 12.5, fontWeight: 650, color: PALETTE.warningText }}>
                          Model mapping required
                        </Typography>
                        <Typography sx={{ fontSize: 11.5, color: PALETTE.warningText, opacity: 0.85 }}>
                          model_name cannot resolve while the mapping is REPLACE_ME.
                        </Typography>
                      </Box>
                      <Button size="small" endIcon={<ArrowRight size={12} />}>
                        Configure tiers
                      </Button>
                    </Stack>
                  </Box>
                )}

                <Box>
                  <Typography sx={{ fontSize: 11, fontWeight: 650, letterSpacing: '0.04em', textTransform: 'uppercase', color: PALETTE.textMuted, mb: 0.5 }}>
                    Why this tier
                  </Typography>
                  <Typography sx={{ fontSize: 12.5, color: PALETTE.textSecondary }}>{decision.reason}</Typography>
                </Box>

                <Box>
                  <Typography sx={{ fontSize: 11, fontWeight: 650, letterSpacing: '0.04em', textTransform: 'uppercase', color: PALETTE.textMuted, mb: 0.5 }}>
                    Resolution path
                  </Typography>
                  <ResolutionPath
                    ruleOutcome={decision.rule_outcome}
                    classifierOutcome={decision.classifier_outcome}
                    fallbackOutcome={decision.fallback_outcome}
                  />
                </Box>

                <Typography sx={{ fontSize: 11.5, color: PALETTE.textFaint }}>
                  Decision fields: tier · model_name · confidence · reason
                </Typography>

                <Button
                  variant="outlined"
                  fullWidth
                  startIcon={<Copy size={14} />}
                  onClick={() => void copyDecision()}
                >
                  {copied ? 'Copied' : 'Copy decision'}
                </Button>
              </Stack>
            )}
          </Panel>
        </Box>

        <Panel title="Feature builder" note="Metadata one-hot + bge-m3 text embeddings.">
          <StepFlow
            steps={[
              { label: 'Metadata one-hot', note: 'Per-field, plus an unknown bucket per field.' },
              { label: 'Text embedding', note: 'bge-m3 · multilingual · MIT.' },
              { label: 'Image embedding', note: 'DINOv2 · optional slot · zero-vector + has-image flag when absent.' },
            ]}
          />
          <Stack direction={{ xs: 'column', md: 'row' }} sx={{ justifyContent: 'space-between', alignItems: 'center', mt: 1.5, gap: 1 }}>
            <Mono size={11.5} muted>Thumbnail slot reserved</Mono>
            <StateTag label="Labeling strategy pending" tone="warning" size="small" />
          </Stack>
        </Panel>
      </Stack>
    </>
  )
}
