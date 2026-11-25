import { Alert, Button, Stack, TextField, Typography } from '@mui/material'
import { useEffect, useState } from 'react'
import { api, describeError } from '../api'
import { Mono, PageHead, Panel, StateTag } from '../components/Primitives'
import { PALETTE } from '../theme'
import type { UncoveredExample } from '../types'

export function HumanJudging({ tierNames }: { tierNames: string[] }) {
  const [examples, setExamples] = useState<UncoveredExample[]>([])
  const [error, setError] = useState('')
  const [judgedBy, setJudgedBy] = useState('')

  async function load() {
    try {
      setExamples((await api.uncovered()).examples)
    } catch (err) {
      setError(describeError(err))
    }
  }

  useEffect(() => {
    void load()
  }, [])

  async function judge(exampleId: string, tier: string) {
    if (!judgedBy.trim()) return setError('Enter your name in "Judged by" first.')
    setError('')
    try {
      await api.addJudgment(exampleId, tier, judgedBy.trim(), '')
      await load()
    } catch (err) {
      setError(describeError(err))
    }
  }

  return (
    <>
      <PageHead
        title="Human judging"
        lede="Examples no rule reaches. Judging one here gives the classifier a label it would otherwise never see."
      />
      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
      <Stack spacing={2}>
        <TextField label="Judged by" value={judgedBy} onChange={(e) => setJudgedBy(e.target.value)} sx={{ maxWidth: 240 }} />
        {examples.length === 0 && <Alert severity="success">Nothing uncovered right now.</Alert>}
        <Stack spacing={1.5}>
          {examples.map((example) => (
            <Panel key={example.example_id}>
              <Stack direction="row" spacing={0.75} sx={{ flexWrap: 'wrap', mb: 1 }}>
                {Object.entries(example.metadata).map(([k, v]) => (
                  <Mono key={k} size={12.5}>
                    {k}={v}
                  </Mono>
                ))}
                {example.has_image && <StateTag label="has image" tone="info" size="small" />}
              </Stack>
              <Typography sx={{ fontSize: 13.5, color: PALETTE.text, mb: 1.5 }}>{example.text}</Typography>
              <Stack direction="row" spacing={1}>
                {tierNames.map((tier) => (
                  <Button key={tier} size="small" variant="outlined" onClick={() => void judge(example.example_id, tier)}>
                    {tier}
                  </Button>
                ))}
              </Stack>
            </Panel>
          ))}
        </Stack>
      </Stack>
    </>
  )
}
