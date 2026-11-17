import { Alert, Button, Stack } from '@mui/material'
import { useState } from 'react'
import { api, describeError } from '../api'
import { EvalSummary } from '../components/EvalSummary'
import { PageHead, Panel } from '../components/Primitives'
import type { TrainResult } from '../types'

export function Train({
  onTrained,
  result,
}: {
  onTrained: (result: TrainResult) => void
  result: TrainResult | null
}) {
  const [training, setTraining] = useState(false)
  const [error, setError] = useState('')

  async function train() {
    setTraining(true)
    setError('')
    try {
      onTrained(await api.train())
    } catch (err) {
      setError(describeError(err))
    } finally {
      setTraining(false)
    }
  }

  return (
    <>
      <PageHead
        title="Train"
        lede="Trains on every rule-covered and human-judged example currently stored. If any of them has an image, the image model loads here for the first time (can take a little while)."
      />
      <Stack spacing={2}>
        <Panel>
          <Button variant="contained" onClick={() => void train()} disabled={training}>
            {training ? 'Training…' : 'Train now'}
          </Button>
        </Panel>
        {error && <Alert severity="error">{error}</Alert>}
        {result && (
          <Panel title="Result">
            <EvalSummary result={result} />
          </Panel>
        )}
      </Stack>
    </>
  )
}
