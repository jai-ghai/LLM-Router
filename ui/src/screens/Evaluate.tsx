import { Alert } from '@mui/material'
import { EvalSummary } from '../components/EvalSummary'
import { PageHead, Panel } from '../components/Primitives'
import type { TrainResult } from '../types'

/** Shows the same report Train produced (lifted state in App.tsx) — the standalone view for
 * coming back to it without retraining. */
export function Evaluate({ result }: { result: TrainResult | null }) {
  return (
    <>
      <PageHead title="Evaluate" lede="The report from the last training run this session." />
      {result ? (
        <Panel title="Result">
          <EvalSummary result={result} />
        </Panel>
      ) : (
        <Alert severity="info">Nothing trained yet this session. Go to Classifier → Train first.</Alert>
      )}
    </>
  )
}
