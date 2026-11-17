import { Alert, Box, Table, TableBody, TableCell, TableHead, TableRow } from '@mui/material'
import { Mono, MetricStrip } from './Primitives'
import type { Metric } from './Primitives'
import type { TrainResult } from '../types'

/**
 * Accuracy vs. baseline, per-bucket breakdown. Reused right after a training run (Classifier >
 * Train) and on the standalone Classifier > Evaluate screen — the same component, same numbers.
 */
export function EvalSummary({ result }: { result: TrainResult }) {
  const { report } = result
  const buckets = Object.entries(report.accuracy_by_bucket ?? {})

  const metrics: Metric[] = [
    { label: 'Labels used', value: result.n_labels, tone: 'brand' },
    { label: 'From rules', value: result.n_rule_labels },
    { label: 'From judgments', value: result.n_human_labels },
    { label: 'Uncovered by any rule', value: result.n_uncovered, tone: result.n_uncovered > 0 ? 'warning' : 'default' },
    { label: 'Uses images', value: result.uses_images ? 'Yes' : 'No', tone: result.uses_images ? 'good' : 'default' },
  ]
  if (report.accuracy !== undefined) {
    metrics.push(
      { label: 'Accuracy on this set', value: report.accuracy.toFixed(2), tone: 'brand' },
      { label: 'Baseline (always one tier)', value: (report.baseline_accuracy ?? 0).toFixed(2) },
    )
  }

  return (
    <Box sx={{ display: 'grid', gap: 1.75 }}>
      <MetricStrip metrics={metrics} />
      {buckets.length > 0 && (
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>{report.field}</TableCell>
              <TableCell align="right">Accuracy</TableCell>
              <TableCell align="right">Examples</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {buckets.map(([bucket, accuracy]) => (
              <TableRow key={bucket}>
                <TableCell>
                  <Mono>{bucket}</Mono>
                </TableCell>
                <TableCell align="right">
                  <Mono>{accuracy.toFixed(2)}</Mono>
                </TableCell>
                <TableCell align="right">
                  <Mono muted>{report.bucket_sizes?.[bucket] ?? '—'}</Mono>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
      <Alert severity="warning">
        This evaluates on the same examples it trained on — there are too few labels yet for an
        honest held-out split. It only confirms the pipeline runs end to end, not real quality
        (see ARCHITECTURE.md §8).
      </Alert>
    </Box>
  )
}
