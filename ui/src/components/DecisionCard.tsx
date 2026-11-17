import { Card, Stack, Typography } from '@mui/material'
import { Mono, ResolutionPath, StateTag } from './Primitives'
import { PALETTE, tierTone } from '../theme'
import type { Decision } from '../types'

/**
 * Shows one resolve() result. Used inline, in an expanded example row on Data > Examples, when
 * a user asks "what would this one resolve to" — the Playground page has its own larger
 * version of this (the big tier card), so this stays compact.
 */
export function DecisionCard({ decision }: { decision: Decision }) {
  return (
    <Card variant="outlined" sx={{ p: 2 }}>
      <Stack direction="row" spacing={1.25} sx={{ alignItems: 'center', flexWrap: 'wrap' }}>
        <StateTag label={decision.tier} tone={tierTone(decision.tier)} />
        <Mono size={13}>{decision.model_name}</Mono>
        {decision.confidence !== null && (
          <Mono size={12} muted>
            confidence {decision.confidence.toFixed(2)}
          </Mono>
        )}
      </Stack>
      <Typography sx={{ fontSize: 13, color: PALETTE.textMuted, mt: 1, mb: 1 }}>{decision.reason}</Typography>
      <ResolutionPath ruleOutcome={decision.rule_outcome} classifierOutcome={decision.classifier_outcome} fallbackOutcome={decision.fallback_outcome} />
    </Card>
  )
}
