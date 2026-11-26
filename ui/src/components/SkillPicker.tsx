import { Chip, Stack, Typography } from '@mui/material'
import { PALETTE } from '../theme'
import type { Skill } from '../types'

export const SKILL_LABELS: Record<Skill, string> = {
  video: 'Video watching',
  audio: 'Hearing speech',
  hindi: 'Hindi / Indian languages',
  reasoning: 'Reasoning and rules',
  long_input: 'Long input',
  writing: 'Writing',
  fast: 'Quick and cheap',
}

/** The same skills phrased as faults, for "What went wrong?". */
export const SKILL_FAULTS: Record<Skill, string> = {
  video: 'Missed what was on screen',
  audio: 'Misheard the speech',
  hindi: 'Hindi was wrong',
  reasoning: 'Broke a rule or got the logic wrong',
  long_input: 'Lost track of the long input',
  writing: 'Poor writing',
  fast: 'Too slow or too expensive',
}

/** Toggle chips, one per skill. */
export function SkillPicker({
  label,
  value,
  onChange,
  labels = SKILL_LABELS,
}: {
  label: string
  value: Skill[]
  onChange: (next: Skill[]) => void
  labels?: Record<Skill, string>
}) {
  return (
    <Stack spacing={0.5}>
      <Typography sx={{ fontSize: 12, color: PALETTE.textMuted }}>{label}</Typography>
      <Stack direction="row" sx={{ flexWrap: 'wrap', gap: 0.75 }}>
        {(Object.keys(labels) as Skill[]).map((skill) => {
          const on = value.includes(skill)
          return (
            <Chip
              key={skill}
              size="small"
              label={labels[skill]}
              color={on ? 'primary' : 'default'}
              variant={on ? 'filled' : 'outlined'}
              aria-pressed={on}
              onClick={() => onChange(on ? value.filter((s) => s !== skill) : [...value, skill])}
            />
          )
        })}
      </Stack>
    </Stack>
  )
}
