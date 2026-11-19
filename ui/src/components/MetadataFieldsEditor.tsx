import { Plus, Trash } from '@phosphor-icons/react'
import { Button, IconButton, Stack, TextField } from '@mui/material'
import { useId, useState, type ReactNode } from 'react'

/**
 * key=value metadata editor. Callers: Resolve (Playground), Examples.
 * `trailing` sits on the same row as Field / Value / Add (e.g. thumbnail upload).
 */
export function MetadataFieldsEditor({
  value,
  onChange,
  size = 'medium',
  trailing,
}: {
  value: Record<string, string>
  onChange: (next: Record<string, string>) => void
  size?: 'small' | 'medium'
  trailing?: ReactNode
}) {
  const keyId = useId()
  const [key, setKey] = useState('')
  const [val, setVal] = useState('')
  const controlH = size === 'small' ? 40 : 56

  function add() {
    const trimmedKey = key.trim()
    if (!trimmedKey || !val.trim()) return
    onChange({ ...value, [trimmedKey]: val.trim() })
    setKey('')
    setVal('')
  }

  function remove(k: string) {
    const next = { ...value }
    delete next[k]
    onChange(next)
  }

  return (
    <Stack spacing={1.25} sx={{ width: '100%' }}>
      {Object.entries(value).map(([k, v]) => (
        <Stack key={k} direction="row" spacing={1} sx={{ alignItems: 'center', width: '100%' }}>
          <TextField size={size} value={k} disabled fullWidth label="Field" />
          <TextField size={size} value={v} disabled fullWidth label="Value" />
          <IconButton size={size} aria-label={`Remove field ${k}`} onClick={() => remove(k)} sx={{ height: controlH, width: controlH, flexShrink: 0 }}>
            <Trash size={16} />
          </IconButton>
        </Stack>
      ))}
      <Stack direction="row" spacing={1} sx={{ alignItems: 'center', width: '100%', flexWrap: { xs: 'wrap', sm: 'nowrap' } }}>
        <TextField
          id={keyId}
          size={size}
          label="Field (e.g. channel)"
          value={key}
          onChange={(e) => setKey(e.target.value)}
          sx={{ flex: 1, minWidth: 120 }}
        />
        <TextField
          size={size}
          label="Value"
          value={val}
          onChange={(e) => setVal(e.target.value)}
          sx={{ flex: 1, minWidth: 120 }}
        />
        <Button
          size={size}
          startIcon={<Plus size={16} />}
          onClick={add}
          variant="outlined"
          sx={{ flexShrink: 0, height: controlH, px: 1.5, textTransform: 'none' }}
        >
          Add
        </Button>
        {trailing}
      </Stack>
    </Stack>
  )
}
