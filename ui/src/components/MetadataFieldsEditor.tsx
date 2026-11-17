import { Plus, Trash } from '@phosphor-icons/react'
import { Button, IconButton, Stack, TextField } from '@mui/material'
import { useId, useState } from 'react'

/**
 * key=value metadata field editor. Reused wherever a caller needs to describe one request's
 * metadata by hand: the "Add example" form (Data > Examples) and the "Try it" form
 * (Try it > Resolve) both use this exact component, so the two screens stay consistent.
 */
export function MetadataFieldsEditor({
  value,
  onChange,
}: {
  value: Record<string, string>
  onChange: (next: Record<string, string>) => void
}) {
  const keyId = useId()
  const [key, setKey] = useState('')
  const [val, setVal] = useState('')

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
    <Stack spacing={1}>
      {Object.entries(value).map(([k, v]) => (
        <Stack key={k} direction="row" spacing={1} sx={{ alignItems: 'center' }}>
          <TextField value={k} disabled fullWidth label="Field" />
          <TextField value={v} disabled fullWidth label="Value" />
          <IconButton aria-label={`Remove field ${k}`} onClick={() => remove(k)}>
            <Trash size={16} />
          </IconButton>
        </Stack>
      ))}
      <Stack direction="row" spacing={1} sx={{ alignItems: 'center' }}>
        <TextField
          id={keyId}
          label="Field (e.g. task)"
          value={key}
          onChange={(e) => setKey(e.target.value)}
          fullWidth
        />
        <TextField label="Value" value={val} onChange={(e) => setVal(e.target.value)} fullWidth />
        <Button startIcon={<Plus size={16} />} onClick={add} variant="outlined">
          Add
        </Button>
      </Stack>
    </Stack>
  )
}
