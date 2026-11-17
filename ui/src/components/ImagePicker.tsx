import { UploadSimple, X } from '@phosphor-icons/react'
import { Box, Button, IconButton, Stack, Typography } from '@mui/material'
import { useEffect, useId, useRef, useState } from 'react'

/**
 * Image upload + preview. Reused in the "Add example" form (Data > Examples) and the
 * "Try it" form (Try it > Resolve) — the same picker, so uploading and clearing an image works
 * identically everywhere it appears.
 */
export function ImagePicker({ value, onChange }: { value: File | null; onChange: (file: File | null) => void }) {
  const inputId = useId()
  const inputRef = useRef<HTMLInputElement>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)

  useEffect(() => {
    if (!value) {
      setPreviewUrl(null)
      return
    }
    const url = URL.createObjectURL(value)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [value])

  return (
    <Stack spacing={1}>
      <Typography variant="caption" color="text.secondary" component="label" htmlFor={inputId}>
        Thumbnail (optional) — used in the prediction once the classifier has been trained with
        at least one image example.
      </Typography>
      <input
        id={inputId}
        ref={inputRef}
        type="file"
        accept="image/*"
        hidden
        onChange={(e) => onChange(e.target.files?.[0] ?? null)}
      />
      <Stack direction="row" spacing={1} sx={{ alignItems: 'center' }}>
        <Button startIcon={<UploadSimple size={16} />} variant="outlined" onClick={() => inputRef.current?.click()}>
          {value ? 'Replace image' : 'Choose image'}
        </Button>
        {value && (
          <IconButton
            aria-label="Remove the chosen image"
            onClick={() => {
              onChange(null)
              if (inputRef.current) inputRef.current.value = ''
            }}
          >
            <X size={16} />
          </IconButton>
        )}
      </Stack>
      {previewUrl && (
        <Box
          component="img"
          src={previewUrl}
          alt="Chosen thumbnail preview"
          sx={{ width: 120, height: 120, objectFit: 'cover', borderRadius: 1, border: '1px solid', borderColor: 'divider' }}
        />
      )}
    </Stack>
  )
}
