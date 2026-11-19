/**
# Fact-forcing
Callers: Resolve.tsx (Playground trailing + media row), Examples.tsx ImagePicker.
Existing ImagePicker was image-only — extended in place so video→JPEG still extract
happens client-side before /v1/resolve (API still only accepts image bytes).
No secrets. User: "playground mai thumbnail or video bhi de skte because aab to tum
video se thumbnail extract kar hi loge .. please make sure frontend pe changes ho rhe
hai .. woh sab align ho .. fizul ki chize nhi chiye"
*/
import { FilmStrip, UploadSimple, X } from '@phosphor-icons/react'
import { Box, Button, CircularProgress, IconButton, Stack, Typography } from '@mui/material'
import { useEffect, useId, useRef, useState } from 'react'

/**
 * Thumbnail or video clip. Videos are never uploaded as-is — a mid-clip JPEG is
 * extracted in the browser and that still is what the router receives.
 */
export function ImagePicker({
  value,
  onChange,
  compact = false,
  inline = false,
  hint,
  sourceKind = null,
  sourceName = null,
}: {
  value: File | null
  onChange: (file: File | null, meta?: { kind: 'image' | 'video'; sourceName: string } | null) => void
  compact?: boolean
  inline?: boolean
  hint?: string
  sourceKind?: 'image' | 'video' | null
  sourceName?: string | null
}) {
  const inputId = useId()
  const inputRef = useRef<HTMLInputElement>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [localError, setLocalError] = useState('')

  useEffect(() => {
    if (!value) {
      setPreviewUrl(null)
      return
    }
    const url = URL.createObjectURL(value)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [value])

  const caption =
    hint ??
    'Thumbnail or video — video stays local; we extract one still for the router (no LLM).'

  async function handleFiles(list: FileList | null) {
    const file = list?.[0] ?? null
    setLocalError('')
    if (!file) {
      onChange(null, null)
      return
    }
    if (file.type.startsWith('image/')) {
      onChange(file, { kind: 'image', sourceName: file.name })
      return
    }
    if (file.type.startsWith('video/') || /\.(mp4|mov|webm|mkv|m4v|avi)$/i.test(file.name)) {
      setBusy(true)
      try {
        const still = await extractVideoThumbnail(file)
        onChange(still, { kind: 'video', sourceName: file.name })
      } catch (err) {
        setLocalError(err instanceof Error ? err.message : 'Could not read a frame from that video.')
        onChange(null, null)
        if (inputRef.current) inputRef.current.value = ''
      } finally {
        setBusy(false)
      }
      return
    }
    setLocalError('Use an image or video file.')
    if (inputRef.current) inputRef.current.value = ''
  }

  function clear() {
    onChange(null, null)
    setLocalError('')
    if (inputRef.current) inputRef.current.value = ''
  }

  const label = busy
    ? 'Extracting…'
    : value
      ? sourceKind === 'video'
        ? 'Replace clip'
        : 'Replace image'
      : 'Image / video'

  const fileInput = (
    <input
      id={inputId}
      ref={inputRef}
      type="file"
      accept="image/*,video/*,.mp4,.mov,.webm,.mkv,.m4v,.avi"
      hidden
      onChange={(e) => void handleFiles(e.target.files)}
    />
  )

  const preview = previewUrl ? (
    <Box
      component="img"
      src={previewUrl}
      alt="Thumbnail preview"
      sx={{
        width: inline ? 40 : compact ? '100%' : 120,
        height: inline ? 40 : compact ? 88 : 120,
        maxWidth: compact ? 140 : undefined,
        objectFit: 'cover',
        borderRadius: inline ? '6px' : 1,
        border: '1px solid',
        borderColor: 'divider',
        flexShrink: 0,
        mt: compact && !inline ? 1 : 0,
      }}
    />
  ) : null

  const controls = (
    <Stack direction="row" spacing={1} sx={{ alignItems: 'center', flexShrink: 0 }}>
      {fileInput}
      <Button
        size="small"
        startIcon={
          busy ? (
            <CircularProgress size={14} />
          ) : sourceKind === 'video' ? (
            <FilmStrip size={16} />
          ) : (
            <UploadSimple size={16} />
          )
        }
        variant="outlined"
        disabled={busy}
        onClick={() => inputRef.current?.click()}
        sx={{ height: 40, px: 1.5, textTransform: 'none', whiteSpace: 'nowrap' }}
      >
        {label}
      </Button>
      {value && (
        <IconButton
          size="small"
          aria-label="Remove media"
          onClick={clear}
          sx={{ height: 40, width: 40 }}
          disabled={busy}
        >
          <X size={16} />
        </IconButton>
      )}
      {inline && preview}
    </Stack>
  )

  if (inline) {
    return (
      <Stack spacing={0.5} sx={{ alignItems: 'flex-start' }}>
        {controls}
        {sourceKind === 'video' && sourceName && (
          <Typography sx={{ fontSize: 11, color: 'text.secondary', maxWidth: 220 }} noWrap title={sourceName}>
            Still from {sourceName}
          </Typography>
        )}
        {localError && <Typography sx={{ fontSize: 11, color: 'error.main' }}>{localError}</Typography>}
      </Stack>
    )
  }

  if (compact) {
    return (
      <Box sx={{ width: '100%' }}>
        <Typography
          component="label"
          htmlFor={inputId}
          sx={{
            display: 'block',
            fontSize: 12,
            fontWeight: 500,
            color: 'text.secondary',
            mb: '6px',
            lineHeight: 1.2,
          }}
        >
          Image / video
        </Typography>
        {controls}
        {!inline && preview}
        {localError && (
          <Typography sx={{ fontSize: 11, color: 'error.main', mt: 0.5 }}>{localError}</Typography>
        )}
      </Box>
    )
  }

  return (
    <Stack spacing={1}>
      <Typography variant="caption" color="text.secondary" component="label" htmlFor={inputId}>
        {caption}
      </Typography>
      {controls}
      {preview}
      {sourceKind === 'video' && sourceName && (
        <Typography variant="caption" color="text.secondary">
          Still extracted from {sourceName}
        </Typography>
      )}
      {localError && (
        <Typography variant="caption" color="error">
          {localError}
        </Typography>
      )}
    </Stack>
  )
}

/** Seek ~35% into the clip and encode one JPEG still. */
export function extractVideoThumbnail(file: File, atRatio = 0.35): Promise<File> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file)
    const video = document.createElement('video')
    video.preload = 'metadata'
    video.muted = true
    video.playsInline = true
    video.setAttribute('playsinline', 'true')

    const fail = (msg: string) => {
      URL.revokeObjectURL(url)
      reject(new Error(msg))
    }

    video.onloadedmetadata = () => {
      const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 1
      const t = Math.min(Math.max(duration * atRatio, 0.05), Math.max(duration - 0.05, 0.05))
      try {
        video.currentTime = t
      } catch {
        video.currentTime = 0
      }
    }

    video.onseeked = () => {
      try {
        const w = video.videoWidth || 640
        const h = video.videoHeight || 360
        const canvas = document.createElement('canvas')
        canvas.width = w
        canvas.height = h
        const ctx = canvas.getContext('2d')
        if (!ctx) {
          fail('Canvas unavailable in this browser.')
          return
        }
        ctx.drawImage(video, 0, 0, w, h)
        canvas.toBlob(
          (blob) => {
            URL.revokeObjectURL(url)
            if (!blob) {
              reject(new Error('Could not encode thumbnail.'))
              return
            }
            const base = file.name.replace(/\.[^.]+$/, '') || 'clip'
            resolve(new File([blob], `${base}_thumb.jpg`, { type: 'image/jpeg' }))
          },
          'image/jpeg',
          0.92,
        )
      } catch (err) {
        fail(err instanceof Error ? err.message : 'Frame extract failed.')
      }
    }

    video.onerror = () => fail('Browser could not decode that video.')
    video.src = url
    void video.load()
  })
}
