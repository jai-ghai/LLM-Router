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

/** What was captured from the picked file, and what is actually sent to the router. */
export interface MediaInfo {
  kind: 'image' | 'video'
  sourceName: string
  sourceType: string
  sourceBytes: number
  width?: number
  height?: number
  durationSec?: number
  frameAtSec?: number
  sentName: string
  sentType: string
  sentBytes: number
}

/**
 * Thumbnail or video clip. Videos are never uploaded as-is — a mid-clip JPEG is
 * extracted in the browser and that still is what the router receives.
 */
export function ImagePicker({
  value,
  onChange,
  compact = false,
  inline = false,
  card = false,
  media = null,
  hint,
  sourceKind = null,
  sourceName = null,
}: {
  value: File | null
  onChange: (file: File | null, meta?: MediaInfo | null) => void
  compact?: boolean
  inline?: boolean
  /** Full-width drop zone; once filled, shows the extracted frame beside the file facts. */
  card?: boolean
  media?: MediaInfo | null
  hint?: string
  sourceKind?: 'image' | 'video' | null
  sourceName?: string | null
}) {
  const inputId = useId()
  const inputRef = useRef<HTMLInputElement>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [localError, setLocalError] = useState('')
  const [dragOver, setDragOver] = useState(false)

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
    const source = { sourceName: file.name, sourceType: file.type || 'unknown', sourceBytes: file.size }
    if (file.type.startsWith('image/')) {
      const size = await imageSize(file)
      onChange(file, { kind: 'image', ...source, ...size, sentName: file.name, sentType: file.type, sentBytes: file.size })
      return
    }
    if (file.type.startsWith('video/') || /\.(mp4|mov|webm|mkv|m4v|avi)$/i.test(file.name)) {
      setBusy(true)
      try {
        const { still, width, height, durationSec, frameAtSec } = await extractVideoThumbnail(file)
        onChange(still, {
          kind: 'video',
          ...source,
          width,
          height,
          durationSec,
          frameAtSec,
          sentName: still.name,
          sentType: still.type,
          sentBytes: still.size,
        })
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
      onChange={(e) => {
        const input = e.currentTarget
        void handleFiles(input.files).finally(() => {
          input.value = ''
        })
      }}
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

  if (card) {
    const facts: string[] = []
    if (media) {
      if (media.kind === 'video') {
        facts.push(`Video · ${formatDuration(media.durationSec ?? 0)} · ${media.width}×${media.height}`)
        facts.push(`Frame at ${formatDuration(media.frameAtSec ?? 0)} (35% into the clip)`)
      } else if (media.width) {
        facts.push(`Image · ${media.width}×${media.height}`)
      }
      facts.push(`Sends ${media.sentType.replace('image/', '').toUpperCase()} · ${formatSize(media.sentBytes)}`)
    }
    return (
      <Box
        onDragOver={(e) => {
          e.preventDefault()
          setDragOver(true)
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragOver(false)
          void handleFiles(e.dataTransfer.files)
        }}
        sx={{
          border: '1px dashed',
          borderStyle: value ? 'solid' : 'dashed',
          borderColor: dragOver ? 'primary.main' : 'divider',
          bgcolor: dragOver ? 'action.hover' : 'transparent',
          borderRadius: '10px',
          p: 1.5,
          transition: 'border-color 120ms, background-color 120ms',
        }}
      >
        {fileInput}
        {value && previewUrl ? (
          <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} sx={{ alignItems: { sm: 'center' } }}>
            <Box
              component="img"
              src={previewUrl}
              alt={media?.kind === 'video' ? 'Frame extracted from the video' : 'Uploaded image'}
              sx={{
                width: 192,
                aspectRatio: '16 / 9',
                objectFit: 'contain',
                bgcolor: '#0b0b0c',
                borderRadius: '8px',
                flexShrink: 0,
              }}
            />
            <Stack spacing={0.5} sx={{ minWidth: 0, flex: 1 }}>
              <Typography sx={{ fontSize: 13, fontWeight: 600 }} noWrap title={media?.sourceName ?? value.name}>
                {media?.sourceName ?? value.name}
              </Typography>
              {facts.map((f) => (
                <Typography key={f} sx={{ fontSize: 12, color: 'text.secondary' }}>
                  {f}
                </Typography>
              ))}
              {media?.kind === 'video' && (
                <Typography sx={{ fontSize: 11.5, color: 'text.secondary', opacity: 0.8 }}>
                  The clip stays in your browser — only this frame is sent.
                </Typography>
              )}
            </Stack>
            <Stack direction="row" spacing={1} sx={{ flexShrink: 0 }}>
              <Button size="small" variant="outlined" disabled={busy} onClick={() => inputRef.current?.click()} sx={{ height: 34, textTransform: 'none' }}>
                Replace
              </Button>
              <IconButton size="small" aria-label="Remove media" onClick={clear} disabled={busy} sx={{ height: 34, width: 34 }}>
                <X size={16} />
              </IconButton>
            </Stack>
          </Stack>
        ) : (
          <Stack
            direction="row"
            spacing={1.5}
            role="button"
            tabIndex={0}
            onClick={() => !busy && inputRef.current?.click()}
            onKeyDown={(e) => {
              if ((e.key === 'Enter' || e.key === ' ') && !busy) inputRef.current?.click()
            }}
            sx={{ alignItems: 'center', cursor: busy ? 'progress' : 'pointer', py: 0.5 }}
          >
            <Box sx={{ width: 40, height: 40, borderRadius: '8px', bgcolor: 'action.hover', display: 'grid', placeItems: 'center', flexShrink: 0 }}>
              {busy ? <CircularProgress size={18} /> : <UploadSimple size={18} />}
            </Box>
            <Box sx={{ minWidth: 0 }}>
              <Typography sx={{ fontSize: 13, fontWeight: 600 }}>
                {busy ? 'Extracting a frame…' : 'Add an image or video'}
              </Typography>
              <Typography sx={{ fontSize: 12, color: 'text.secondary' }}>
                {hint ?? 'Drop a file here or click to browse. A video stays in your browser; one frame is sent.'}
              </Typography>
            </Box>
          </Stack>
        )}
        {localError && <Typography sx={{ fontSize: 11.5, color: 'error.main', mt: 1 }}>{localError}</Typography>}
      </Box>
    )
  }

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

function formatDuration(sec: number): string {
  const s = Math.round(sec)
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}

function formatSize(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / 1024 / 1024).toFixed(2)} MB`
}

async function imageSize(file: File): Promise<{ width?: number; height?: number }> {
  try {
    const bitmap = await createImageBitmap(file)
    const size = { width: bitmap.width, height: bitmap.height }
    bitmap.close()
    return size
  } catch {
    return {}
  }
}

/** Seek ~35% into the clip and encode one JPEG still. */
export function extractVideoThumbnail(
  file: File,
  atRatio = 0.35,
): Promise<{ still: File; width: number; height: number; durationSec: number; frameAtSec: number }> {
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
            resolve({
              still: new File([blob], `${base}_thumb.jpg`, { type: 'image/jpeg' }),
              width: w,
              height: h,
              durationSec: Number.isFinite(video.duration) ? video.duration : 0,
              frameAtSec: video.currentTime,
            })
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
