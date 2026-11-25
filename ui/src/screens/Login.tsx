// [Fact-Forcing preamble]
//
// Callers: `ui/src/App.tsx` renders this screen when AuthProvider's state is `anonymous`.
//
// No existing file serves the same purpose: no login screen existed before P0.
//
// Data schemas: calls `useAuth().login(username, password)` which POSTs to /v1/auth/login.
//
// No secrets in plaintext: password lives in component state only while typing; submitted
// over POST, never persisted. Browser manages the resulting session cookie.
//
// The user's verbatim instruction: "create auth or required things because maybe user have
// different workflow and they can create difference judgments".

import { Box, Button, Stack, TextField, Typography } from '@mui/material'
import { useState, type FormEvent } from 'react'
import { useAuth } from '../auth'
import { describeError } from '../api'
import { PALETTE } from '../theme'

/**
 * Centered login card. Deliberately spartan — no "forgot password" link because P0 has no
 * self-service reset (admin reissues via the bootstrap CLI).
 */
export function Login() {
  const { login } = useAuth()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!username.trim() || !password) return
    setSubmitting(true)
    setError('')
    try {
      await login(username.trim(), password)
    } catch (err) {
      setError(describeError(err))
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Box
      sx={{
        minHeight: '100dvh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        bgcolor: PALETTE.canvas,
        px: 2,
      }}
    >
      <Box
        component="form"
        onSubmit={onSubmit}
        noValidate
        sx={{
          width: '100%',
          maxWidth: 384,
          bgcolor: PALETTE.surface,
          borderRadius: '14px',
          border: `1px solid ${PALETTE.hairline}`,
          px: 4,
          py: 4,
          boxShadow: '0 8px 24px rgba(0,0,0,0.04)',
        }}
      >
        <Stack direction="row" spacing={1.5} sx={{ alignItems: 'center', mb: 2.5 }}>
          <Box
            sx={{
              width: 32,
              height: 32,
              borderRadius: '9px',
              bgcolor: PALETTE.brand,
              color: PALETTE.onBrand,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 15,
              fontWeight: 800,
            }}
            aria-hidden
          >
            mR
          </Box>
          <Typography sx={{ fontSize: 18, fontWeight: 700, letterSpacing: '-0.015em', color: PALETTE.text }}>
            Sign in to mRouter
          </Typography>
        </Stack>
        <Typography sx={{ fontSize: 13, color: PALETTE.textMuted, mb: 3 }}>
          Enter the credentials your admin created for you. Each operator records their own
          judgments and ratings under their own name.
        </Typography>
        <Stack spacing={2}>
          <TextField
            label="Username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            autoComplete="username"
            autoFocus
            fullWidth
            size="small"
            required
          />
          <TextField
            label="Password"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="current-password"
            fullWidth
            size="small"
            required
          />
          {error && (
            <Typography role="alert" sx={{ fontSize: 12.5, color: PALETTE.criticalText, fontWeight: 500 }}>
              {error}
            </Typography>
          )}
          <Button
            type="submit"
            variant="contained"
            color="primary"
            disabled={submitting || !username.trim() || !password}
            sx={{ mt: 1 }}
          >
            {submitting ? 'Signing in…' : 'Sign in'}
          </Button>
        </Stack>
        <Typography sx={{ fontSize: 11.5, color: PALETTE.textFaint, mt: 3, textAlign: 'center' }}>
          Don’t have an account? Ask an admin to create one for you.
        </Typography>
      </Box>
    </Box>
  )
}
