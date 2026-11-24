// [Fact-Forcing preamble]
//
// Callers: `ui/src/App.tsx` routes here when active === 'users'; the Nav entry that leads
// here is only rendered when the signed-in user has role === 'admin'.
//
// No existing file serves the same purpose: no users admin screen existed before P0.
//
// Data schemas: calls `api.listUsers()`, `api.createUser({username, password, role, notes})`,
// `api.updateUser(id, {disabled})`. Types `User`, `UserRole` from `types.ts`.
//
// No secrets in plaintext: the create-user form holds the new user's password in state only
// for the duration of the submit (same pattern as Login.tsx). Cleared on success.
//
// The user's verbatim instruction: "create auth or required things because maybe user have
// different workflow and they can create difference judgments" — PLAN-v2.md §4.

import { Plus } from '@phosphor-icons/react'
import {
  Alert,
  Box,
  Button,
  Chip,
  MenuItem,
  Stack,
  Switch,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from '@mui/material'
import { useEffect, useState, type FormEvent } from 'react'
import { api, describeError } from '../api'
import { PageHead, Panel } from '../components/Primitives'
import { PALETTE } from '../theme'
import type { User, UserRole } from '../types'

/**
 * Admin-only operator management. Create users, toggle disabled, see who exists. No
 * password reset or delete here — PLAN-v2.md §4.6 keeps reset on the CLI and defers
 * deletion to a later phase (so historical judgments stay attributable).
 */
export function Users() {
  const [users, setUsers] = useState<User[] | null>(null)
  const [error, setError] = useState('')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [role, setRole] = useState<UserRole>('member')
  const [notes, setNotes] = useState('')
  const [submitting, setSubmitting] = useState(false)

  async function refresh() {
    try {
      const resp = await api.listUsers()
      setUsers(resp.users)
    } catch (err) {
      setError(describeError(err))
      setUsers([])
    }
  }

  useEffect(() => {
    void refresh()
  }, [])

  async function onCreate(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!username.trim() || !password) return
    setSubmitting(true)
    setError('')
    try {
      await api.createUser({ username: username.trim(), password, role, notes: notes.trim() })
      setUsername('')
      setPassword('')
      setNotes('')
      setRole('member')
      await refresh()
    } catch (err) {
      setError(describeError(err))
    } finally {
      setSubmitting(false)
    }
  }

  async function onToggleDisabled(user: User) {
    setError('')
    try {
      await api.updateUser(user.id, { disabled: !user.disabled })
      await refresh()
    } catch (err) {
      setError(describeError(err))
    }
  }

  return (
    <Box>
      <PageHead
        title="Operators"
        lede="Who can sign in, who can approve catalog tiers, and who gets attribution on every rating and judgment."
      />
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Stack direction={{ xs: 'column', md: 'row' }} spacing={2.5} sx={{ alignItems: 'stretch' }}>
        <Box sx={{ flex: 1, minWidth: 0 }}>
        <Panel title="Current operators">
          {users === null ? (
            <Typography sx={{ fontSize: 13, color: PALETTE.textMuted }}>Loading…</Typography>
          ) : users.length === 0 ? (
            <Typography sx={{ fontSize: 13, color: PALETTE.textMuted }}>
              No operators yet. Create the first one with the form on the right.
            </Typography>
          ) : (
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell sx={{ fontWeight: 600, fontSize: 12 }}>Username</TableCell>
                  <TableCell sx={{ fontWeight: 600, fontSize: 12 }}>Role</TableCell>
                  <TableCell sx={{ fontWeight: 600, fontSize: 12 }}>Notes</TableCell>
                  <TableCell sx={{ fontWeight: 600, fontSize: 12, textAlign: 'right' }}>Active</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {users.map((u) => (
                  <TableRow key={u.id} hover>
                    <TableCell sx={{ fontSize: 13, fontWeight: 600, color: PALETTE.text }}>{u.username}</TableCell>
                    <TableCell>
                      <Chip
                        size="small"
                        label={u.role}
                        sx={{
                          height: 22,
                          fontSize: 11.5,
                          fontWeight: 600,
                          bgcolor: u.role === 'admin' ? PALETTE.brandSoft : PALETTE.raised,
                          color: u.role === 'admin' ? PALETTE.brand : PALETTE.textMuted,
                        }}
                      />
                    </TableCell>
                    <TableCell sx={{ fontSize: 12.5, color: PALETTE.textMuted }}>{u.notes || '—'}</TableCell>
                    <TableCell sx={{ textAlign: 'right' }}>
                      <Switch
                        size="small"
                        checked={!u.disabled}
                        onChange={() => void onToggleDisabled(u)}
                        slotProps={{
                          input: { 'aria-label': `${u.disabled ? 'Enable' : 'Disable'} ${u.username}` },
                        }}
                      />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </Panel>
        </Box>

        <Box sx={{ width: { xs: '100%', md: 340 }, flexShrink: 0 }}>
        <Panel title="Invite an operator">
          {/* autoComplete off + new-password: browsers otherwise paste the signed-in
              operator's login into this "create someone else" form. */}
          <Box component="form" onSubmit={onCreate} noValidate autoComplete="off">
            <Stack spacing={2}>
              <TextField
                label="Username"
                name="new-operator-username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                size="small"
                required
                autoComplete="off"
                slotProps={{
                  htmlInput: { autoComplete: 'off', 'data-1p-ignore': true, 'data-lpignore': 'true' },
                }}
              />
              <TextField
                label="Password"
                name="new-operator-password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                size="small"
                required
                autoComplete="new-password"
                slotProps={{
                  htmlInput: { autoComplete: 'new-password', 'data-1p-ignore': true, 'data-lpignore': 'true' },
                }}
                helperText="Share this password once. The operator can change it later with an admin."
              />
              <TextField
                select
                label="Role"
                value={role}
                onChange={(e) => setRole(e.target.value as UserRole)}
                size="small"
              >
                <MenuItem value="member">member</MenuItem>
                <MenuItem value="admin">admin</MenuItem>
              </TextField>
              <TextField
                label="Notes (optional)"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                size="small"
                multiline
                rows={2}
              />
              <Button
                type="submit"
                variant="contained"
                color="primary"
                startIcon={<Plus size={14} />}
                disabled={submitting || !username.trim() || !password}
              >
                {submitting ? 'Creating…' : 'Create operator'}
              </Button>
            </Stack>
          </Box>
        </Panel>
        </Box>
      </Stack>
    </Box>
  )
}
