// [Fact-Forcing preamble]
//
// Callers: `ui/src/App.tsx` wraps the whole app in `AuthProvider` and reads `useUser()` for
// the top-bar user menu; `ui/src/Nav.tsx` reads `useUser()` to decide whether to show the
// admin-only `Users` nav entry; `ui/src/screens/Users.tsx` reads it for role gating.
//
// No existing file serves the same purpose: no auth context existed in the UI before P0.
//
// Data schemas: reads the `User` type from `types.ts`; calls `api.me()`, `api.login()`,
// `api.logout()` from `api.ts`. State held in memory only — no localStorage; the browser's
// cookie jar is the source of truth for the session.
//
// No secrets in plaintext: the password passes through `login()` to the server once and is
// not retained. The provider holds only the server's public `User` record (no hash).
//
// The user's verbatim instruction: "create auth or required things because maybe user have
// different workflow and they can create difference judgments".

import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { ApiError, api } from './api'
import type { User } from './types'

type AuthState =
  | { status: 'loading'; user: null }
  | { status: 'anonymous'; user: null }
  | { status: 'authenticated'; user: User }

interface AuthContextValue {
  state: AuthState
  login: (username: string, password: string) => Promise<void>
  logout: () => Promise<void>
  refresh: () => Promise<void>
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>({ status: 'loading', user: null })

  const refresh = useCallback(async () => {
    try {
      const { user } = await api.me()
      setState({ status: 'authenticated', user })
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        setState({ status: 'anonymous', user: null })
      } else {
        // Backend unreachable: leave the previous state alone so a transient blip does not
        // kick a signed-in user out. The shell's /healthz check reports connectivity.
        setState((prev) => (prev.status === 'loading' ? { status: 'anonymous', user: null } : prev))
      }
    }
  }, [])

  const login = useCallback(async (username: string, password: string) => {
    const { user } = await api.login(username, password)
    setState({ status: 'authenticated', user })
  }, [])

  const logout = useCallback(async () => {
    try {
      await api.logout()
    } finally {
      // Even if the server rejected the logout (e.g. session already gone), move the UI to
      // anonymous so the shell renders Login and doesn't spin forever.
      setState({ status: 'anonymous', user: null })
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const value = useMemo<AuthContextValue>(() => ({ state, login, logout, refresh }), [state, login, logout, refresh])
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext)
  if (ctx === null) {
    throw new Error('useAuth() must be used inside <AuthProvider>')
  }
  return ctx
}

/** Returns the signed-in user, or null when loading/anonymous. Screens that are only
 *  rendered after the provider has resolved can treat null as a bug. */
export function useUser(): User | null {
  const { state } = useAuth()
  return state.status === 'authenticated' ? state.user : null
}

export function isUnauthorized(err: unknown): err is ApiError {
  return err instanceof ApiError && err.status === 401
}
